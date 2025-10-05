#!/usr/bin/env python3
"""
wm_pipeline_gpu.py
GPU-accelerated watermark pipeline (DCT-based video + dual audio watermark).
- embed: embeds session ID into frames (DCT bits) and audio (echo + spectrum spread)
- detect: attempts to recover session ID from video+audio with alignment heuristics
- simulate: creates several pirated variants (scale/bitrate) and reports stats

Author: Generated for user (adapted for CUDA PyTorch)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import math
import glob
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# GPU/torch
import torch

# images/audio
import cv2
import numpy as np
from reedsolo import RSCodec

# audio
import librosa
import soundfile as sf

# ============ CONFIG =============
# High strength by default (user selected C)
EMBED_STRENGTH = 4.0    # high strength (tune if visible)
EMBED_MARGIN = 0.6
RS_PARITY = 16           # parity bytes
BLOCK_SIZE = 8
BLOCKS_PER_FRAME = 400
REPEAT_EVERY = 10
SYNC = "SYNC-01"

# audio config
AUDIO_ECHO_DEPTH = 0.002
AUDIO_ECHO_DELAY_MS = 30
AUDIO_SPECTRUM_MAG_CHANGE = 0.004  # small magnitude change (inaudible)
# output default filename
DEFAULT_OUTPUT_VIDEO = "final_coded_video.mp4"
# ==================================

# Check CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------ UTIL / FFmpeg wrappers -------------
def run(cmd, check=True):
    # helper to run shell commands
    # print(cmd)  # debug
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\nSTDOUT: {res.stdout.decode()}\nSTDERR: {res.stderr.decode()}")
    return res


def ffprobe_get_fps_and_duration(input_file):
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,duration -of default=noprint_wrappers=1 "{input_file}"'
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = res.stdout.decode()
    fps = 25.0
    duration = None
    for line in out.splitlines():
        if line.startswith("r_frame_rate="):
            val = line.split("=", 1)[1].strip()
            # val like "30000/1001" or "25/1"
            try:
                if "/" in val:
                    a, b = val.split("/")
                    fps = float(a) / float(b)
                else:
                    fps = float(val)
            except:
                fps = 25.0
        if line.startswith("duration="):
            duration = float(line.split("=", 1)[1].strip())
    return fps, duration


def extract_frames_ffmpeg(input_vid, out_dir, fps=None):
    os.makedirs(out_dir, exist_ok=True)
    if fps is None:
        fps, _ = ffprobe_get_fps_and_duration(input_vid)
    # ffmpeg -i input -vf fps=FPS out_dir/frame_%05d.jpg
    cmd = f'ffmpeg -y -i "{input_vid}" -vf fps={fps} "{out_dir}{os.sep}frame_%05d.jpg"'
    run(cmd)
    return sorted(glob.glob(os.path.join(out_dir, "frame_*.jpg")))


def extract_audio_ffmpeg(input_vid, out_wav):
    # Extract audio to wav with pcm16
    cmd = f'ffmpeg -y -i "{input_vid}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{out_wav}"'
    run(cmd)
    return out_wav


def reassemble_video_ffmpeg(frames_dir, audio_file, out_video, fps=25, crf=18):
    # frames_dir should contain frame_%05d.jpg
    frames_pattern = os.path.join(frames_dir, "frame_%05d.jpg")
    if os.path.exists(audio_file):
        cmd = f'ffmpeg -y -framerate {fps} -i "{frames_pattern}" -i "{audio_file}" -c:v libx264 -preset medium -crf {crf} -c:a aac -b:a 192k -shortest "{out_video}"'
    else:
        cmd = f'ffmpeg -y -framerate {fps} -i "{frames_pattern}" -c:v libx264 -preset medium -crf {crf} "{out_video}"'
    run(cmd)
    return out_video


# ---------- Reed-Solomon payload ----------
def rs_encode(session_str: str, parity=RS_PARITY):
    rsc = RSCodec(parity)
    payload = (SYNC + "|" + session_str).encode("utf-8")
    coded = rsc.encode(payload)
    bits = "".join(format(b, "08b") for b in coded)
    return bits, len(coded)


def rs_try_decode_from_bytes(bts, parity=RS_PARITY):
    rsc = RSCodec(parity)
    try:
        dec = rsc.decode(bytes(bts))
        txt = dec.decode("utf-8", errors="ignore")
        if "|" in txt:
            sync, session = txt.split("|", 1)
            if sync == SYNC:
                return session
    except Exception:
        pass
    return None


# --------- DCT utilities using torch (GPU) ----------
# We'll create a DCT basis matrix for 8x8 blocks, apply by matrix multiplication on device.
def make_dct_basis(N=8, device=DEVICE):
    # DCT-II basis NxN
    n = torch.arange(N, dtype=torch.float32, device=device)
    k = n.view(N, 1)
    # Convert the constant to tensor before sqrt
    alpha = torch.sqrt(torch.tensor(2.0 / N, dtype=torch.float32, device=device)) * torch.ones(N, device=device)
    alpha[0] = torch.sqrt(torch.tensor(1.0 / N, dtype=torch.float32, device=device))
    basis = alpha.view(N, 1) * torch.cos(math.pi * (2 * n + 1).view(N, 1) * k.view(1, N) / (2.0 * N))
    # Basis shape (N, N) for transform: X = B @ x @ B.T
    return basis


# Precompute basis on device
DCT_B = make_dct_basis(BLOCK_SIZE, device=DEVICE)
DCT_BT = DCT_B.t()


def dct2_torch(blocks_tensor):
    # blocks_tensor shape: (B, 8, 8) (float32)
    B = blocks_tensor.shape[0]
    # apply: D * block * D^T
    return DCT_B @ blocks_tensor @ DCT_BT


def idct2_torch(dct_blocks):
    # inverse using transpose properties (D^T * X * D)
    return DCT_BT @ dct_blocks @ DCT_B


# ---------- embed/extract per-frame ----------
def select_textured_positions(gray_np, max_blocks):
    # receives numpy gray image
    h, w = gray_np.shape
    h8, w8 = h - (h % BLOCK_SIZE), w - (w % BLOCK_SIZE)
    variances = []
    positions = []
    for by in range(0, h8, BLOCK_SIZE):
        for bx in range(0, w8, BLOCK_SIZE):
            blk = gray_np[by:by+BLOCK_SIZE, bx:bx+BLOCK_SIZE].astype(np.float32)
            variances.append(np.var(blk))
    if len(variances) == 0:
        return [], h8, w8
    idx_sorted = np.argsort(-np.array(variances))
    return idx_sorted[:min(max_blocks, len(idx_sorted))].tolist(), h8, w8


def seeded_positions(secret, frame_index, total_blocks, count):
    seed = hashlib.sha256((secret + str(frame_index)).encode("utf-8")).digest()
    out = []
    i = 0
    while len(out) < count:
        h = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        for k in range(0, len(h), 4):
            val = int.from_bytes(h[k:k+4], "big")
            out.append(val % total_blocks)
            if len(out) >= count:
                break
        i += 1
    return out[:count]


def embed_in_frame_torch(frame_bgr, coded_bits, frame_index, secret,
                         blocks=BLOCKS_PER_FRAME, strength=EMBED_STRENGTH, margin=EMBED_MARGIN,
                         repeat=REPEAT_EVERY):
    # frame_bgr: np.uint8 HxWx3
    h, w = frame_bgr.shape[:2]
    h8, w8 = h - (h % BLOCK_SIZE), w - (w % BLOCK_SIZE)
    frame = frame_bgr[:h8, :w8].copy()
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y = ycbcr[:, :, 0].astype(np.float32)
    textured, h8, w8 = select_textured_positions(y, blocks)
    total_blocks = (h8 // BLOCK_SIZE) * (w8 // BLOCK_SIZE)
    if len(textured) < blocks:
        seeded = seeded_positions(secret, frame_index, total_blocks, blocks - len(textured))
        positions = list(textured) + list(seeded)
    else:
        positions = textured[:blocks]

    # Prepare block indices
    blocks_coords = []
    stride_w = w8 // BLOCK_SIZE
    for linear_pos in positions:
        by = (linear_pos // stride_w) * BLOCK_SIZE
        bx = (linear_pos % stride_w) * BLOCK_SIZE
        blocks_coords.append((by, bx))

    L = len(coded_bits)
    bits_per_frame = blocks
    base_ptr = (frame_index % repeat) * bits_per_frame

    # Collect all blocks to GPU tensor
    blk_list = []
    for (by, bx) in blocks_coords:
        blk = y[by:by+BLOCK_SIZE, bx:bx+BLOCK_SIZE]
        blk_list.append(blk)
    if not blk_list:
        return frame_bgr  # nothing

    block_stack = np.stack(blk_list, axis=0).astype(np.float32)  # (B,8,8)
    t_blocks = torch.from_numpy(block_stack).to(DEVICE)  # float32 on GPU

    # compute dct
    t_d = dct2_torch(t_blocks)  # (B,8,8)

    # positions to modify, use coefficient positions (3,2) and (2,3)
    c1 = (3, 2)
    c2 = (2, 3)
    for k in range(t_d.shape[0]):
        bit = coded_bits[(base_ptr + k) % L]
        val1 = float(t_d[k, c1[0], c1[1]].item())
        val2 = float(t_d[k, c2[0], c2[1]].item())
        diff = val1 - val2
        target = 1.0 if bit == '1' else -1.0
        if math.copysign(1.0, diff) == target:
            if abs(diff) < margin:
                adjust = max(strength, margin - abs(diff))
                t_d[k, c1[0], c1[1]] += target * adjust
        else:
            # swap and push
            temp = t_d[k, c1[0], c1[1]].clone()
            t_d[k, c1[0], c1[1]] = t_d[k, c2[0], c2[1]]
            t_d[k, c2[0], c2[1]] = temp
            t_d[k, c1[0], c1[1]] += target * (abs(diff) + strength + margin)

    # inverse dct
    t_idct = idct2_torch(t_d)
    # clamp and copy back
    out_blocks = t_idct.cpu().numpy()
    for i, (by, bx) in enumerate(blocks_coords):
        patch = np.clip(np.round(out_blocks[i]), 0, 255).astype(np.uint8)
        # only replace Y channel
        y[by:by+BLOCK_SIZE, bx:bx+BLOCK_SIZE] = patch

    ycbcr[:, :, 0] = y.astype(np.uint8)
    bgr_out = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)
    # place into full sized frame
    out_full = frame_bgr.copy()
    out_full[:h8, :w8] = bgr_out
    return out_full


def extract_soft_values_frame_torch(frame_bgr, secret, blocks=BLOCKS_PER_FRAME):
    h, w = frame_bgr.shape[:2]
    h8, w8 = h - (h % BLOCK_SIZE), w - (w % BLOCK_SIZE)
    frame = frame_bgr[:h8, :w8].copy()
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y = ycbcr[:, :, 0].astype(np.float32)
    textured, h8, w8 = select_textured_positions(y, blocks)
    total_blocks = (h8 // BLOCK_SIZE) * (w8 // BLOCK_SIZE)
    if len(textured) < blocks:
        positions = textured + seeded_positions(secret, 0, total_blocks, blocks - len(textured))
    else:
        positions = textured[:blocks]
    # collect blocks
    blk_list = []
    stride_w = w8 // BLOCK_SIZE
    for linear_pos in positions:
        by = (linear_pos // stride_w) * BLOCK_SIZE
        bx = (linear_pos % stride_w) * BLOCK_SIZE
        blk = y[by:by+BLOCK_SIZE, bx:bx+BLOCK_SIZE]
        blk_list.append(blk)
    if not blk_list:
        return []
    block_stack = np.stack(blk_list, axis=0).astype(np.float32)
    t_blocks = torch.from_numpy(block_stack).to(DEVICE)
    t_d = dct2_torch(t_blocks)
    c1 = (3, 2)
    c2 = (2, 3)
    diffs = (t_d[:, c1[0], c1[1]] - t_d[:, c2[0], c2[1]]).cpu().numpy().tolist()
    return diffs


# -------- audio watermark functions ----------
def audio_embed_echo(y, sr, session, depth=AUDIO_ECHO_DEPTH, delay_ms=AUDIO_ECHO_DELAY_MS):
    bits = "".join(format(b, "08b") for b in session.encode("utf-8"))
    out = y.copy()
    delay = int(sr * (delay_ms / 1000.0))
    for i, b in enumerate(bits):
        start = i * delay * 2
        if start >= len(y):
            break
        if b == "1":
            end = min(len(y), start + delay)
            out[start:end] += y[:end - start] * depth
    return out


def audio_embed_spectrum(y, sr, session, mag_change=AUDIO_SPECTRUM_MAG_CHANGE):
    # simple magnitude perturbation across STFT bins to hide bits
    # We'll use short-time Fourier transform (librosa)
    bits = "".join(format(b, "08b") for b in session.encode("utf-8"))
    # mono
    if y.ndim > 1:
        y_mono = librosa.to_mono(y.T)
    else:
        y_mono = y
    S = librosa.stft(y_mono, n_fft=1024, hop_length=512)
    mag, ph = np.abs(S), np.angle(S)
    n_bins, n_frames = mag.shape
    # embed across time by selecting frames spaced evenly
    frames_needed = len(bits)
    step = max(1, n_frames // (frames_needed + 1))
    for i, b in enumerate(bits):
        frame_idx = i * step
        if frame_idx >= n_frames:
            break
        # modify high-frequency bins slightly
        bins = slice(int(n_bins * 0.7), n_bins)
        if b == "1":
            mag[bins, frame_idx] *= (1.0 + mag_change)
        else:
            mag[bins, frame_idx] *= (1.0 - mag_change)
    S2 = mag * np.exp(1j * ph)
    y_rec = librosa.istft(S2, hop_length=512)
    # pad/trim to original
    if len(y_rec) < len(y_mono):
        y_rec = np.pad(y_rec, (0, len(y_mono) - len(y_rec)))
    y_out = y_rec[:len(y_mono)]
    # If original was stereo, duplicate
    if y.ndim > 1:
        y_out_full = np.vstack([y_out, y_out]).T
    else:
        y_out_full = y_out
    return y_out_full


def audio_extract_echo(y, sr, delay_ms=AUDIO_ECHO_DELAY_MS, max_chars=64, debug=False):
    """
    Enhanced echo-based audio watermark extraction.
    Handles stereo, volume normalization, small drift, and adaptive thresholds.

    Args:
        y (np.ndarray): audio signal (mono or stereo)
        sr (int): sample rate
        delay_ms (float): delay used in watermarking (milliseconds)
        max_chars (int): maximum number of characters to decode
        debug (bool): if True, prints detailed correlation values
    Returns:
        str: decoded text (best-effort)
    """

    # ---- 1️⃣ Normalize & handle stereo ----
    if y.ndim > 1:
        y = librosa.to_mono(y.T)
    y = y.astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-9)  # normalize

    # ---- 2️⃣ Initialize variables ----
    delay = int(sr * (delay_ms / 1000.0))
    bits = []
    corr_values = []

    # ---- 3️⃣ Iterate over bit segments ----
    for i in range(max_chars * 8):
        start = i * delay * 2
        if start + delay >= len(y):
            break

        orig = y[start:start + delay]
        delayed = y[start + delay:start + 2 * delay]

        L = min(len(orig), len(delayed))
        if L < 100:  # very short
            break

        o = orig[:L]
        d = delayed[:L]

        # ---- 4️⃣ Windowed cross-correlation ----
        win = np.hanning(L)
        o_win = o * win
        d_win = d * win
        corr = np.dot(o_win, d_win) / (np.linalg.norm(o_win) * np.linalg.norm(d_win) + 1e-9)
        corr_values.append(corr)

    # ---- 5️⃣ Adaptive thresholding ----
    if not corr_values:
        return ""

    mean_corr = np.mean(corr_values)
    std_corr = np.std(corr_values)
    thresh = mean_corr + 0.5 * std_corr  # dynamic threshold

    if debug:
        print(f"[audio_extract_echo] Adaptive threshold: {thresh:.4f} (mean={mean_corr:.4f}, std={std_corr:.4f})")

    # Convert correlation values to bits
    for corr in corr_values:
        bits.append("1" if corr > thresh else "0")

    # ---- 6️⃣ Convert bits to characters ----
    chs = []
    for j in range(0, len(bits), 8):
        byte = bits[j:j + 8]
        if len(byte) < 8:
            break
        try:
            ch = chr(int("".join(byte), 2))
            chs.append(ch)
        except Exception:
            pass

    decoded = "".join(chs)

    # ---- 7️⃣ Optional Debug Output ----
    if debug:
        print(f"[audio_extract_echo] Decoded: {decoded}")
        print(f"[audio_extract_echo] Avg corr={np.mean(corr_values):.4f}, bits={len(bits)}")

    return decoded

    for i in range(max_chars * 8):
        start = i * delay * 2
        if start + delay >= len(y):
            break
        orig = y[start:start + delay]
        delayed = y[start + delay:start + 2 * delay]
        if len(orig) == 0 or len(delayed) == 0:
            break
        L = min(len(orig), len(delayed))
if L < 50:
    break  # too short to be meaningful
corr = np.dot(orig[:L], delayed[:L]) / (
    np.linalg.norm(orig[:L]) * np.linalg.norm(delayed[:L]) + 1e-9
)

        bits.append("1" if corr > 0.02 else "0")
    chs = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i + 8]
        if len(byte) < 8:
            break
        chs.append(chr(int("".join(byte), 2)))
    return "".join(chs)


def audio_extract_spectrum(y, sr, max_chars=64):
    if y.ndim > 1:
        y_mono = librosa.to_mono(y.T)
    else:
        y_mono = y
    S = librosa.stft(y_mono, n_fft=1024, hop_length=512)
    mag = np.abs(S)
    n_bins, n_frames = mag.shape
    # heuristic: examine high-bin energy variation across frames and infer bits
    bits = []
    frames_needed = min(max_chars * 8, n_frames)
    step = max(1, n_frames // (frames_needed + 1))
    for i in range(frames_needed):
        frame_idx = i * step
        if frame_idx >= n_frames:
            break
        hf = mag[int(n_bins * 0.7):, frame_idx]
        avg = float(np.mean(hf))
        # threshold relative to global median
        med = float(np.median(mag[int(n_bins * 0.7):, :]))
        bits.append("1" if avg > med else "0")
    chs = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i + 8]
        if len(byte) < 8:
            break
        chs.append(chr(int("".join(byte), 2)))
    return "".join(chs)


# ---------- FRAME ALIGNMENT heuristics ----------
def phase_correlation_align(ref_gray, img_gray):
    # use cv2.phaseCorrelate with Hanning window; returns (dx, dy)
    # create float32
    reff = np.float32(ref_gray)
    imgf = np.float32(img_gray)
    # pad/crop to same shape
    if reff.shape != imgf.shape:
        h = min(reff.shape[0], imgf.shape[0])
        w = min(reff.shape[1], imgf.shape[1])
        reff = reff[:h, :w]
        imgf = imgf[:h, :w]
    # use Hanning window
    win = cv2.createHanningWindow(reff.shape[::-1], cv2.CV_32F)
    shift = cv2.phaseCorrelate(reff * win, imgf * win)
    # returns (dx, dy)
    return shift[0]


def align_frame_to_reference(img_bgr, ref_bgr):
    # Try small scale adjustments and compute best alignment using phase correlation.
    # If a scale is detected (best correlation improvement), return the transformed frame.
    # scales to try
    scales = [0.95, 0.975, 1.0, 1.025, 1.05]
    best_score = -1e9
    best_img = img_bgr
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    for s in scales:
        if s == 1.0:
            cand = img_bgr
        else:
            cand = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        # crop or pad cand to ref size
        ch, cw = cand.shape[:2]
        rh, rw = ref_gray.shape[:2]
        if ch < rh or cw < rw:
            # pad
            padded = np.zeros((rh, rw, 3), dtype=cand.dtype)
            padded[:ch, :cw] = cand
            cand2 = padded
        else:
            cand2 = cand[:rh, :rw]
        cand_gray = cv2.cvtColor(cand2, cv2.COLOR_BGR2GRAY)
        dx, dy = phase_correlation_align(ref_gray, cand_gray)
        # score: negative of translation magnitude (prefers closer alignment)
        score = -math.hypot(dx, dy)
        if score > best_score:
            best_score = score
            best_img = cand2
    return best_img


# ---------- CLI operations ----------
def cmd_embed(args):
    input_vid = args.input
    session = args.session
    secret = args.secret
    out_video = args.output or DEFAULT_OUTPUT_VIDEO

    # temp directories
    tmp = tempfile.mkdtemp(prefix="wm_embed_")
    frames_in = os.path.join(tmp, "frames_in")
    frames_out = os.path.join(tmp, "frames_out")
    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    print("✅ Extracting frames...")
    fps, _ = ffprobe_get_fps_and_duration(input_vid)
    extract_frames_ffmpeg(input_vid, frames_in, fps=fps)
    frames = sorted(glob.glob(os.path.join(frames_in, "frame_*.jpg")))
    if not frames:
        raise RuntimeError("No frames extracted.")

    # audio extract
    audio_file = os.path.join(tmp, "audio.wav")
    try:
        extract_audio_ffmpeg(input_vid, audio_file)
        audio_exists = True
    except Exception:
        audio_exists = False

    # prepare payload bits
    coded_bits, coded_len = rs_encode(session, parity=RS_PARITY)
    print(f"[embed] Session ID bits: {len(coded_bits)} bits (coded bytes={coded_len})")

    # iterate frames and embed using GPU torch
    print("Embedding watermark:", flush=True)
    for i, f in enumerate(tqdm(frames)):
        img = cv2.imread(f)
        out_img = embed_in_frame_torch(img, coded_bits, i, secret,
                                       blocks=args.blocks or BLOCKS_PER_FRAME,
                                       strength=args.strength or EMBED_STRENGTH,
                                       margin=args.margin or EMBED_MARGIN,
                                       repeat=args.repeat or REPEAT_EVERY)
        # save to frames_out with same index name
        name = os.path.basename(f)
        out_path = os.path.join(frames_out, name)
        cv2.imwrite(out_path, out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # audio embedding: load audio and apply both methods if audio exists
    if audio_exists:
        y, sr = librosa.load(audio_file, sr=None, mono=True)
        y_echo = audio_embed_echo(y.copy(), sr, session, depth=AUDIO_ECHO_DEPTH, delay_ms=AUDIO_ECHO_DELAY_MS)
        y_spec = audio_embed_spectrum(y_echo, sr, session, mag_change=AUDIO_SPECTRUM_MAG_CHANGE)
        audio_out = os.path.join(tmp, "audio_wm.wav")
        sf.write(audio_out, y_spec, sr)
        audio_to_use = audio_out
        print("[embed] Audio watermark embedded (echo + spectrum).")
    else:
        audio_to_use = None
        print("[embed] No audio track found; skipping audio watermarking.")

    # reassemble
    print("🎬 Reassembling watermarked video...")
    reassemble_video_ffmpeg(frames_out, audio_to_use, out_video, fps=fps)
    print("✅ Done. Output:", out_video)

    # cleanup
    shutil.rmtree(tmp, ignore_errors=True)


def cmd_detect(args):
    input_vid = args.input
    secret = args.secret
    session = args.session
    frames_limit = args.frames or 600

    tmp = tempfile.mkdtemp(prefix="wm_detect_")
    frames_dir = os.path.join(tmp, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    print("✅ Extracting frames for detection...")
    fps, duration = ffprobe_get_fps_and_duration(input_vid)
    extract_frames_ffmpeg(input_vid, frames_dir, fps=fps)
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))[:frames_limit]
    if not frames:
        raise RuntimeError("No frames extracted for detection.")

    # attempt audio extraction
    audio_file = os.path.join(tmp, "audio_detect.wav")
    audio_exists = False
    try:
        extract_audio_ffmpeg(input_vid, audio_file)
        audio_exists = True
    except Exception:
        audio_exists = False

    # alignment: use first frame as reference
    ref_img = cv2.imread(frames[0])
    collected = defaultdict(list)

    print("Detecting watermark (video)...")
    for i, f in enumerate(tqdm(frames)):
        img = cv2.imread(f)
        # alignment heuristics (small overhead)
        aligned = align_frame_to_reference(img, ref_img)
        soft_vals = extract_soft_values_frame_torch(aligned, secret, blocks=args.blocks or BLOCKS_PER_FRAME)
        # append soft values to collected slots
        for k, v in enumerate(soft_vals):
            collected[k].append(v)

    # infer bits from collected soft values
    L_guess = 4096
    inferred = []
    for i in range(L_guess):
        vals = collected.get(i, [])
        if not vals:
            inferred.append("0")
        else:
            avg = sum(vals) / len(vals)
            inferred.append("1" if avg > 0 else "0")
    bits_str = "".join(inferred)

    # attempt RS decode by scanning alignments and possible byte lengths
    rsc = RSCodec(RS_PARITY)
    found_session = None
    matched = 0
    checked = 0
    # We'll do a rough matching metric: correlate reconstructed bytes to expected RS-coded payload bits
    expected_bits, coded_len = rs_encode(session, parity=RS_PARITY)
    target_bytes = coded_len

    for align in range(8):
        aligned = bits_str[align:]
        # try several byte lengths
        for B in range(4, target_bytes+1):
            bits_needed = B * 8
            if len(aligned) < bits_needed:
                break
            chunk = aligned[:bits_needed]
            ba = bytearray(int(chunk[i:i+8], 2) for i in range(0, bits_needed, 8))
            try:
                dec = rsc.decode(bytes(ba))
                txt = dec.decode("utf-8", errors="ignore")
                if "|" in txt:
                    sync, sess = txt.split("|", 1)
                    if sync == SYNC:
                        found_session = sess
                        break
            except Exception:
                pass
        if found_session:
            break

    # compute match percentage with expected bits
    # compare prefix length to min length
    cmp_len = min(len(bits_str), len(expected_bits))
    match_bits = sum(1 for i in range(cmp_len) if bits_str[i] == expected_bits[i])
    match_pct = 100.0 * match_bits / (len(expected_bits) or 1)

    # audio detection
    audio_candidate = ""
    audio_conf = 0.0
    if audio_exists:
        y, sr = librosa.load(audio_file, sr=None, mono=True)
        cand_echo = audio_extract_echo(y, sr, delay_ms=AUDIO_ECHO_DELAY_MS, max_chars=64)
        cand_spec = audio_extract_spectrum(y, sr, max_chars=64)
        # heuristics: compare similarity to session bytes
        expected_session = session
        # naive similarity scoring
        def str_sim(a, b):
            if not a or not b:
                return 0.0
            # longest common substring ratio
            la, lb = len(a), len(b)
            matches = sum(1 for i in range(min(la, lb)) if a[i] == b[i])
            return 100.0 * matches / max(1, max(la, lb))
        s1 = str_sim(cand_echo, expected_session)
        s2 = str_sim(cand_spec, expected_session)
        # choose best as candidate
        if s1 >= s2:
            audio_candidate = cand_echo
            audio_conf = s1
        else:
            audio_candidate = cand_spec
            audio_conf = s2

    summary = {
        "found_session": found_session,
        "video_match_percent": match_pct,
        "video_matched": match_bits,
        "video_checked": len(expected_bits),
        "audio_candidate": audio_candidate,
        "audio_confidence": audio_conf
    }

    print("==== Detection Summary ====")
    print(json.dumps(summary, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)
    return summary


def cmd_simulate(args):
    """Simulate phone re-encode attacks and check watermark robustness."""
    input_vid = args.input
    session = args.session
    secret = args.secret
    tmp = tempfile.mkdtemp(prefix="wm_sim_")

    variants = [
        {"scale": 1.0, "crf": 23, "suffix": "orig_recode"},
        {"scale": 0.9, "crf": 28, "suffix": "scale90_crf28"},
        {"scale": 0.8, "crf": 30, "suffix": "scale80_crf30"},
        {"scale": 0.75, "crf": 35, "suffix": "scale75_crf35"},
    ]

    results = []
    print("\n🔬 Starting simulation attacks (scale + re-encode tests)...\n")

    for v in variants:
        outp = os.path.join(tmp, f"variant_{v['suffix']}.mp4")
        scale = v["scale"]
        crf = v["crf"]

        # Build FFmpeg scale + compression
        vf = f"scale=iw*{scale}:ih*{scale}" if scale != 1.0 else "null"
        cmd = f'ffmpeg -y -i "{input_vid}" -vf "{vf}" -c:v libx264 -crf {crf} -preset medium -c:a aac -b:a 128k "{outp}"'

        print(f"🎬 Creating variant: {v['suffix']} (scale={scale}, crf={crf})")
        try:
            run(cmd)
        except Exception as e:
            print(f"⚠️ ffmpeg failed for {v['suffix']}: {e}")
            results.append({"variant": v["suffix"], "error": str(e)})
            continue

        # Properly populate args for detection
        class DetectArgs:
            pass

        detect_args = DetectArgs()
        detect_args.input = outp
        detect_args.secret = secret
        detect_args.session = session
        detect_args.frames = getattr(args, "frames", 600)
        detect_args.blocks = getattr(args, "blocks", BLOCKS_PER_FRAME)

        print("🔎 Running detection...")

        try:
            summary = cmd_detect(detect_args)
            if not isinstance(summary, dict):
                summary = {}

            found = summary.get("found_session")
            vmatch = summary.get("video_match_percent", 0)
            amatch = summary.get("audio_confidence", 0)

            print(f"\n📊 Variant: {v['suffix']}")
            if found:
                print(f"✅ Session ID recovered: {found}")
                print(f"🎥 Video match: {vmatch:.2f}%")
                print(f"🎧 Audio confidence: {amatch:.2f}%\n")
            else:
                print(f"⚠️ No session ID recovered.")
                print(f"🎥 Video match: {vmatch:.2f}%")
                print(f"🎧 Audio confidence: {amatch:.2f}%\n")

            results.append({"variant": v["suffix"], "summary": summary})
        except Exception as e:
            print(f"💥 Detection failed for {v['suffix']}: {e}")
            results.append({"variant": v["suffix"], "error": str(e)})

    print("=== 🧾 FINAL ROBUSTNESS REPORT ===")
    print(json.dumps(results, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)



# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated watermark pipeline (video DCT + audio echo/spectrum)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pE = sub.add_parser("embed", help="Embed watermark into video (both video+audio).")
    pE.add_argument("--input", required=True, help="input video file")
    pE.add_argument("--session", required=True, help="session id string")
    pE.add_argument("--secret", default="top_secret_key", help="secret key for PRNG")
    pE.add_argument("--output", default=None, help="output watermarked video filename (default final_coded_video.mp4)")
    pE.add_argument("--blocks", type=int, default=BLOCKS_PER_FRAME)
    pE.add_argument("--strength", type=float, default=EMBED_STRENGTH)
    pE.add_argument("--margin", type=float, default=EMBED_MARGIN)
    pE.add_argument("--repeat", type=int, default=REPEAT_EVERY)

    pD = sub.add_parser("detect", help="Detect watermark from video (video+audio).")
    pD.add_argument("--input", required=True, help="input video (possibly pirated)")
    pD.add_argument("--session", required=True)
    pD.add_argument("--secret", default="top_secret_key")
    pD.add_argument("--frames", type=int, default=600)
    pD.add_argument("--blocks", type=int, default=BLOCKS_PER_FRAME)

    pS = sub.add_parser("simulate", help="Simulate phone re-encodes and test robustness.")
    pS.add_argument("--input", required=True)
    pS.add_argument("--session", required=True)
    pS.add_argument("--secret", default="top_secret_key")

    args = parser.parse_args()
    if args.cmd == "embed":
        # ensure output default
        if not args.output:
            args.output = DEFAULT_OUTPUT_VIDEO
        return cmd_embed(args)
    elif args.cmd == "detect":
        return cmd_detect(args)
    elif args.cmd == "simulate":
        return cmd_simulate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    print(f"[wm_pipeline_gpu] Running on device: {DEVICE} (torch {torch.__version__ if 'torch' in sys.modules else 'unknown'})")
    main()
