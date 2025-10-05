#!/usr/bin/env python3
"""
Advanced Invisible Watermark System
- DCT-based Video Frame Watermark (with Reed-Solomon ECC)
- Audio Echo Watermark
- Parallel CPU processing for embedding/extraction
"""

import os
import argparse
import hashlib
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from reedsolo import RSCodec
import numpy as np
import cv2
from scipy.fftpack import dct, idct
from tqdm import tqdm
import librosa
import soundfile as sf
import glob

# ===== CONFIG =====
RS_PARITY_BYTES = 16
BLOCKS_PER_FRAME = 400
REPEAT_EVERY_FRAMES = 10
EMBED_STRENGTH = 2.5
EMBED_MARGIN = 0.6
SYNC_PATTERN = "SYNC-01"
# ===================

def dct2(x): return dct(dct(x.T, norm='ortho').T, norm='ortho')
def idct2(x): return idct(idct(x.T, norm='ortho').T, norm='ortho')

def rs_encode_bits(session_str: str, parity=RS_PARITY_BYTES):
    rsc = RSCodec(parity)
    payload = (SYNC_PATTERN + "|" + session_str).encode('utf-8')
    coded = rsc.encode(payload)
    bits = ''.join(format(b, '08b') for b in coded)
    return bits, len(coded)

def seeded_positions(secret: str, frame_index: int, total_blocks: int, count: int):
    seed = hashlib.sha256((secret + str(frame_index)).encode('utf-8')).digest()
    out = []
    i = 0
    while len(out) < count:
        h = hashlib.sha256(seed + i.to_bytes(4, 'big')).digest()
        for k in range(0, len(h), 4):
            val = int.from_bytes(h[k:k+4], 'big')
            out.append(val % total_blocks)
            if len(out) >= count:
                break
        i += 1
    return out[:count]

def select_textured_positions(gray_frame, max_blocks):
    h, w = gray_frame.shape
    h8, w8 = h - (h % 8), w - (w % 8)
    variances = []
    for by in range(0, h8, 8):
        for bx in range(0, w8, 8):
            blk = gray_frame[by:by+8, bx:bx+8].astype(np.float32)
            variances.append(np.var(blk))
    idx_sorted = np.argsort(-np.array(variances))
    return idx_sorted[:min(max_blocks, len(idx_sorted))].tolist(), h8, w8

def embed_bit_in_block(block, bit, strength=EMBED_STRENGTH, margin=EMBED_MARGIN):
    d = dct2(block.astype(np.float32))
    c1, c2 = (3,2), (2,3)
    diff = float(d[c1] - d[c2])
    target = 1 if bit == '1' else -1
    if np.sign(diff) == target:
        if abs(diff) < margin:
            d[c1] += target * max(strength, margin - abs(diff))
    else:
        d[c1], d[c2] = d[c2], d[c1]
        d[c1] += target * (abs(diff) + strength + margin)
    new_block = idct2(d)
    return np.clip(new_block, 0, 255)

def embed_frame_worker(item):
    frame_index, in_path, out_path, coded_bits, args = item
    try:
        frame = cv2.imread(in_path)
        if frame is None:
            return (False, in_path, "read-fail")
        h, w = frame.shape[:2]
        h8, w8 = h - (h % 8), w - (w % 8)
        frame = frame[:h8, :w8]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        textured, h8, w8 = select_textured_positions(gray, args.blocks)
        total_blocks = (h8 // 8) * (w8 // 8)
        if len(textured) < args.blocks:
            seeded = seeded_positions(args.secret, frame_index, total_blocks, args.blocks - len(textured))
            positions = list(textured) + list(seeded)
        else:
            positions = textured[:args.blocks]
        L = len(coded_bits)
        bits_per_frame = args.blocks
        base_ptr = (frame_index % args.repeat) * bits_per_frame
        for k, linear_pos in enumerate(positions):
            by = (linear_pos // (w8 // 8)) * 8
            bx = (linear_pos % (w8 // 8)) * 8
            bit = coded_bits[(base_ptr + k) % L]
            block = frame[by:by+8, bx:bx+8, 0].astype(np.float32)
            new_block = embed_bit_in_block(block, bit, strength=args.strength, margin=args.margin)
            frame[by:by+8, bx:bx+8, 0] = np.round(np.clip(new_block, 0, 255)).astype(np.uint8)
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return (True, in_path, out_path)
    except Exception as e:
        return (False, in_path, str(e))

def embed_frames_parallel(input_dir, output_dir, session, secret, blocks=BLOCKS_PER_FRAME,
                          strength=EMBED_STRENGTH, margin=EMBED_MARGIN, repeat=REPEAT_EVERY_FRAMES,
                          parity=RS_PARITY_BYTES, workers=None):
    os.makedirs(output_dir, exist_ok=True)
    coded_bits, coded_len = rs_encode_bits(session, parity=parity)
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    if not files:
        print("[embed] No frames found in input folder.")
        return
    print(f"[embed] Embedding session '{session}' into {len(files)} frames.")
    args = argparse.Namespace(blocks=blocks, strength=strength, margin=margin, repeat=repeat, secret=secret)
    tasks = [(i, os.path.join(input_dir, f), os.path.join(output_dir, f"wm_{f}"), coded_bits, args) for i, f in enumerate(files)]
    workers = workers or max(1, cpu_count() - 1)
    with Pool(workers) as p:
        results = list(tqdm(p.imap_unordered(embed_frame_worker, tasks), total=len(tasks)))
    ok = sum(1 for r in results if r[0])
    print(f"✅ Watermark embedded in {ok}/{len(tasks)} frames. Output: {output_dir}")

# ================== Extraction ===================

def extract_frame_soft_values(in_path, secret, blocks):
    frame = cv2.imread(in_path)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    h8, w8 = h - (h % 8), w - (w % 8)
    frame = frame[:h8, :w8]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    textured, h8, w8 = select_textured_positions(gray, blocks)
    total_blocks = (h8 // 8) * (w8 // 8)
    if len(textured) < blocks:
        positions = textured + seeded_positions(secret, 0, total_blocks, blocks - len(textured))
    else:
        positions = textured[:blocks]
    soft_vals = []
    for linear_pos in positions:
        by = (linear_pos // (w8 // 8)) * 8
        bx = (linear_pos % (w8 // 8)) * 8
        block = frame[by:by+8, bx:bx+8, 0].astype(np.float32)
        d = dct2(block)
        c1, c2 = (3,2), (2,3)
        soft_vals.append(float(d[c1] - d[c2]))
    return soft_vals

def extract_frames(input_dir, secret, frames=240, blocks=BLOCKS_PER_FRAME, parity=RS_PARITY_BYTES):
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    files = files[:min(frames, len(files))]
    if not files:
        print("[extract] No frames found.")
        return
    print(f"[extract] Analyzing {len(files)} frames...")
    L_guess = 2048
    collected = defaultdict(list)
    for idx, fpath in enumerate(tqdm(files)):
        vals = extract_frame_soft_values(fpath, secret, blocks)
        if vals is None:
            continue
        for k, v in enumerate(vals):
            slot = k % L_guess
            collected[slot].append(v)
    inferred_bits = [('1' if np.mean(v) > 0 else '0') for v in collected.values()]
    bits_str = ''.join(inferred_bits)
    rsc = RSCodec(parity)
    for shift in range(8):
        b = bits_str[shift:]
        for length in range(4, 256):
            chunk = b[:length*8]
            if len(chunk) < 8:
                break
            try:
                ba = bytearray(int(chunk[i:i+8], 2) for i in range(0, len(chunk), 8))
                decoded = rsc.decode(bytes(ba))
                txt = decoded.decode('utf-8', errors='ignore')
                if '|' in txt:
                    sync, sess = txt.split('|', 1)
                    if sync == SYNC_PATTERN:
                        print("✅ Recovered session:", sess)
                        return sess
            except Exception:
                pass
    print("❌ Watermark not detected.")
    return None

# ================= Audio Echo =====================

def audio_embed_echo(in_wav, out_wav, session, depth=0.002, delay_ms=30):
    y, sr = librosa.load(in_wav, sr=None, mono=True)
    bits = ''.join(format(ord(c), '08b') for c in session)
    out = y.copy()
    delay = int(sr * (delay_ms / 1000.0))
    for i, b in enumerate(bits):
        start = i * delay * 2
        if start + delay >= len(y):
            break
        if b == '1':
            out[start+delay:start+2*delay] += y[start:start+delay] * depth
    sf.write(out_wav, out, sr)
    print("✅ Audio watermark embedded:", out_wav)

def audio_extract_echo(in_wav, max_chars=32, delay_ms=30):
    y, sr = librosa.load(in_wav, sr=None, mono=True)
    delay = int(sr * (delay_ms / 1000.0))
    bits = []
    for i in range(max_chars * 8):
        start = i * delay * 2
        if start + delay >= len(y):
            break
        orig = y[start:start + delay]
        delayed = y[start + delay:start + 2*delay]
        if len(orig) == 0 or len(delayed) == 0:
            break
        corr = np.dot(orig, delayed) / (np.linalg.norm(orig) * np.linalg.norm(delayed) + 1e-9)
        bits.append('1' if corr > 0.02 else '0')
    chars = ''.join(chr(int(''.join(bits[i:i+8]), 2)) for i in range(0, len(bits), 8))
    print("🔍 Extracted audio text:", chars)
    return chars

# ================= CLI =====================

def main():
    parser = argparse.ArgumentParser(description="Advanced Video+Audio Watermark Tool")
    sub = parser.add_subparsers(dest='mode', required=True)

    pE = sub.add_parser('embed')
    pE.add_argument('--input', required=True)
    pE.add_argument('--output', required=True)
    pE.add_argument('--session', required=True)
    pE.add_argument('--secret', default="top_secret")

    pX = sub.add_parser('extract')
    pX.add_argument('--input', required=True)
    pX.add_argument('--secret', default="top_secret")

    aE = sub.add_parser('audio_embed')
    aE.add_argument('--in_wav', required=True)
    aE.add_argument('--out_wav', required=True)
    aE.add_argument('--session', required=True)

    aX = sub.add_parser('audio_extract')
    aX.add_argument('--in_wav', required=True)

    args = parser.parse_args()
    if args.mode == 'embed':
        embed_frames_parallel(args.input, args.output, args.session, args.secret)
    elif args.mode == 'extract':
        extract_frames(args.input, args.secret)
    elif args.mode == 'audio_embed':
        audio_embed_echo(args.in_wav, args.out_wav, args.session)
    elif args.mode == 'audio_extract':
        audio_extract_echo(args.in_wav)

if __name__ == "__main__":
    main()
