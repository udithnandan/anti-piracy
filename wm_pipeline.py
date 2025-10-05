import os
import cv2
import numpy as np
import subprocess
from tqdm import tqdm
from scipy.fftpack import dct, idct

# === CONFIG ===
FRAME_RATE = 30
QUALITY = 97  # JPEG quality for saving frames (min compression)

# === STEP 1: Extract frames ===
def extract_frames(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-qscale:v", "1",
        os.path.join(frames_dir, "frame_%05d.jpg")
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"✅ Frames extracted to {frames_dir}")

# === STEP 2: Embed watermark ===
def embed_watermark(frames_dir, output_dir, session_id):
    os.makedirs(output_dir, exist_ok=True)
    binary = ''.join(format(ord(c), '08b') for c in session_id)
    print(f"[embed] Session ID bits: {len(binary)} bits")

    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    bit_idx = 0
    for f in tqdm(frames, desc="Embedding watermark"):
        img_path = os.path.join(frames_dir, f)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # DCT on blue channel
        b, g, r = cv2.split(img)
        dct_b = dct(dct(b.T, norm='ortho').T, norm='ortho')

        if bit_idx < len(binary):
            bit = int(binary[bit_idx])
            dct_b[5, 5] = dct_b[5, 5] + (3 if bit == 1 else -3)
            bit_idx += 1

        idct_b = idct(idct(dct_b.T, norm='ortho').T, norm='ortho')
        watermarked = cv2.merge((np.uint8(np.clip(idct_b, 0, 255)), g, r))

        out_path = os.path.join(output_dir, f)
        cv2.imwrite(out_path, watermarked, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])

    print(f"✅ Watermark embedded in {len(frames)} frames.")
    print(f"Output: {output_dir}")

# === STEP 3: Reassemble video with original audio ===
def assemble_video(frames_pattern, input_video, out_video):
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(FRAME_RATE),
        '-i', frames_pattern,
        '-i', input_video,
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-r', str(FRAME_RATE),
        '-g', '30', '-bf', '2',
        '-c:a', 'copy',
        out_video
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"🎬 Reassembled watermarked video: {out_video}")

# === STEP 4: Detect watermark ===
def detect_watermark(video_path, session_id):
    tmp_dir = "temp_detect"
    extract_frames(video_path, tmp_dir)

    binary = ''.join(format(ord(c), '08b') for c in session_id)
    frames = sorted([f for f in os.listdir(tmp_dir) if f.endswith(".jpg")])
    detected_bits = ""

    for f in tqdm(frames, desc="Detecting watermark"):
        img = cv2.imread(os.path.join(tmp_dir, f), cv2.IMREAD_COLOR)
        if img is None:
            continue

        b, g, r = cv2.split(img)
        dct_b = dct(dct(b.T, norm='ortho').T, norm='ortho')
        detected_bits += '1' if dct_b[5, 5] > 0 else '0'

    # Match score
    match = 0
    for i in range(min(len(binary), len(detected_bits))):
        if binary[i] == detected_bits[i]:
            match += 1
    match_percent = (match / len(binary)) * 100

    print(f"🔍 Watermark match: {match_percent:.2f}% ({match}/{len(binary)})")
    return match_percent

# === STEP 5: Main CLI ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seamless Watermarking Pipeline with Audio Sync")
    parser.add_argument("--mode", required=True, choices=["embed", "detect"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--session", required=True)

    args = parser.parse_args()

    if args.mode == "embed":
        frames_dir = "frames_input"
        wm_dir = "frames_watermarked"

        extract_frames(args.input, frames_dir)
        embed_watermark(frames_dir, wm_dir, args.session)
        assemble_video(os.path.join(wm_dir, "frame_%05d.jpg"), args.input, args.output)

    elif args.mode == "detect":
        detect_watermark(args.input, args.session)
