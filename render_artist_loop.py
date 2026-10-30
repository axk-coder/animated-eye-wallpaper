import sys
import subprocess
import cv2
import numpy as np

def main():
    video_src = '/home/akidin/.gemini/antigravity-cli/brain/01b1d0d1-9297-4c77-9bd6-e97cf72eeec3/.user_uploaded/uploaded_media_1793390682599.mp4'
    output_path = '/home/akidin/Videos/Hidamari/eye_animated.mp4'

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        sys.exit(1)

    raw_frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if 40 <= idx < 279:
            clean_f = frame[:280, :]
            gray = cv2.cvtColor(clean_f, cv2.COLOR_BGR2GRAY)
            raw_frames.append(gray)
        idx += 1
    cap.release()

    n_frames = len(raw_frames)
    if n_frames == 0:
        sys.exit(1)

    target_h, target_w = 1080, 1920
    orig_h, orig_w = raw_frames[0].shape
    scale = target_h / orig_h
    scaled_w = int(orig_w * scale)
    x_off = (target_w - scaled_w) // 2
    r_w = target_w - (x_off + scaled_w)

    vignette = np.ones((target_h, target_w), dtype=np.float32)
    fade_len = 140
    for i in range(fade_len):
        f = 0.5 - 0.5 * np.cos(np.pi * i / float(fade_len))
        vignette[:, i] = f
        vignette[:, -1 - i] = f

    def process_frame(img_gray):
        scaled = cv2.resize(img_gray, (scaled_w, target_h), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        canvas = np.zeros((target_h, target_w), dtype=np.float32)
        canvas[:, x_off:x_off + scaled_w] = scaled
        canvas[:, :x_off] = cv2.flip(scaled[:, :x_off], 1)
        canvas[:, x_off + scaled_w:] = cv2.flip(scaled[:, -r_w:], 1)
        canvas = canvas * vignette
        blurred = cv2.GaussianBlur(canvas, (0, 0), 1.2)
        sharpened = np.clip(canvas * 1.35 - blurred * 0.35, 0.0, 255.0)
        return sharpened.astype(np.uint8)

    processed_frames = [process_frame(f) for f in raw_frames]

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{target_w}x{target_h}',
        '-pix_fmt', 'gray',
        '-r', '60',
        '-i', '-',
        '-c:v', 'h264_nvenc',
        '-preset', 'p6',
        '-tune', 'hq',
        '-b:v', '16M',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    except Exception:
        fallback_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{target_w}x{target_h}',
            '-pix_fmt', 'gray',
            '-r', '60',
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '16',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        proc = subprocess.Popen(fallback_cmd, stdin=subprocess.PIPE)

    for i in range(n_frames):
        curr_f = processed_frames[i]
        next_f = processed_frames[(i + 1) % n_frames]
        interp_f = ((curr_f.astype(np.float32) + next_f.astype(np.float32)) * 0.5).astype(np.uint8)
        proc.stdin.write(curr_f.tobytes())
        proc.stdin.write(interp_f.tobytes())

    proc.stdin.close()
    proc.wait()

if __name__ == '__main__':
    main()
