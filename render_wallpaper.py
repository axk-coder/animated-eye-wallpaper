import sys
import subprocess
import cv2
import numpy as np

def get_blink(t):
    b1_start, b1_dur = 2.4, 0.34
    b2_start, b2_dur = 5.8, 0.34
    c = 0.0
    for start, dur in [(b1_start, b1_dur), (b2_start, b2_dur)]:
        if start <= t < start + dur:
            tau = (t - start) / dur
            if tau < 0.38:
                u = tau / 0.38
                c = u * u * (3.0 - 2.0 * u)
            elif tau < 0.52:
                c = 1.0
            else:
                v = (tau - 0.52) / 0.48
                c = 1.0 - (v * v * (3.0 - 2.0 * v))
            break
    return float(c)

def main():
    cloth_path = '/home/akidin/animated-eye-wallpaper/cloth_base.png'
    output_path = '/home/akidin/Videos/Hidamari/eye_animated.mp4'

    cloth = cv2.imread(cloth_path, cv2.IMREAD_GRAYSCALE)
    if cloth is None:
        sys.exit(1)

    h, w = cloth.shape
    fps = 60
    duration = 8.0
    total_frames = int(fps * duration)

    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = 960.0, 540.0
    r = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)

    r_inner, r_outer = 110.0, 320.0
    wave_mask = np.clip((r - r_inner) / (r_outer - r_inner), 0.0, 1.0)
    wave_mask = (wave_mask * wave_mask * (3.0 - 2.0 * wave_mask)).astype(np.float32)

    kx1, ky1 = 1.0 / 140.0, 1.0 / 180.0
    kx2, ky2 = 1.0 / 110.0, 1.0 / 150.0

    phi1 = x_grid * kx1 + y_grid * ky1
    sin_p1 = np.sin(phi1) * wave_mask * 12.0
    cos_p1 = np.cos(phi1) * wave_mask * 12.0

    phi2 = -x_grid * kx2 + y_grid * ky2
    sin_p2 = np.sin(phi2) * wave_mask * 5.5
    cos_p2 = np.cos(phi2) * wave_mask * 5.5

    phi3 = -y_grid * ky1 + x_grid * kx1
    sin_p3 = np.sin(phi3) * wave_mask * 10.0
    cos_p3 = np.cos(phi3) * wave_mask * 10.0

    phi4 = y_grid * ky2
    sin_p4 = np.sin(phi4) * wave_mask * 5.0
    cos_p4 = np.cos(phi4) * wave_mask * 5.0

    phi_sheen = x_grid * 0.006 + y_grid * 0.005
    sin_ps = np.sin(phi_sheen) * wave_mask * 14.0
    cos_ps = np.cos(phi_sheen) * wave_mask * 14.0

    ey1, ey2 = int(cy - 120), int(cy + 120)
    ex1, ex2 = int(cx - 120), int(cx + 120)
    eye_y, eye_x = np.mgrid[ey1:ey2, ex1:ex2].astype(np.float32)
    eye_r = np.sqrt((eye_x - cx)**2 + (eye_y - cy)**2)

    r_dot_base = 14.5
    r_ring_base = 47.5
    th_ring_base = 11.5

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'gray',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'h264_nvenc',
        '-preset', 'p4',
        '-b:v', '12M',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    for f in range(total_frames):
        t = f / fps
        theta = 2.0 * np.pi * t / duration
        s1, c1 = np.sin(theta), np.cos(theta)
        s2, c2 = np.sin(2.0 * theta), np.cos(2.0 * theta)

        dx = (s1 * cos_p1 + c1 * sin_p1) + (c2 * cos_p2 - s2 * sin_p2)
        dy = (c1 * cos_p3 - s1 * sin_p3) + (s2 * cos_p4 + c2 * sin_p4)

        map_x = np.clip(x_grid + dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(y_grid + dy, 0, h - 1).astype(np.float32)

        warped = cv2.remap(cloth, map_x, map_y, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        sheen = s1 * cos_ps + c1 * sin_ps
        warped = np.clip(warped + sheen, 0.0, 255.0)

        pulse = 1.0 + 0.015 * np.sin(2.0 * np.pi * t / (duration / 2.0))
        r_dot = r_dot_base * pulse
        r_ring = r_ring_base * pulse
        th_ring = th_ring_base * pulse

        dot_m = np.clip(r_dot + 1.0 - eye_r, 0.0, 1.0)
        dist_ring = np.abs(eye_r - r_ring)
        ring_m = np.clip((th_ring / 2.0 + 1.0) - dist_ring, 0.0, 1.0)
        eye_patch = np.clip(dot_m + ring_m, 0.0, 1.0) * 255.0

        blink_c = get_blink(t)
        if blink_c > 0.0:
            h_open = 68.0 * (1.0 - blink_c)
            curv = 0.0028
            y_top = cy - h_open + curv * (eye_x - cx)**2
            y_bottom = cy + (h_open * 0.35) - curv * 0.5 * (eye_x - cx)**2
            lid_mask = np.clip(eye_y - y_top, 0.0, 1.0) * np.clip(y_bottom - eye_y, 0.0, 1.0)
            eye_patch = eye_patch * lid_mask

            if blink_c > 0.8:
                seam_y = cy + 4.0 + 0.0015 * (eye_x - cx)**2
                seam_dist = np.abs(eye_y - seam_y)
                seam_x_mask = np.clip(1.0 - np.abs(eye_x - cx) / 65.0, 0.0, 1.0)
                seam_line = np.clip(1.5 - seam_dist, 0.0, 1.0) * seam_x_mask * ((blink_c - 0.8) / 0.2) * 22.0
                warped[ey1:ey2, ex1:ex2] = np.clip(warped[ey1:ey2, ex1:ex2] + seam_line, 0.0, 255.0)

        warped[ey1:ey2, ex1:ex2] = np.maximum(warped[ey1:ey2, ex1:ex2], eye_patch)
        final_frame = warped.astype(np.uint8)
        proc.stdin.write(final_frame.tobytes())

    proc.stdin.close()
    proc.wait()

if __name__ == '__main__':
    main()
