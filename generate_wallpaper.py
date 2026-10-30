import sys
import subprocess
import cv2
import numpy as np

def get_morph(t):
    if 3.2 <= t < 4.0:
        u = (t - 3.2) / 0.8
        return float(u * u * (3.0 - 2.0 * u))
    elif 4.0 <= t < 5.8:
        return 1.0
    elif 5.8 <= t < 6.6:
        u = (t - 5.8) / 0.8
        return float(1.0 - (u * u * (3.0 - 2.0 * u)))
    return 0.0

def get_blink(t):
    b1_start, b1_dur = 1.85, 0.32
    b2_start, b2_dur = 7.30, 0.32
    for bs, bd in [(b1_start, b1_dur), (b2_start, b2_dur)]:
        if bs <= t < bs + bd:
            tau = (t - bs) / bd
            if tau < 0.38:
                u = tau / 0.38
                return float(u * u * (3.0 - 2.0 * u))
            elif tau < 0.52:
                return 1.0
            else:
                v = (tau - 0.52) / 0.48
                return float(1.0 - (v * v * (3.0 - 2.0 * v)))
    return 0.0

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
    r_inner, r_outer = 90.0, 310.0
    wave_mask = np.clip((r - r_inner) / (r_outer - r_inner), 0.0, 1.0)
    wave_mask = (wave_mask * wave_mask * (3.0 - 2.0 * wave_mask)).astype(np.float32)

    kx1, ky1 = 1.0 / 140.0, 1.0 / 180.0
    kx2, ky2 = 1.0 / 110.0, 1.0 / 150.0

    phi1 = x_grid * kx1 + y_grid * ky1
    sin_p1 = np.sin(phi1) * wave_mask * 13.0
    cos_p1 = np.cos(phi1) * wave_mask * 13.0

    phi2 = -x_grid * kx2 + y_grid * ky2
    sin_p2 = np.sin(phi2) * wave_mask * 6.0
    cos_p2 = np.cos(phi2) * wave_mask * 6.0

    phi3 = -y_grid * ky1 + x_grid * kx1
    sin_p3 = np.sin(phi3) * wave_mask * 11.0
    cos_p3 = np.cos(phi3) * wave_mask * 11.0

    phi4 = y_grid * ky2
    sin_p4 = np.sin(phi4) * wave_mask * 5.0
    cos_p4 = np.cos(phi4) * wave_mask * 5.0

    phi_sheen = x_grid * 0.006 + y_grid * 0.005
    sin_ps = np.sin(phi_sheen) * wave_mask * 15.0
    cos_ps = np.cos(phi_sheen) * wave_mask * 15.0

    ry1, ry2 = int(cy - 140), int(cy + 140)
    rx1, rx2 = int(cx - 340), int(cx + 340)
    roi_y, roi_x = np.mgrid[ry1:ry2, rx1:rx2].astype(np.float32)
    roi_r = np.sqrt((roi_x - cx)**2 + (roi_y - cy)**2)

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
            '-s', f'{w}x{h}',
            '-pix_fmt', 'gray',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '16',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        proc = subprocess.Popen(fallback_cmd, stdin=subprocess.PIPE)

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

        m = get_morph(t)
        blink_c = get_blink(t)
        roi_warped = warped[ry1:ry2, rx1:rx2].copy()

        if m < 1.0:
            pulse = 1.0 + 0.015 * np.sin(2.0 * np.pi * t / 4.0)
            r_dot = r_dot_base * pulse
            r_ring = r_ring_base * pulse
            th_ring = th_ring_base * pulse

            dot_m = np.clip(r_dot + 1.0 - roi_r, 0.0, 1.0)
            dist_ring = np.abs(roi_r - r_ring)
            ring_m = np.clip((th_ring / 2.0 + 1.0) - dist_ring, 0.0, 1.0)
            eye_patch = np.clip(dot_m + ring_m, 0.0, 1.0) * 255.0

            if blink_c > 0.0:
                h_open = 68.0 * (1.0 - blink_c)
                curv = 0.0028
                y_top = cy - h_open + curv * (roi_x - cx)**2
                y_bottom = cy + (h_open * 0.35) - curv * 0.5 * (roi_x - cx)**2
                lid_mask = np.clip(roi_y - y_top, 0.0, 1.0) * np.clip(y_bottom - roi_y, 0.0, 1.0)
                eye_patch = eye_patch * lid_mask

                if blink_c > 0.8:
                    seam_y = cy + 4.0 + 0.0015 * (roi_x - cx)**2
                    seam_dist = np.abs(roi_y - seam_y)
                    seam_x_mask = np.clip(1.0 - np.abs(roi_x - cx) / 65.0, 0.0, 1.0)
                    seam_line = np.clip(1.5 - seam_dist, 0.0, 1.0) * seam_x_mask * ((blink_c - 0.8) / 0.2) * 22.0
                    roi_warped = np.clip(roi_warped + seam_line, 0.0, 255.0)

            roi_warped = np.maximum(roi_warped, eye_patch * (1.0 - m))

        if m > 0.0:
            breath = 1.0 + 0.12 * np.sin(theta * 2.0)
            w_mouth = (240.0 + 35.0 * breath) * m
            h_mouth = (65.0 + 15.0 * breath) * m
            gap_h = (11.0 + 5.0 * breath) * m

            dx_m = (roi_x - cx) / np.maximum(w_mouth, 1.0)
            valid_x = np.abs(dx_m) < 1.0
            arc = np.sqrt(np.clip(1.0 - dx_m**2, 0.0, 1.0))
            y_top_m = cy - h_mouth * arc
            y_bot_m = cy + h_mouth * arc
            slit_top = cy - gap_h * arc
            slit_bot = cy + gap_h * arc

            mouth_void = np.clip(1.0 - np.abs(roi_y - cy) / np.maximum(h_mouth * arc, 1.0), 0.0, 1.0) * valid_x * m
            roi_warped = roi_warped * (1.0 - mouth_void)

            in_teeth = (roi_y >= y_top_m) & (roi_y <= y_bot_m) & valid_x & ~((roi_y >= slit_top) & (roi_y <= slit_bot))
            bar_period = 20.0
            bar_center_dist = np.abs(((roi_x - cx + bar_period * 0.5) % bar_period) - bar_period * 0.5)
            bar_profile = np.clip(1.0 - (bar_center_dist / 4.5)**2, 0.0, 1.0)

            tip_fade_top = np.clip((roi_y - y_top_m) / 6.0, 0.0, 1.0)
            tip_fade_bot = np.clip((y_bot_m - roi_y) / 6.0, 0.0, 1.0)
            tip_fade_slit_top = np.clip((slit_top - roi_y) / 5.0, 0.0, 1.0)
            tip_fade_slit_bot = np.clip((roi_y - slit_bot) / 5.0, 0.0, 1.0)
            tip_fade = np.maximum(tip_fade_top * tip_fade_slit_top, tip_fade_bot * tip_fade_slit_bot)

            teeth_val = np.power(bar_profile, 1.5) * tip_fade * 230.0 * m
            teeth_patch = np.zeros_like(roi_warped)
            teeth_patch[in_teeth] = teeth_val[in_teeth]
            roi_warped = np.maximum(roi_warped, teeth_patch)

        warped[ry1:ry2, rx1:rx2] = roi_warped
        final_frame = warped.astype(np.uint8)
        proc.stdin.write(final_frame.tobytes())

    proc.stdin.close()
    proc.wait()

if __name__ == '__main__':
    main()
