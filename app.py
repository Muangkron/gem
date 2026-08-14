import math
import numpy as np

def calculate_accurate_spiral_angle(centroids, img_w, img_h):
    """
    คำนวณมุมเกลียวสับปะรด (theta) จากพิกัดตาที่ได้จาก YOLO
    """
    if len(centroids) < 2:
        return None, None, None, None

    # กำหนดช่วงระยะห่างระหว่างตาที่เป็นเกลียวเดียวกัน (เทียบตามความสูงภาพ)
    min_neighbor_dist = img_h * 0.05
    max_neighbor_dist = img_h * 0.30
    spiral_slopes = []

    n = len(centroids)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p1, p2 = centroids[i], centroids[j]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]

            # คัดเฉพาะคู่จุดที่มีทิศทางไปทางขวาลงล่าง (เกลียวสับปะรด)
            if dx > 10 and dy > 10:
                dist = math.hypot(dx, dy)
                if min_neighbor_dist <= dist <= max_neighbor_dist:
                    slope_px = dy / dx
                    # ครองช่วงความชันที่เป็นไปได้ของร่องเกลียว
                    if 0.35 <= slope_px <= 2.5:
                        spiral_slopes.append(slope_px)

    # หากพบคู่จุดเกลียว ให้ใช้ค่า Median ความชัน
    if spiral_slopes:
        m_pixel = float(np.median(spiral_slopes))
    else:
        # Fallback: ใช้ Linear Regression เส้นถดถอยรวมถ้าคู่จุดน้อย
        x_coords = np.array([p[0] for p in centroids], dtype=np.float64)
        y_coords = np.array([p[1] for p in centroids], dtype=np.float64)
        m_pixel, _ = np.polyfit(x_coords, y_coords, 1)
        m_pixel = abs(float(m_pixel))

    # แปลงความชันเป็นองศา
    phi_deg = math.degrees(math.atan(m_pixel))
    theta_deg = 180.0 - phi_deg

    # คำนวณแกนกลางเพื่อใช้วาดเส้น
    mean_x = float(np.mean([p[0] for p in centroids]))
    mean_y = float(np.mean([p[1] for p in centroids]))
    intercept = mean_y - (m_pixel * mean_x)

    return m_pixel, intercept, phi_deg, theta_deg
