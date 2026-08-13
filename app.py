import json
import math
import re
import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw
import streamlit as st

# -----------------------------------------------------------------------------
# 1. ตั้งค่าโครงสร้างหน้าเว็บ (Page Configuration)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ระบบประเมินความหวานสับปะรดจากมุมแนวเกลียวตา",
    page_icon="🍍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🍍 ระบบประเมินความหวานสับปะรดภูเก็ตอัตโนมัติ (°Brix Estimator)")
st.caption(
    "ระบบประมวลผลภาพถ่ายด้วย Computer Vision และคณิตศาสตร์สัดส่วนตามหลัก Fibonacci & Golden Angle"
)

# -----------------------------------------------------------------------------
# 2. ฟังก์ชันคำนวณความหวานตามแบบจำลองคณิตศาสตร์ (Polynomial Regression)
# -----------------------------------------------------------------------------
def calculate_brix(theta, model_choice):
    """
    คำนวณค่า x (ระยะห่างจากมุมอุดมคติ) และค่า Brix ตามโมเดลที่เลือก
    """
    if model_choice == "Model 5-8-13":
        ideal_angle = 155.0
        x = abs(theta - ideal_angle)
        brix = (-0.0196 * (x**2)) + (0.0045 * x) + 16.757
    else:  # Model 8-13-21
        ideal_angle = 136.0
        x = abs(theta - ideal_angle)
        brix = (0.0082 * (x**2)) - (0.6667 * x) + 16.362
    
    return ideal_angle, x, max(0.0, brix)


# -----------------------------------------------------------------------------
# 3. ฟังก์ชัน Computer Vision สแกนร่องตาและวัดมุมเรขาคณิต (OpenCV Engine)
# -----------------------------------------------------------------------------
def process_center_roi_and_detect_angle(pil_img):
    """
    1. ครอบตัดเฉพาะแกนกลางผล (Center ROI 40%) เพื่อลด Distortion จากความโค้ง 3D
    2. ใช้ OpenCV ตรวจจับ Edge และ Hough Lines ในแนวเกลียวซ้ายบน -> ขวาล่าง
    3. คำนวณมุมแหลม phi และมุมจริง theta
    """
    img_np = np.array(pil_img.convert("RGB"))
    height, width, _ = img_np.shape

    # Crop Center ROI (ความกว้างช่วง 30%-70%, ความสูงช่วง 25%-75%)
    x1, x2 = int(width * 0.30), int(width * 0.70)
    y1, y2 = int(height * 0.25), int(height * 0.75)
    roi = img_np[y1:y2, x1:x2]

    # แปลงเป็น Grayscale และปรับ Contrast
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # ตรวจจับเส้นตรงด้วย Probabilistic Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=30,
        maxLineGap=10,
    )

    angles_phi = []

    if lines is not None:
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            dx = lx2 - lx1
            dy = ly2 - ly1

            if dx != 0:
                slope = dy / dx
                # เลือกเฉพาะเส้นที่มีความชันสอดคล้องกับแนวเกลียวซ้ายบนลงขวาล่าง (Slope > 0)
                if slope > 0.2:
                    angle_rad = math.atan(slope)
                    angle_deg = math.degrees(angle_rad)
                    if 20.0 <= angle_deg <= 75.0:
                        angles_phi.append(angle_deg)

    # หากตรวจพบเส้น ให้ใช้ค่าเฉลี่ย ถ้าไม่พบให้ใช้ค่า Default 41.0° (theta = 139.0°)
    if angles_phi:
        detected_phi = float(np.median(angles_phi))
    else:
        detected_phi = 41.0

    detected_theta = 180.0 - detected_phi
    return detected_phi, detected_theta, (x1, y1, x2, y2)


# -----------------------------------------------------------------------------
# 4. ฟังก์ชันวาดเส้น Overlay อ้างอิงลงบนภาพ (Visual Feedback)
# -----------------------------------------------------------------------------
def draw_angle_overlay(pil_img, phi_deg, roi_box):
    """
    วาดเส้นอ้างอิงแนวนอน (Baseline 0°) และเส้นแนวเกลียวตาตามมุม phi ลงบนภาพ
    เพื่อแสดงผลให้ผู้ใช้เห็นว่าระบบวัดมุมตรงไหน
    """
    img_copy = pil_img.copy()
    draw = PIL.ImageDraw.Draw(img_copy)
    width, height = img_copy.size

    # วาดกรอบ Center ROI (เส้นสีเหลืองประ)
    x1, y1, x2, y2 = roi_box
    draw.rectangle([x1, y1, x2, y2], outline="#FFD700", width=3)

    # จุดศูนย์กลางของ Center ROI
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    line_length = min(width, height) // 3

    # 1. วาดเส้นอ้างอิงแนวนอน (Horizontal Baseline - สีฟ้า)
    draw.line(
        [(cx - line_length, cy), (cx + line_length, cy)],
        fill="#00E5FF",
        width=4,
    )

    # 2. คำนวณจุดปลายเส้นแนวเกลียวตาตามมุม phi (สีแดงส้ม)
    phi_rad = math.radians(phi_deg)
    dx = int(line_length * math.cos(phi_rad))
    dy = int(line_length * math.sin(phi_rad))

    # เส้นทแยงจาก ซ้ายบน -> ขวาล่าง
    p1 = (cx - dx, cy - dy)
    p2 = (cx + dx, cy + dy)
    draw.line([p1, p2], fill="#FF3D00", width=5)

    # วาดจุดหมุนตรงกลาง
    draw.ellipse(
        [cx - 6, cy - 6, cx + 6, cy + 6], fill="#FFFFFF", outline="#000000"
    )

    return img_copy


# -----------------------------------------------------------------------------
# 5. ส่วนติดต่อผู้ใช้ Sidebar (Sidebar Controls & Information)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ การตั้งค่าและสูตรที่ใช้")

    st.markdown("### 📐 แบบจำลอง Fibonacci")
    st.info(
        "**Model 5-8-13 (เล็ก-กลาง):**\n"
        "- มุมอุดมคติ $(\\theta_0) = 155^\\circ$\n"
        "- $\\text{Brix} = -0.0196x^2 + 0.0045x + 16.757$\n\n"
        "**Model 8-13-21 (กลาง-ใหญ่):**\n"
        "- มุมอุดมคติ $(\\theta_0) = 136^\\circ$\n"
        "- $\\text{Brix} = 0.0082x^2 - 0.6667x + 16.362$"
    )

    st.markdown("---")
    st.markdown("### 💡 คำแนะนำการถ่ายภาพ")
    st.markdown(
        "1. ถ่ายภาพด้านข้างตรงๆ (Side-view Center)\n"
        "2. วางผลสับปะรดให้ขั้วตั้งตรงแนวแกนดิ่ง\n"
        "3. ใช้แสงสว่างทั่วถึง ลดเงาสะท้อนแรงบนผิว"
    )

# -----------------------------------------------------------------------------
# 6. ส่วนการรับภาพและการประมวลผลหลัก (Main Application Logic)
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("📷 1. อัปโหลดและตรวจสอบภาพถ่าย")
    uploaded_file = st.file_uploader(
        "เลือกรูปถ่ายสับปะรด (PNG, JPG, JPEG)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)

        # ประมวลผลวัดมุมอัตโนมัติด้วย OpenCV
        auto_phi, auto_theta, roi_box = process_center_roi_and_detect_angle(
            image
        )

        st.markdown("---")
        st.subheader("🎛️ 2. เครื่องมือปรับแก้ทาบเส้นมุม (Human-in-the-Loop)")
        st.caption(
            "หากเส้นที่ระบบตรวจจับอัตโนมัติไม่ตรงร่องตา คุณสามารถใช้ Slider ปรับหมุนเส้นให้ตรงกับร่องตาจริงได้ทันที"
        )

        # Slider สำหรับ Fine-tune มุมแหลม phi
        manual_phi = st.slider(
            "ปรับมุมแหลม ($\phi$) ที่เส้นเกลียวทำกับแนวนอนฝั่งขวา:",
            min_value=15.0,
            max_value=75.0,
            value=float(auto_phi),
            step=0.5,
            help="ปรับหมุนเส้นสีแดงส้มให้ทาบขนานไปตามร่องตาสับปะรดบริเวณแกนกลางผล",
        )

        calc_theta = 180.0 - manual_phi

        # ปุ่มเลือกโมเดล (เน้นย้ำ: แยกจากลักษณะกายภาพ/จำนวนตา ไม่ใช้มุมเลือกโมเดล)
        selected_model = st.radio(
            "เลือกแบบจำลองสับปะรด (พิจารณาจากขนาดผล/จำนวนเกลียวตา):",
            ("Model 8-13-21", "Model 5-8-13"),
            index=0,
            help="Model 8-13-21: ผลกลาง-ใหญ่ ตาแน่นถี่ | Model 5-8-13: ผลเล็ก-กลาง ตาห่างกว่า",
        )

        # วาด Overlay แสดงผล
        overlay_img = draw_angle_overlay(image, manual_phi, roi_box)
        st.image(
            overlay_img,
            caption="ภาพวิเคราะห์: เส้นสีฟ้า (Horizontal Baseline 0°) | เส้นสีแดงส้ม (Primary Spiral Line) | กรอบสีเหลือง (Center ROI)",
            use_container_width=True,
        )

with col_right:
    st.subheader("📊 3. ผลการคำนวณและประเมินค่า Brix")

    if uploaded_file is not None:
        ideal_angle, x_val, brix_val = calculate_brix(
            calc_theta, selected_model
        )

        # แสดง Metrics ผลลัพธ์
        m1, m2 = st.columns(2)
        m1.metric(
            label="มุมแหลมวัดได้ ($\phi$)", value=f"{manual_phi:.1f}°"
        )
        m2.metric(
            label="มุมเกลียวจริง ($\theta = 180^\circ - \phi$)",
            value=f"{calc_theta:.1f}°",
        )

        m3, m4 = st.columns(2)
        m3.metric(
            label="ระยะเบี่ยงเบนจากมุมอุดมคติ ($x = |\\theta - \\theta_0|$)",
            value=f"{x_val:.2f}°",
        )
        m4.metric(
            label="ค่าความหวานประเมิน (°Brix)",
            value=f"{brix_val:.2f} °Brix",
            delta=f"{'หวานมาก' if brix_val >= 15.0 else 'หวานปานกลาง/ปกติ'}",
        )

        st.markdown("---")
        st.markdown("### 📝 รายละเอียดการประมวลผล")

        st.json(
            {
                "Selected Model": selected_model,
                "Ideal Angle (theta_0)": f"{ideal_angle:.1f}°",
                "Detected Acute Angle (phi)": f"{manual_phi:.1f}°",
                "Calculated Spiral Angle (theta)": f"{calc_theta:.1f}°",
                "Deviation Value (x)": f"{x_val:.4f}",
                "Estimated Sweetness": f"{brix_val:.2f} °Brix",
                "Processing Region": "Center ROI (30%-70% Width, 25%-75% Height)",
            }
        )

        st.success(
            f"✅ **ประเมินสำเร็จ:** สับปะรดผลนี้มีมุมเกลียว $\\theta = {calc_theta:.1f}^\\circ$ "
            f"ห่างจากมุมอุดมคติ ${ideal_angle:.0f}^\\circ$ อยู่ $x = {x_val:.2f}^\\circ$ "
            f"คำนวณความหวานได้ **{brix_val:.2f} °Brix**"
        )
    else:
        st.info("👈 กรุณาอัปโหลดรูปถ่ายสับปะรดที่เมนูด้านซ้ายเพื่อเริ่มต้นวิเคราะห์")
