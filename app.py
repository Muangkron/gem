import json
import math
import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw
import streamlit as st

# -----------------------------------------------------------------------------
# 1. ตั้งค่าโครงสร้างหน้าเว็บ (Page Configuration)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ระบบประเมินความหวานสับปะรด (Phyllotaxis Math System)",
    page_icon="🍍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🍍 ระบบประเมินความหวานสับปะรดภูเก็ต (°Brix Estimator)")
st.caption(
    "จำแนกโมเดลด้วยคณิตศาสตร์สัดส่วนทรงกระบอก (Phyllotaxis Extrapolation) และวัดมุมด้วย Computer Vision"
)

# -----------------------------------------------------------------------------
# 2. Sidebar แสดงสูตรและข้อมูลอ้างอิง
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ ข้อมูลและสมการอ้างอิง")
    st.info(
        "**Model 5-8-13 (ผลเล็ก-กลาง / ตาห่าง):**\n"
        "- มุมอุดมคติ $(\\theta_0) = 155^\\circ$\n"
        "- $\\text{Brix} = -0.0196x^2 + 0.0045x + 16.757$\n\n"
        "**Model 8-13-21 (ผลกลาง-ใหญ่ / ตาแน่น):**\n"
        "- มุมอุดมคติ $(\\theta_0) = 136^\\circ$\n"
        "- $\\text{Brix} = 0.0082x^2 - 0.6667x + 16.362$"
    )
    st.markdown("---")
    st.markdown("### 📐 หลักการ Math Extrapolation")
    st.caption(
        "ระบบคำนวณจำนวนเกลียวรอบผล $N = \\frac{\\pi \\times D}{W}$\n"
        "- $D$ = ความกว้างผลในรูป 2D\n"
        "- $W$ = ระยะห่างระหว่างเกลียวใน Center ROI\n"
        "- หาก $N < 10.5$ เลือก Model 5-8-13\n"
        "- หาก $N \\ge 10.5$ เลือก Model 8-13-21"
    )


# -----------------------------------------------------------------------------
# 3. ฟังก์ชันคำนวณความหวานตามแบบจำลองคณิตศาสตร์
# -----------------------------------------------------------------------------
def calculate_brix(theta, model_choice):
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
# 4. ฟังก์ชันประมวลผลรูปภาพ (Math Extrapolation & Line Detection)
# -----------------------------------------------------------------------------
def process_pineapple_geometry(pil_img):
    img_np = np.array(pil_img.convert("RGB"))
    height, width, _ = img_np.shape

    # 4.1 หาความกว้างผลสับปะรด (Fruit Width - D)
    gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_full, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x_box, y_box, w_box, h_box = cv2.boundingRect(c)
        fruit_width_D = float(w_box)
    else:
        fruit_width_D = float(width * 0.6)

    # 4.2 Crop Center ROI (30%-70% Width, 25%-75% Height)
    x1, x2 = int(width * 0.30), int(width * 0.70)
    y1, y2 = int(height * 0.25), int(height * 0.75)
    roi = img_np[y1:y2, x1:x2]

    # 4.3 OpenCV Canny & Hough Lines ตรวจจับเส้นเกลียว
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_roi)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=30,
        maxLineGap=10,
    )

    angles_phi = []
    x_intercepts = []

    if lines is not None:
        for line in lines:
            coords = line.flatten()
            if len(coords) == 4:
                lx1, ly1, lx2, ly2 = coords
                dx = float(lx2 - lx1)
                dy = float(ly2 - ly1)
                if dx != 0:
                    slope = dy / dx
                    if slope > 0.2:  # เกณฑ์เส้นเกลียวซ้ายบน -> ขวาล่าง
                        angle_deg = math.degrees(math.atan(slope))
                        if 20.0 <= angle_deg <= 75.0:
                            angles_phi.append(angle_deg)
                            # หาจุดตัดแกน X เพื่อคำนวณระยะห่าง W
                            x_int = lx1 - (ly1 / slope)
                            x_intercepts.append(x_int)

    # คำนวณมุมแหลม phi
    detected_phi = float(np.median(angles_phi)) if angles_phi else 41.0

    # 4.4 คำนวณระยะห่างระหว่างเกลียว (Spiral Pitch - W)
    if len(x_intercepts) >= 2:
        x_intercepts.sort()
        diffs = np.diff(x_intercepts)
        valid_diffs = [d for d in diffs if d > 10.0]
        spiral_pitch_W = float(np.median(valid_diffs)) if valid_diffs else (fruit_width_D / 8.0)
    else:
        spiral_pitch_W = fruit_width_D / 8.0

    # 4.5 สูตร Math Extrapolation: N = (pi * D) / W
    estimated_N = (math.pi * fruit_width_D) / max(1.0, spiral_pitch_W)

    # 4.6 ตัดสินใจเลือกโมเดลจากค่า N
    if estimated_N < 10.5:
        auto_model = "Model 5-8-13"
    else:
        auto_model = "Model 8-13-21"

    return detected_phi, auto_model, estimated_N, (x1, y1, x2, y2)


# -----------------------------------------------------------------------------
# 5. ฟังก์ชันวาดเส้น Overlay อ้างอิงลงบนรูปภาพ
# -----------------------------------------------------------------------------
def draw_angle_overlay(pil_img, phi_deg, roi_box):
    img_copy = pil_img.copy()
    draw = PIL.ImageDraw.Draw(img_copy)
    width, height = img_copy.size

    x1, y1, x2, y2 = roi_box
    # กรอบ Center ROI
    draw.rectangle([x1, y1, x2, y2], outline="#FFD700", width=3)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    line_length = min(width, height) // 3

    # 1. Baseline 0° (สีฟ้า)
    draw.line([(cx - line_length, cy), (cx + line_length, cy)], fill="#00E5FF", width=4)

    # 2. แนวเกลียวตาตามมุม phi (สีแดงส้ม)
    phi_rad = math.radians(phi_deg)
    dx = int(line_length * math.cos(phi_rad))
    dy = int(line_length * math.sin(phi_rad))

    p1 = (cx - dx, cy - dy)
    p2 = (cx + dx, cy + dy)
    draw.line([p1, p2], fill="#FF3D00", width=5)

    draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill="#FFFFFF", outline="#000000")

    return img_copy


# -----------------------------------------------------------------------------
# 6. ส่วนประมวลผลหลัก (Main Application UI)
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("📷 1. อัปโหลดภาพถ่ายสับปะรด")
    uploaded_file = st.file_uploader(
        "เลือกรูปถ่ายสับปะรด (ระบบจะประมวลผล Math Extrapolation อัตโนมัติทันที)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)

        # ประมวลผลคำนวณสัดส่วนและมุมอัตโนมัติ
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if "current_file_id" not in st.session_state or st.session_state.current_file_id != file_id:
            st.session_state.current_file_id = file_id

            auto_phi, auto_model, est_N, roi_box = process_pineapple_geometry(image)
            st.session_state.detected_phi = auto_phi
            st.session_state.detected_model = auto_model
            st.session_state.estimated_N = est_N
            st.session_state.roi_box = roi_box

        st.markdown("---")
        st.subheader("🎛️ 2. เครื่องมือปรับแก้ทาบเส้นมุม (Manual Adjust)")
        st.caption("ขยับ Slider ด้านล่างหากต้องการปรับหมุนเส้นสีแดงส้มให้ทาบขนานไปกับร่องตาจริง")

        # Slider ปรับมุมแหลม phi
        manual_phi = st.slider(
            "ปรับมุมแหลม ($\phi$) ที่เส้นเกลียวทำกับแนวนอนฝั่งขวา:",
            min_value=15.0,
            max_value=75.0,
            value=float(st.session_state.detected_phi),
            step=0.5,
        )

        # Radio เลือกโมเดล (ค่าเริ่มต้นดึงจาก Math Extrapolation)
        default_index = 0 if st.session_state.detected_model == "Model 8-13-21" else 1
        selected_model = st.radio(
            "โมเดลสับปะรด (คำนวณอัตโนมัติจากสัดส่วน $N$):",
            ("Model 8-13-21", "Model 5-8-13"),
            index=default_index,
            help="ระบบเลือกให้อัตโนมัติจากสัดส่วน N หากต้องการเปลี่ยนสามารถเลือกได้ที่นี่",
        )

        calc_theta = 180.0 - manual_phi

        # วาดเส้นทาบภาพเรียลไทม์
        overlay_img = draw_angle_overlay(image, manual_phi, st.session_state.roi_box)
        st.image(
            overlay_img,
            caption=f"เส้นสีฟ้า (Baseline 0°) | เส้นสีแดงส้ม (แนวเกลียวตา ϕ = {manual_phi:.1f}°) | กรอบสีเหลือง (Center ROI)",
            use_container_width=True,
        )

with col_right:
    st.subheader("📊 3. ผลการคำนวณและประเมินค่า Brix")

    if uploaded_file is not None:
        ideal_angle, x_val, brix_val = calculate_brix(calc_theta, selected_model)

        m1, m2 = st.columns(2)
        m1.metric("โมเดลสับปะรดที่เลือก", selected_model)
        m2.metric("จำนวนเกลียวประมาณการ ($N$)", f"{st.session_state.estimated_N:.1f} เกลียว")

        m3, m4 = st.columns(2)
        m3.metric("มุมแหลมวัดได้ ($\phi$)", f"{manual_phi:.1f}°")
        m4.metric("มุมเกลียวจริง ($\theta = 180^\circ - \phi$)", f"{calc_theta:.1f}°")

        st.markdown("---")
        st.metric(
            label="ค่าความหวานประเมิน (°Brix)",
            value=f"{brix_val:.2f} °Brix",
            delta=f"{'ระดับหวานมาก' if brix_val >= 15.0 else 'ระดับหวานปกติ'}",
        )

        st.markdown("---")
        st.markdown("### 📝 รายละเอียดพารามิเตอร์เรขาคณิต")
        st.json(
            {
                "Selected Model": selected_model,
                "Extrapolated Spirals Count (N)": f"{st.session_state.estimated_N:.2f}",
                "Ideal Angle (theta_0)": f"{ideal_angle:.1f}°",
                "Detected Acute Angle (phi)": f"{manual_phi:.1f}°",
                "Calculated Spiral Angle (theta)": f"{calc_theta:.1f}°",
                "Deviation Value (x)": f"{x_val:.4f}",
                "Estimated Sweetness": f"{brix_val:.2f} °Brix",
            }
        )

        st.success(
            f"✅ **ประเมินสำเร็จ:** คำนวณสัดส่วนเกลียวรอบผลได้ประมาณ {st.session_state.estimated_N:.1f} เกลียว "
            f"จัดเข้า **{selected_model}** มุมเกลียวจริง $\\theta = {calc_theta:.1f}^\\circ$ "
            f"คำนวณความหวานได้ **{brix_val:.2f} °Brix**"
        )
    else:
        st.info("👈 กรุณาอัปโหลดรูปถ่ายสับปะรดที่เมนูด้านซ้ายเพื่อเริ่มต้นวิเคราะห์")
