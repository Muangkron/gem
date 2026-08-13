import math
import cv2
import numpy as np
import streamlit as st

# ==========================================================
# 1. Page Config & Title
# ==========================================================
st.set_page_config(
    page_title="High-Precision Vision & Ridge Analysis System",
    page_icon="📐",
    layout="wide",
)

st.title("📐 ระบบวัดมุม Crop อัตโนมัติ และแยกโมเดลด้วยความถี่ร่อง (Ridge Frequency)")
st.caption(
    "วิเคราะห์ความแม่นยำสูงด้วย OpenCV: Gabor Filter Analysis + Dynamic Hough Transform"
)


# ==========================================================
# 2. Core Feature Functions
# ==========================================================
def analyze_ridge_frequency(gray_img, ksize=21, sigma=5.0, lambd=10.0):
    """วิเคราะห์ความถี่ร่อง/ความหนาแน่นโครงสร้าง (Ridge/Groove Frequency)

    ด้วย Gabor Filter สำหรับนำไปใช้แยกประเภทโมเดล
    """
    if gray_img is None:
        return 0.0, None

    # สร้าง Gabor Kernel เพื่อสแกนหาทิศทางและความถี่ร่องโครงสร้าง
    kernel = cv2.getGaborKernel(
        (ksize, ksize),
        sigma=sigma,
        theta=np.pi / 4,
        lambd=lambd,
        gamma=0.5,
        psi=0,
        ktype=cv2.CV_32F,
    )

    filtered_img = cv2.filter2D(gray_img, cv2.CV_8UC3, kernel)

    # คำนวณค่าความเข้มข้นของการตอบสนองความถี่ (Ridge Density Score)
    freq_score = float(np.mean(filtered_img))
    return freq_score, filtered_img


def detect_precise_angle_opencv(
    image, canny_low, canny_high, min_line_len, max_line_gap
):
    """วัดมุมและพื้นที่ ROI พร้อมระบบคัดกรองความแม่นยำสูง"""
    default_phi = 0.0
    default_roi = None

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return default_phi, default_roi, None

    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # ลด Noise ถนอมขอบเส้น
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny Edge Detection ตามสไลเดอร์
        edges = cv2.Canny(blurred, threshold1=canny_low, threshold2=canny_high)

        # Hough Lines P
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360,  # ความละเอียดระดับ Sub-degree (0.5 องศา)
            threshold=35,
            minLineLength=min_line_len,
            maxLineGap=max_line_gap,
        )

        # 🛡️ ป้องกัน TypeError กรณีหาเส้นไม่เจอ
        if lines is None or len(lines) == 0:
            return default_phi, default_roi, edges

        angles = []
        lengths = []
        x_coords, y_coords = [], []

        for line in lines:
            if len(line) > 0 and len(line[0]) == 4:
                x1, y1, x2, y2 = line[0]
                length = math.hypot(x2 - x1, y2 - y1)

                if length < min_line_len:
                    continue

                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)
                lengths.append(length)

                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])

        if angles and x_coords and y_coords:
            # คำนวณมุมแบบ Weighted Median (ถ่วงน้ำหนักตามความยาวเส้นตรง)
            weighted_angles = []
            for a, l in zip(angles, lengths):
                weighted_angles.extend([a] * int(l))

            cv_phi = float(np.median(weighted_angles))

            roi_box = (
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords)),
                int(max(y_coords)),
            )
            return round(cv_phi, 2), roi_box, edges
        else:
            return default_phi, default_roi, edges

    except Exception:
        return default_phi, default_roi, None


def auto_crop_roi(image, roi_box, padding=10):
    """ตัดภาพออโต้ (Auto-Crop) บริเวณ ROI พร้อมเพิ่ม Margin/Padding"""
    if image is None or roi_box is None:
        return None

    h, w = image.shape[:2]
    x1, y1, x2, y2 = roi_box

    # เพิ่มระยะขอบ Padding ไม่ให้ล้นเกินขนาดภาพจริง
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(w, x2 + padding)
    y2_pad = min(h, y2 + padding)

    cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]
    return cropped


# ==========================================================
# 3. Sidebar Controls (แถบสไลเดอร์ปรับค่าแบบเรียลไทม์)
# ==========================================================
st.sidebar.header("⚙️ สไลเดอร์ปรับค่าความแม่นยำ")

st.sidebar.subheader("1. ตรวจจับขอบ & เส้นตรง")
canny_low = st.sidebar.slider("Canny Threshold Low", 10, 150, 40)
canny_high = st.sidebar.slider("Canny Threshold High", 100, 300, 150)
min_line_len = st.sidebar.slider("Min Line Length (พิกเซล)", 10, 100, 25)
max_line_gap = st.sidebar.slider("Max Line Gap (พิกเซล)", 1, 30, 10)

st.sidebar.subheader("2. ความถี่ร่อง (Gabor / Ridge)")
gabor_ksize = st.sidebar.slider("Gabor Kernel Size", 9, 31, 21, step=2)
gabor_lambda = st.sidebar.slider("Wavelength (Lambda)", 3.0, 25.0, 10.0)

st.sidebar.subheader("3. การครอปอัตโนมัติ")
crop_padding = st.sidebar.slider("Auto-Crop Padding (px)", 0, 50, 15)


# ==========================================================
# 4. Main App Workflow
# ==========================================================
uploaded_file = st.file_uploader(
    "อัปโหลดรูปภาพเพื่อเริ่มการวิเคราะห์", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()), dtype=np.uint8
    )
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 1. ประมวลผลวัดมุม & หา ROI
    cv_phi, roi_box, edges = detect_precise_angle_opencv(
        image, canny_low, canny_high, min_line_len, max_line_gap
    )

    # 2. วิเคราะห์ความถี่ร่องโครงสร้าง (Ridge Frequency)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    freq_score, ridge_map = analyze_ridge_frequency(
        gray, ksize=gabor_ksize, lambd=gabor_lambda
    )

    # 3. Auto-Crop
    cropped_img = auto_crop_roi(image, roi_box, padding=crop_padding)

    # ------------------------------------------------------
    # การแสดงผล (Layout 3 คอลัมน์)
    # ------------------------------------------------------
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("1. ภาพต้นฉบับ & ROI")
        annotated_image = image.copy()
        if roi_box is not None:
            x1, y1, x2, y2 = roi_box
            cv2.rectangle(
                annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3
            )
        st.image(
            cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    with col2:
        st.subheader("2. แผนที่ความถี่ร่อง (Ridge Map)")
        if ridge_map is not None:
            st.image(ridge_map, use_container_width=True)

    with col3:
        st.subheader("3. Auto-Crop (ผลลัพธ์)")
        if cropped_img is not None and cropped_img.size > 0:
            st.image(
                cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )
        else:
            st.warning("ยังไม่พบกรอบ ROI สำหรับทำการ Auto-Crop")

    st.markdown("---")

    # ------------------------------------------------------
    # ผลการแยกโมเดล & ค่าความแม่นยำ (Model Classification)
    # ------------------------------------------------------
    st.subheader("📊 สรุปผลการวัดและการแยกโมเดล (Model Classification)")

    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.metric(label="มุมที่ตรวจจับได้ (cv_phi)", value=f"{cv_phi}°")

    with m_col2:
        st.metric(
            label="ค่าคะแนนความถี่ร่อง (Ridge Density Score)",
            value=f"{freq_score:.2f}",
        )

    with m_col3:
        # ตรรกะแยกประเภทโมเดลโดยใช้องศาและความถี่ร่องร่วมกัน
        if freq_score > 35.0:
            model_class = "Model Type A (ความถี่ร่องสูง / โครงสร้างหนาแน่น)"
        elif freq_score > 15.0:
            model_class = "Model Type B (ความถี่ร่องปานกลาง / โครงสร้างทั่วไป)"
        else:
            model_class = "Model Type C (ความถี่ร่องต่ำ / ผิวเรียบ)"

        st.metric(label="การจำแนกประเภทโมเดล", value=model_class)

else:
    st.info("👈 กรุณาอัปโหลดรูปภาพ และปรับสไลเดอร์ที่เมนูด้านซ้ายเพื่อเริ่มประมวลผล")
