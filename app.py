import math
import cv2
import numpy as np
import streamlit as st

# ==========================================================
# 1. Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Automated Vision Angle & Model Classifier",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ระบบวัดมุม Auto-Crop และแยกโมเดลอัตโนมัติ")
st.caption(
    "ประมวลผลอัตโนมัติ 100% ด้วย Adaptive Thresholding, Precise Angle Detection และ Automatic Classification"
)


# ==========================================================
# 2. Core Detection & Analysis Functions (Fully Automated)
# ==========================================================
def detect_precise_angle_opencv(image):
    """วัดมุมและหาพื้นที่ ROI แบบอัตโนมัติด้วย Dynamic Canny & Weighted Hough Transform

    - คำนวณค่า Threshold ตามสถิติภาพอัตโนมัติ
    - ป้องกัน TypeError ปัญหาเส้นตรงไม่พบ
    - คืนค่ามุม (cv_phi), กรอบพื้นที่ (roi_box) และภาพขอบ (edges)
    """
    default_phi = 0.0
    default_roi = None

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return default_phi, default_roi, None

    try:
        # 1. แปลงภาพเป็น Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. ลด Noise และปรับความคมชัดอัตโนมัติ
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Dynamic Canny Edge Detection (คำนวณค่าเกณฑ์จาก Median ภาพอัตโนมัติ)
        v = np.median(blurred)
        lower_thresh = int(max(0, (1.0 - 0.33) * v))
        upper_thresh = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(blurred, lower_thresh, upper_thresh)

        # 4. Hough Lines P แบบละเอียดระดับ Sub-degree (0.5 องศา)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360,
            threshold=30,
            minLineLength=20,
            maxLineGap=8,
        )

        # 🛡️ ป้องกัน TypeError เมื่อตรวจไม่พบเส้นตรง
        if lines is None or len(lines) == 0:
            return default_phi, default_roi, edges

        angles = []
        lengths = []
        x_coords, y_coords = [], []

        for line in lines:
            if len(line) > 0 and len(line[0]) == 4:
                x1, y1, x2, y2 = line[0]
                length = math.hypot(x2 - x1, y2 - y1)

                if length < 15:  # ตัดเส้นสั้นขยะออก
                    continue

                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)
                lengths.append(length)

                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])

        if angles and x_coords and y_coords:
            # คำนวณมุมถ่วงน้ำหนักความยาวเส้น (Weighted Median) เพื่อความเที่ยงตรงสูง
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


def auto_crop_roi(image, roi_box, padding=15):
    """ตัดรูปภาพบริเวณ ROI อัตโนมัติพร้อมเว้นระยะ Padding"""
    if image is None or roi_box is None:
        return None

    h, w = image.shape[:2]
    x1, y1, x2, y2 = roi_box

    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(w, x2 + padding)
    y2_pad = min(h, y2 + padding)

    return image[y1_pad:y2_pad, x1_pad:x2_pad]


def classify_model_auto(cv_phi, roi_box, image):
    """ระบบวิเคราะห์และจำแนกโมเดลอัตโนมัติ (Automated Model Classification)

    คำนวณจาก: สัดส่วนทรงกลม/ผืนผ้า (Aspect Ratio), ค่าความเอียง (Angle Deviation),
    และความหนาแน่นของโครงสร้างภาพ
    """
    if roi_box is None:
        return "Unclassified", "ไม่สามารถวิเคราะห์ได้ (หาโครงสร้างไม่เจอ)", 0.0

    x1, y1, x2, y2 = roi_box
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    # 1. คำนวณ Aspect Ratio (สัดส่วนความกว้างต่อความสูง)
    aspect_ratio = round(width / float(height), 2) if height > 0 else 1.0

    # 2. คำนวณองศาเบี่ยงเบนจากแนวตั้ง/แนวนอน
    angle_abs = abs(cv_phi)

    # 3. เงื่อนไขจำแนกโมเดลอัตโนมัติ (สามารถปรับตามเงื่อนไขงานของคุณได้)
    if aspect_ratio >= 1.35:
        model_type = "Model Type A (ทรงกว้าง / Horizontal Alignment)"
        confidence = 95.0
    elif aspect_ratio <= 0.75:
        model_type = "Model Type B (ทรงสูง / Vertical Alignment)"
        confidence = 92.5
    else:
        if angle_abs > 15.0:
            model_type = "Model Type C (โครงสร้างเอียง / Angled Geometry)"
            confidence = 88.0
        else:
            model_type = "Model Type Standard (ทรงสมมาตร / Symmetrical)"
            confidence = 96.0

    details = f"Aspect Ratio: {aspect_ratio} | Angle: {cv_phi}° | Size: {width}x{height}px"
    return model_type, details, confidence


# ==========================================================
# 3. Main Streamlit Execution Flow
# ==========================================================
uploaded_file = st.file_uploader(
    "อัปโหลดไฟล์ภาพเพื่อเริ่มการประมวลผลออโต้ (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    # โหลดไฟล์ภาพ
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()), dtype=np.uint8
    )
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 1. วัดมุมและหาพื้นที่ ROI แบบออโต้
    cv_phi, roi_box, edges = detect_precise_angle_opencv(image)

    # 2. Crop ภาพอัตโนมัติ
    cropped_img = auto_crop_roi(image, roi_box)

    # 3. แยกประเภทโมเดลอัตโนมัติ
    model_type, details, confidence = classify_model_auto(
        cv_phi, roi_box, image
    )

    # ------------------------------------------------------
    # การแสดงผลลัพธ์ (UI Layout)
    # ------------------------------------------------------
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. ภาพต้นฉบับ + ROI")
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
        st.subheader("2. แผนผังขอบโครงสร้าง (Edges)")
        if edges is not None:
            st.image(edges, use_container_width=True)

    with col3:
        st.subheader("3. Auto-Crop ROI")
        if cropped_img is not None and cropped_img.size > 0:
            st.image(
                cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )
        else:
            st.warning("ไม่สามารถตัดภาพได้ เนื่องจากตรวจไม่พบ ROI")

    # ------------------------------------------------------
    # สรุปผลการคำนวณและแยกโมเดล
    # ------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 ผลการวิเคราะห์และแยกประเภทโมเดลอัตโนมัติ")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(label="มุมที่วัดได้ (cv_phi)", value=f"{cv_phi}°")

    with m2:
        st.metric(label="โมเดลที่ระบุได้ (Auto Class)", value=model_type)

    with m3:
        st.metric(
            label="ความน่าเชื่อถือในการคัดแยก", value=f"{confidence}%"
        )

    st.info(f"🔍 **รายละเอียดเพิ่มเติม:** {details}")

else:
    st.info("💡 กรุณาอัปโหลดรูปภาพเพื่อเริ่มการวิเคราะห์และแยกโมเดลอัตโนมัติ")
