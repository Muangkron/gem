import math
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# ==========================================================
# 1. Page Configuration & Title
# ==========================================================
st.set_page_config(
    page_title="High-Precision Angle Detection",
    page_icon="📐",
    layout="wide",
)

st.title("📐 ระบบวัดมุมและตรวจจับ ROI ความแม่นยำสูง")
st.write("อัปโหลดภาพเพื่อทำการตรวจจับขอบโครงสร้างและคำนวณมุมองศาด้วย OpenCV")


# ==========================================================
# 2. Core Detection Function (แก้ไขและเพิ่มความแม่นยำเรียบร้อย)
# ==========================================================
def detect_precise_angle_opencv(image):
    """ฟังก์ชันตรวจจับมุมและพื้นที่ ROI จากภาพด้วย OpenCV

    - แก้ไข TypeError: line[0] เมื่อหาเส้นตรงไม่พบ
    - กรองเส้นขยะ และใช้ค่า Median เพื่อความแม่นยำสูงสุด
    """
    default_phi = 0.0
    default_roi = None

    # ป้องกันกรณีภาพที่ส่งเข้ามาไม่ถูกต้อง
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return default_phi, default_roi

    try:
        # แปลงภาพเป็น Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # ลด Noise เพื่อจับขอบภาพได้คมชัดขึ้น
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # ตรวจจับขอบภาพด้วย Canny Edge Detection
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # ตรวจหาเส้นตรงด้วย Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,  # ความไวในการตรวจจับเส้น
            minLineLength=25,  # ตัดเส้นสั้นที่เป็น Noise ออก
            maxLineGap=10,  # เชื่อมเส้นที่ขาดออกจากกัน
        )

        # 🛡️ [จุดแก้ Error หลัก] เช็คว่าพบเส้นตรงหรือไม่
        if lines is None or len(lines) == 0:
            return default_phi, default_roi

        angles = []
        x_coords = []
        y_coords = []

        # วนลูปอ่านค่าพิกัดและคำนวณมุมองศา
        for line in lines:
            if len(line) > 0 and len(line[0]) == 4:
                x1, y1, x2, y2 = line[0]

                # คำนวณความยาวเส้น (ข้ามเส้นที่สั้นเกินไปเพื่อความแม่นยำ)
                length = math.hypot(x2 - x1, y2 - y1)
                if length < 15:
                    continue

                # คำนวณมุมองศา (-180 ถึง 180)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

                angles.append(angle)
                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])

        # สรุปผลลัพธ์
        if angles and x_coords and y_coords:
            # ใช้ Median หาค่ากลางมุม ป้องกันเส้นหลอกดึงค่า
            cv_phi = float(np.median(angles))

            # คำนวณกรอบ ROI Box (x_min, y_min, x_max, y_max)
            roi_box = (
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords)),
                int(max(y_coords)),
            )

            return round(cv_phi, 2), roi_box
        else:
            return default_phi, default_roi

    except Exception as e:
        # ป้องกันแอปพังหากเกิด Exception
        return default_phi, default_roi


# ==========================================================
# 3. Streamlit App Interface & Main Logic
# ==========================================================
uploaded_file = st.file_uploader(
    "เลือกไฟล์ภาพ (JPG, PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # อ่านไฟล์ภาพและแปลงเป็น OpenCV Format (BGR)
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()), dtype=np.uint8
    )
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ ภาพต้นฉบับ")
        # แปลง BGR -> RGB สำหรับแสดงผลใน Streamlit
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True
        )

    # ------------------------------------------------------
    # ประมวลผลวัดมุม (บรรทัดเดิมของคุณที่มีปัญหา)
    # ------------------------------------------------------
    cv_phi, roi_box = detect_precise_angle_opencv(image)

    # วาดกรอบ ROI บนภาพเพื่อแสดงผลลัพธ์
    annotated_image = image.copy()
    if roi_box is not None:
        x1, y1, x2, y2 = roi_box
        # วาดสี่เหลี่ยมสีเขียวล้อมรอบพื้นที่ ROI
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    with col2:
        st.subheader("🎯 ผลการวิเคราะห์")
        st.image(
            cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

        st.markdown("---")
        if roi_box is not None:
            st.success(f"**มุมที่ตรวจจับได้ (cv_phi):** `{cv_phi}°`")
            st.info(f"**ขอบเขต ROI Box (x1, y1, x2, y2):** `{roi_box}`")
        else:
            st.warning(
                "⚠️ ไม่พบเส้นขอบโครงสร้างที่ชัดเจนในภาพ ระบบแสดงค่าเริ่มต้น (0.0°)"
            )
else:
    st.info("💡 กรุณาอัปโหลดรูปภาพด้านบนเพื่อเริ่มต้นประมวลผล")
