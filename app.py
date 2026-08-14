import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
from ultralytics import YOLO

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title="Brix Estimation from Keypoints Angle",
    page_icon="🍇",
    layout="wide"
)

st.title("🍇 ระบบวิเคราะห์ค่า Brix จากมุมเกลียวด้วย YOLO Pose")
st.markdown("คำนวณมุมจาก **แกน X ด้านขวา (0°)** หมุน **ทวนเข็มนาฬิกา (0° - 360°)** ไปยังตำแหน่งจุดเกลียว")

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def calculate_angle_ccw(p_origin, p_target):
    """
    คำนวณมุมจากแกน X ด้านขวา (0 องศา) หมุนทวนเข็มนาฬิกาไปยัง p_target
    p_origin : (x, y) จุดหมุน / จุดโคน
    p_target : (x, y) จุดปลาย / จุดเกลียว
    """
    dx = p_target[0] - p_origin[0]
    # กลับทิศแกน Y เนื่องจาก OpenCV Y=0 อยู่ที่ขอบบนของภาพ
    dy = -(p_target[1] - p_origin[1])
    
    angle = math.degrees(math.atan2(dy, dx))
    
    # ปรับช่วงให้อยู่ระหว่าง 0 ถึง 360 องศา
    if angle < 0:
        angle += 360
        
    return angle

def calc_brix_m1(angle):
    """Model 1 (5-8-13): x = |angle - 155|"""
    x = abs(angle - 155)
    return -0.0196 * (x**2) + 0.0045 * x + 16.757

def calc_brix_m2(angle):
    """Model 2 (8-13-21): x = |angle - 136|"""
    x = abs(angle - 136)
    return 0.0082 * (x**2) - 0.6667 * x + 16.362

@st.cache_resource
def load_yolo_model(model_path="best.pt"):
    """โหลดโมเดล YOLO และใช้ Cache เพื่อไม่ให้โหลดซ้ำทุกครั้งที่กดปุ่ม"""
    return YOLO(model_path)

# ==========================================
# 3. SIDEBAR & MODEL INITIALIZATION
# ==========================================
st.sidebar.header("⚙️ การตั้งค่าระบบ")

# ตัวเลือกโมเดล
selected_model_name = st.sidebar.selectbox(
    "เลือกสูตรโมเดลคำนวณ Brix:",
    ["Model 1 (Keypoints 5-8-13)", "Model 2 (Keypoints 8-13-21)"]
)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# ตัวเลือกกรณี Index ของคุณเริ่มจาก 1 ในฉลาก
index_offset = st.sidebar.checkbox("ปรับ Index แบบ 1-based (ลบ 1 จาก Index โค้ด)", value=False)

uploaded_file = st.sidebar.file_uploader("อัปโหลดภาพที่ต้องการทดสอบ...", type=["jpg", "jpeg", "png"])

# โหลดไฟล์ best.pt
try:
    model = load_yolo_model("best.pt")
    st.sidebar.success("โหลดไฟล์ `best.pt` สำเร็จ!")
except Exception as e:
    st.sidebar.error(f"ไม่สามารถโหลดไฟล์ best.pt ได้: {e}")
    st.info("กรุณาตรวจสอบว่ามีไฟล์ `best.pt` วางอยู่ในโฟลเดอร์เดียวกับ `app.py` บน GitHub")
    st.stop()

# ==========================================
# 4. MAIN PROCESSING LOGIC
# ==========================================
if uploaded_file is not None:
    # อ่านรูปภาพ
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    annotated_img = image.copy()
    
    # รัน YOLO Pose Inference
    results = model(image, conf=conf_threshold)
    
    # ตรวจสอบการพบ Keypoints
    if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints) > 0:
        # ดึง Keypoints ของวัตถุชิ้นแรกที่ตรวจพบ
        keypoints = results[0].keypoints.xy.cpu().numpy()[0]
        
        # กำหนดดัชนีจุด Origin (จุดโคน) และ Target (จุดเกลียว)
        if "Model 1" in selected_model_name:
            raw_origin_idx, raw_target_idx = 8, 13
            calc_brix_fn = calc_brix_m1
        else:
            raw_origin_idx, raw_target_idx = 13, 21
            calc_brix_fn = calc_brix_m2

        # ปรับ Index ตามออฟเซ็ต
        origin_idx = raw_origin_idx - 1 if index_offset else raw_origin_idx
        target_idx = raw_target_idx - 1 if index_offset else raw_target_idx
        
        max_required_idx = max(origin_idx, target_idx)
        
        if max_required_idx < len(keypoints):
            p_origin = keypoints[origin_idx]
            p_target = keypoints[target_idx]
            
            # ตรวจสอบว่าพิกัดถูกต้อง (ไม่เป็น [0,0])
            if np.all(p_origin > 0) and np.all(p_target > 0):
                # คำนวณมุมทวนเข็มนาฬิกาจากแกน X
                angle = calculate_angle_ccw(p_origin, p_target)
                
                # คำนวณค่า Brix
                brix_val = calc_brix_fn(angle)
                
                # --- พล็อตจุดและเส้นบนภาพ ---
                pt_origin = tuple(p_origin.astype(int))
                pt_target = tuple(p_target.astype(int))
                
                # 1. วาดเส้นแกน X อ้างอิง (แกนแนวนอนชี้ไปทางขวา สีเหลือง)
                axis_length = 60
                ref_x_end = (pt_origin[0] + axis_length, pt_origin[1])
                cv2.line(annotated_img, pt_origin, ref_x_end, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated_img, "0 deg (X-axis)", (ref_x_end[0] + 5, ref_x_end[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # 2. วาดเส้นเวกเตอร์ไปยังจุดเกลียว (สีเขียว)
                cv2.line(annotated_img, pt_origin, pt_target, (0, 255, 0), 2, cv2.LINE_AA)
                
                # 3. วาดจุดวงกลม Origin & Target
                cv2.circle(annotated_img, pt_origin, 7, (0, 0, 255), -1)  # จุด Origin สีแดง
                cv2.circle(annotated_img, pt_target, 7, (255, 0, 0), -1)  # จุด Target สีฟ้า
                
                # ข้อความแสดงเลข Keypoint
                cv2.putText(annotated_img, f"Origin P{raw_origin_idx}", (pt_origin[0] - 20, pt_origin[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(annotated_img, f"Target P{raw_target_idx}", (pt_target[0] + 10, pt_target[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # แสดงมุมตรงกลางภาพ
                cv2.putText(annotated_img, f"Angle: {angle:.1f} deg", (pt_origin[0] - 40, pt_origin[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # --- layout การแสดงผลบน STREAMLIT ---
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(
                        cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), 
                        caption="ผลการตรวจจับจุดและพล็อตมุมวัดจากแกน X", 
                        use_container_width=True
                    )
                    
                with col2:
                    st.subheader("🎯 ผลการวิเคราะห์")
                    st.metric(label="มุมทวนเข็มจากแกน X (Angle)", value=f"{angle:.2f}°")
                    st.metric(label="ค่า Brix ประเมิน", value=f"{brix_val:.2f} °Bx")
                    
                    st.markdown("---")
                    st.markdown("### 📍 พิกัดจุดที่ใช้คำนวณ")
                    st.write(f"- **จุดโคน (Origin P{raw_origin_idx}):** `{pt_origin}`")
                    st.write(f"- **จุดปลายเกลียว (Target P{raw_target_idx}):** `{pt_target}`")
                    st.write(f"- **ความต่าง $\Delta X, \Delta Y$:** `dx={pt_target[0]-pt_origin[0]}, dy={-(pt_target[1]-pt_origin[1])}`")
            else:
                st.warning("ไม่พบค่าน้ำหนัก Keypoints ที่สมบูรณ์สำหรับจุดที่เลือก")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.error(f"โมเดล ตรวจจับได้เพียง {len(keypoints)} จุด ไม่ครอบคลุม Index P{max_required_idx}")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        st.warning("ไม่พบวัตถุหรือ Keypoints ในรูปภาพนี้ กรุณาปรับค่า Confidence Threshold ที่ Sidebar")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
else:
    st.info("👈 กรุณาอัปโหลดรูปภาพผ่านทาง Sidebar ด้านซ้ายมือเพื่อเริ่มต้น")
