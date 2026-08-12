if st.button("🚀 เริ่มวิเคราะห์ด้วย Gemini AI", type="primary"):
    if not api_key:
        st.error(
            "ไม่พบ API Key! กรุณากรอกใน Secrets บน Streamlit Cloud ก่อนครับ"
        )
    else:
        with st.spinner("Gemini AI กำลังวิเคราะห์แนวเกลียวตาสับปะรด..."):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-3.6-flash")

                # Prompt ปรับตามวิธีวัดจากรูปถ่ายจริง
                prompt = """
                คุณคือระบบ AI วิเคราะห์เรขาคณิตเกลียวตาสับปะรดจากรูปถ่าย

                จงวิเคราะห์ภาพสับปะรดตามขั้นตอนอย่างเคร่งครัด:

                1. **การหาเส้นอ้างอิงและการวัดมุม:**
                   - สมมติเส้นอ้างอิงแนวนอน (Horizontal Baseline) ลากตัดผ่านกลางลูกสับปะรด
                   - ลากเส้นตามแนวเกลียวตาสับปะรดจาก "ซ้ายบน" เอียงลงไป "ขวาล่าง" (Top-Left to Bottom-Right)
                   - วัดมุมแหลม (Acute Angle) ที่เส้นเกลียวทำกับเส้นแนวนอนฝั่งขวา (สมมติได้เป็น ϕ เช่น 41° หรือ 51°)
                   - คำนวณมุมจริง θ = 180° - ϕ (เช่น 180° - 41° = 139° หรือ 180° - 51.5° = 128.5°)

                2. **การจำแนกรูปแบบเกลียวและรูปทรงเพื่อเลือกโมเดล:**
                   - ประเมินรูปทรงและจำนวนแนวเกลียวตาสับปะรด:
                     * หากสับปะรดเป็นทรงยาว/เกลียวชัน (กลุ่มเกลียว 7/11/17 หรือ 5-8-13) -> ให้เลือก "Model 1"
                     * หากสับปะรดเป็นทรงกลมป้อม/เกลียวถี่ (กลุ่มเกลียว 4/6/10 หรือ 8-13-21) -> ให้เลือก "Model 2"

                ตอบกลับเฉพาะข้อความ JSON รูปแบบนี้เท่านั้น (ห้ามมีคำเกริ่นหรือ Markdown Block):
                {
                  "acute_angle_phi": 41.0,
                  "detected_angle_degrees": 139.0,
                  "spiral_pattern": "7/11/17 หรือ 4/6/10",
                  "selected_model": "Model 1",
                  "reasoning": "อธิบายสั้นๆ ว่าพบมุมแหลมกี่องศา คำนวณได้มุม θ เท่าใด และเลือกโมเดลจากทรง/เกลียวอย่างไร"
                }
                """

                response = model.generate_content([prompt, image])

                raw_text = response.text
                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)

                if json_match:
                    data = json.loads(json_match.group(0))

                    phi = float(data.get("acute_angle_phi", 0))
                    theta = float(data.get("detected_angle_degrees", 0))
                    selected_model = data.get("selected_model", "Model 1")
                    pattern = data.get("spiral_pattern", "ไม่ระบุ")
                    reasoning = data.get("reasoning", "")

                    # คำนวณค่า x และ Brix ด้วย Python
                    x, brix = calculate_brix(theta, selected_model)

                    st.success("วิเคราะห์สำเร็จ!")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("มุมแหลม (ϕ)", f"{phi:.1f}°")
                    col2.metric("มุมจริง θ (180°-ϕ)", f"{theta:.1f}°")
                    col3.metric("ค่าตัวแปร (x)", f"{x:.2f}")
                    col4.metric("ความหวานประเมิน", f"{brix:.2f} °Brix")

                    st.info(
                        f"**รูปแบบเกลียว/รูปทรง:** {pattern}\n\n"
                        f"**โมเดลที่เลือกใช้:** {selected_model}\n\n"
                        f"**ผลวิเคราะห์เพิ่มเติม:** {reasoning}"
                    )
                else:
                    st.error("ไม่สามารถแปลงข้อมูลผลลัพธ์ได้ ลองใหม่อีกครั้งครับ")

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {str(e)}")
