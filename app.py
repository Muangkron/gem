if st.button("🚀 เริ่มวิเคราะห์ด้วย Gemini AI", type="primary"):
    if not api_key:
        st.error("กรุณาใส่ Gemini API Key ในระบบ Secrets ก่อนครับ!")
    else:
        with st.spinner("Gemini AI กำลังวิเคราะห์แนวเกลียวตาสับปะรด..."):
            try:
                genai.configure(api_key=api_key)

                # ใช้โมเดลเวอร์ชันล่าสุดที่คุณตั้งค่าไว้
                model = genai.GenerativeModel("gemini-3.6-flash")

                # Prompt ปรับแต่งใหม่เพื่อความแม่นยำสูง
                prompt = """
                คุณคือ AI ผู้เชี่ยวชาญด้านเรขาคณิตและโครงสร้างผลไม้ (Pineapple Parastichy & Spiral Analysis)
                
                จงวิเคราะห์ภาพสับปะรดนี้อย่างละเอียดตามขั้นตอนต่อไปนี้:

                1. **ระบบการวัดมุม (Angle Measurement):**
                   - ให้หาแนวเส้นเกลียวตาสับปะรดหลัก (Eye Spiral Line)
                   - วัดมุม θ (องศา) โดยเริ่มวัดจาก "แกน +X ในแนวนอน" (Horizontal 0 degrees ชี้ไปทางขวา) แล้วหมุน "ทวนเข็มนาฬิกา" (Counter-Clockwise) ขึ้นไปจนถึงแนวเส้นเกลียวตาสับปะรด

                2. **การเลือกโมเดลจากรูปแบบเกลียวตา (Parastichy Pattern):**
                   - สแกนนับหรือประเมินความหนาแน่นและจำนวนแนวเกลียวตาตามลำดับฟีโบนักชี (Fibonacci Series):
                     * หากสับปะรดอยู่ในกลุ่มลวดลาย **5-8-13** (แนวเกลียวชันมาตรฐาน) -> ให้เลือก **"Model 1"**
                     * หากสับปะรดอยู่ในกลุ่มลวดลาย **8-13-21** (แนวเกลียวชันมาก/ตาสถี่) -> ให้เลือก **"Model 2"**

                ตอบกลับเฉพาะข้อความ JSON รูปแบบนี้เท่านั้น (ห้ามมีคำเกริ่นหรือ Markdown Block):
                {
                  "spiral_pattern": "5-8-13 หรือ 8-13-21",
                  "detected_angle_degrees": 45.0,
                  "selected_model": "Model 1",
                  "reasoning": "อธิบายสั้นๆ ว่าพบเกลียวรูปแบบใด และวัดมุมทวนเข็มจากแกน +X ได้กี่องศา"
                }
                """

                response = model.generate_content([prompt, image])

                raw_text = response.text
                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)

                if json_match:
                    data = json.loads(json_match.group(0))

                    theta = float(data.get("detected_angle_degrees", 0))
                    selected_model = data.get("selected_model", "Model 1")
                    pattern = data.get("spiral_pattern", "ไม่ระบุ")
                    reasoning = data.get("reasoning", "")

                    # คำนวณค่า x และ Brix ด้วย Python
                    x, brix = calculate_brix(theta, selected_model)

                    st.success("วิเคราะห์สำเร็จ!")

                    # แสดงผลลัพธ์
                    col1, col2, col3 = st.columns(3)
                    col1.metric("มุมเกลียวทวนเข็ม (θ)", f"{theta:.2f}°")
                    col2.metric("ค่าตัวแปร (x)", f"{x:.2f}")
                    col3.metric("ความหวานประเมิน", f"{brix:.2f} °Brix")

                    st.info(
                        f"**รูปแบบเกลียวตาที่พบ:** {pattern}\n\n"
                        f"**โมเดลที่เลือกใช้:** {selected_model}\n\n"
                        f"**วิเคราะห์เพิ่มเติม:** {reasoning}"
                    )
                else:
                    st.error(
                        "ไม่สามารถอ่านข้อมูล JSON จาก AI ได้ ลองกดวิเคราะห์ใหม่อีกครั้งครับ"
                    )

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {str(e)}")
