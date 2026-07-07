import streamlit as st
import cv2
import mediapipe as mp
import time
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from pose.mediapipe_pose import create_pose
from pose.landmarks import get_pose_points
from pose.drawing import draw_leg
from biomechanics.angles import calculate_angle
from biomechanics.velocity import calculate_velocity
from biomechanics.acceleration import calculate_acceleration
from biomechanics.asymmetry import calculate_asymmetry
from biomechanics.valgus import calculate_knee_valgus
from risk.acl_risk import acl_risk_score
from data.logger import SessionLogger

# ====================================================================
# PAGE CONFIG & RESPONSIVE DESIGN
# ====================================================================
st.set_page_config(
    page_title="ACL Biomechanics Intelligence System",
    layout="centered"
)

st.markdown("""
<style>
.stApp { background:#0f172a; }
h1, h2, h3 { color:white; text-align: center; }
p { color: #94a3b8; text-align: center; }
.stButton>button { width: 100% !important; padding: 12px !important; font-size: 16px !important; border-radius: 10px !important; }
[data-testid="stMetric"] { background:#1e293b; border:1px solid #334155; border-radius:12px; padding:12px; }
img { border-radius: 12px; width: 100% !important; }
</style>
""", unsafe_allow_html=True)

st.title("📊 ACL Biomechanics Intelligence System")
st.markdown("Real-Time ACL Risk Analysis using MediaPipe and Biomechanics")

# ====================================================================
# ATHLETE PROFILE
# ====================================================================
with st.expander("👤 ข้อมูลผู้ทดสอบ (Athlete Profile)", expanded=True):
    age = st.number_input("Age", 10, 100, 18)
    gender = st.selectbox("Gender", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", 20.0, 200.0, 60.0)
    height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)

st.markdown("---")

# ====================================================================
# INPUT SOURCE
# ====================================================================
source_type = st.radio("เลือกแหล่งที่มาของวิดีโอ (Input Source):", ["เปิดกล้องสด (Webcam/Live Phone)", "ถ่ายวิดีโอ หรือ อัปโหลดไฟล์"])

video_file = None
if source_type == "ถ่ายวิดีโอ หรือ อัปโหลดไฟล์":
    video_file = st.file_uploader("กดถ่ายวิดีโอหน้างาน หรือเลือกไฟล์วิดีโอในเครื่อง", type=["mp4", "mov", "avi"])

# ====================================================================
# INITIALIZE STATE
# ====================================================================
if 'history_data' not in st.session_state:
    st.session_state.history_data = []
if 'logger' not in st.session_state:
    st.session_state.logger = SessionLogger()

run = st.checkbox("▶️ Start Analysis / เปิดกล้องวิเคราะห์")

# ====================================================================
# PDF GENERATION FUNCTION (เพิ่มแผนภาพกราฟชุดใหม่ในหน้าที่ 2)
# ====================================================================
def generate_pdf_report(df_history, age, gender, weight, height):
    pdf = FPDF()
    
    # ---------------------------------------------------------
    # หน้าที่ 1: ข้อมูลผู้ทดสอบ + ตารางสรุปพารามิเตอร์สูงสุด
    # ---------------------------------------------------------
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 10, "ACL Biomechanics Report", ln=True, align="C")
    pdf.ln(10)
    
    # Athlete Profile
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Athlete Profile:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Age: {age} | Gender: {gender}", ln=True)
    pdf.cell(0, 8, f"Height: {height} cm | Weight: {weight} kg", ln=True)
    pdf.ln(5)
    
    if not df_history.empty:
        max_risk = df_history["Risk"].max()
        avg_risk = df_history["Risk"].mean()
        
        peak_theta = df_history["Left Knee"].max() 
        peak_omega = df_history["Velocity"].max()
        peak_alpha = df_history["Acceleration"].max()
        
        status = "Low" if max_risk < 30 else "Moderate" if max_risk < 60 else "High"

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Analysis Summary:", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 8, f"Peak ACL Risk: {max_risk}%", ln=True)
        pdf.cell(0, 8, f"Average ACL Risk: {avg_risk:.1f}%", ln=True)
        pdf.ln(5)

        # ตารางผลลัพธ์
        pdf.set_font("Arial", "B", 11)
        pdf.cell(50, 10, "Variables", border=1, align="C")
        pdf.cell(70, 10, "Peak Values Captured", border=1, align="C")
        pdf.cell(60, 10, "Unit", border=1, align="C", ln=True)
        
        pdf.set_font("Arial", "", 11)
        pdf.cell(50, 10, "Knee Angle (theta)", border=1, align="C")
        pdf.cell(70, 10, f"{peak_theta:.1f}", border=1, align="C")
        pdf.cell(60, 10, "degrees (deg)", border=1, align="C", ln=True)
        
        pdf.cell(50, 10, "Angular Velocity (omega)", border=1, align="C")
        pdf.cell(70, 10, f"{peak_omega:.1f}", border=1, align="C")
        pdf.cell(60, 10, "deg / s", border=1, align="C", ln=True)
        
        pdf.cell(50, 10, "Angular Accel (alpha)", border=1, align="C")
        pdf.cell(70, 10, f"{peak_alpha:.1f}", border=1, align="C")
        pdf.cell(60, 10, "deg / s^2", border=1, align="C", ln=True)
        
        pdf.cell(50, 10, "Risk Status", border=1, align="C")
        pdf.set_font("Arial", "B", 11)
        pdf.cell(70, 10, f"{status} Risk", border=1, align="C")
        pdf.set_font("Arial", "", 11)
        pdf.cell(60, 10, "Level", border=1, align="C", ln=True)

        # ---------------------------------------------------------
        # หน้าที่ 2: แสดงกราฟวิเคราะห์ข้อมูลทั้งหมด (พล็อตแบบ Subplots แนวตั้ง)
        # ---------------------------------------------------------
        pdf.add_page()
        
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Kinematics & Analytics Visualization", ln=True, align="L")
        pdf.ln(5)

        # สร้างรูปภาพขนาดพอดีที่มีกราฟซ้อนกันเป็น 2 ชั้น (บน-ล่าง)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.5))
        
        # กราฟย่อยที่ 1: มุมข้อเข่าและความเสี่ยง (Knee Angles & Risk)
        ax1.plot(df_history["Left Knee"], label="Left Knee Angle", color="blue", linewidth=1.5)
        ax1.plot(df_history["Right Knee"], label="Right Knee Angle", color="green", linewidth=1.5)
        ax1.plot(df_history["Risk"], label="ACL Risk (%)", color="red", linestyle="--", linewidth=1.5)
        ax1.set_title("Knee Kinematics & Injury Risk Over Time", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Degrees / Percentage", fontsize=9)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", fontsize=8)
        
        # กราฟย่อยที่ 2 (อันใหม่): ความเร็วและความเร่งเชิงมุม (Angular Velocity & Acceleration)
        ax2.plot(df_history["Velocity"], label="Velocity (omega)", color="darkorange", linewidth=1.5)
        ax2.plot(df_history["Acceleration"], label="Acceleration (alpha)", color="purple", linestyle="-.", linewidth=1.2)
        ax2.set_title("Knee Angular Velocity & Acceleration Over Time", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Frames", fontsize=9)
        ax2.set_ylabel("Velocity (deg/s) / Accel (deg/s²)", fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left", fontsize=8)
        
        plt.tight_layout()
        
        # เซฟกราฟและนำมาแปะลงในหน้าคู่ที่ 2 ของ PDF
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            plt.savefig(tmp_img.name, dpi=180)
            plt.close()
            # วางกราฟเต็มแนวตั้งของหน้าที่ 2 อย่างสวยงาม
            pdf.image(tmp_img.name, x=10, y=25, w=190)
            try: os.unlink(tmp_img.name)
            except: pass
                
    return bytes(pdf.output())

save_data = st.button("💾 Save Session & Generate PDF Report")
if save_data:
    st.session_state.logger.save_csv()
    if st.session_state.history_data:
        df_report = pd.DataFrame(st.session_state.history_data)
        pdf_bytes = generate_pdf_report(df_report, age, gender, weight, height)
        st.success("Saved successfully!")
        st.download_button(
            label="📥 Download Complete Report (with Velocity/Acceleration Graph)",
            data=pdf_bytes,
            file_name=f"ACL_Full_Report_{int(time.time())}.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("No data captured yet. Please run the analysis first.")

st.markdown("---")

# ====================================================================
# DASHBOARD METRICS ZONE (8 พารามิเตอร์)
# ====================================================================
col_r1_1, col_r1_2 = st.columns(2)
left_metric = col_r1_1.empty()
right_metric = col_r1_2.empty()

col_r2_1, col_r2_2 = st.columns(2)
risk_metric = col_r2_1.empty()
asymmetry_metric = col_r2_2.empty()

col_r3_1, col_r3_2 = st.columns(2)
velocity_metric = col_r3_1.empty()
acceleration_metric = col_r3_2.empty()

col_r4_1, col_r4_2 = st.columns(2)
left_valgus_metric = col_r4_1.empty()
right_valgus_metric = col_r4_2.empty()

frame_placeholder = st.empty()
chart_placeholder = st.empty()

# ====================================================================
# MAIN PROCESSING LOOP
# ====================================================================
if run:
    cap = None
    tfile = None
    fps = 30.0
    
    if source_type == "เปิดกล้องสด (Webcam/Live Phone)":
        cap = cv2.VideoCapture(0)
    elif source_type == "ถ่ายวิดีโอ หรือ อัปโหลดไฟล์":
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps > 0:
                fps = video_fps
        else:
            st.warning("Please upload or record a video first.")
            run = False

    if cap is not None:
        pose = create_pose()
        mp_pose = mp.solutions.pose
        
        prev_left_angle = None
        prev_velocity = 0
        st.session_state.history_data = []
        
        risk_history = []
        left_angle_history = []
        right_angle_history = []
        velocity_history = []
        acceleration_history = []

        dt = 1.0 / fps

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                points = get_pose_points(landmarks, mp_pose)
                
                left_hip, left_knee, left_ankle = points["left_hip"], points["left_knee"], points["left_ankle"]
                right_hip, right_knee, right_ankle = points["right_hip"], points["right_knee"], points["right_ankle"]
                
                left_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_angle = calculate_angle(right_hip, right_knee, right_ankle)
                
                left_angle_history.append(left_angle); right_angle_history.append(right_angle)
                if len(left_angle_history) > 5: left_angle_history.pop(0); right_angle_history.pop(0)
                left_angle = sum(left_angle_history) / len(left_angle_history)
                right_angle = sum(right_angle_history) / len(right_angle_history)
                
                asymmetry = calculate_asymmetry(left_angle, right_angle)
                left_valgus = calculate_knee_valgus(left_hip, left_knee, left_ankle)
                right_valgus = calculate_knee_valgus(right_hip, right_knee, right_ankle)
                
                if prev_left_angle is not None:
                    raw_velocity = calculate_velocity(left_angle, prev_left_angle, dt)
                    raw_acceleration = calculate_acceleration(raw_velocity, prev_velocity, dt)
                else:
                    raw_velocity, raw_acceleration = 0, 0
                    
                prev_left_angle = left_angle
                prev_velocity = raw_velocity
                
                velocity_history.append(raw_velocity); acceleration_history.append(raw_acceleration)
                if len(velocity_history) > 5: velocity_history.pop(0); acceleration_history.pop(0)
                velocity = sum(velocity_history) / len(velocity_history)
                acceleration = sum(acceleration_history) / len(acceleration_history)
                
                raw_risk = acl_risk_score(left_angle, right_angle, asymmetry, left_valgus, right_valgus, velocity, acceleration, age, gender)
                
                avg_angle = (left_angle + right_angle) / 2
                if avg_angle > 165 and abs(velocity) < 50:
                    risk = 0
                else:
                    risk = raw_risk
                
                risk_history.append(risk)
                if len(risk_history) > 8: risk_history.pop(0)
                risk = round(sum(risk_history) / len(risk_history))
                
                status = "LOW" if risk < 30 else "MODERATE" if risk < 60 else "HIGH"
                
                draw_leg(frame, left_hip, left_knee, left_ankle)
                draw_leg(frame, right_hip, right_knee, right_ankle)
                cv2.putText(frame, f"Risk: {risk}%", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(frame, status, (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                
                st.session_state.logger.add_record(time.time(), left_angle, right_angle, asymmetry, velocity, risk)
                
                left_metric.metric("Left Knee", f"{left_angle:.1f}°")
                right_metric.metric("Right Knee", f"{right_angle:.1f}°")
                risk_metric.metric("ACL Risk", f"{risk}%")
                asymmetry_metric.metric("Asymmetry", f"{asymmetry:.1f}%")
                velocity_metric.metric("Velocity", f"{velocity:.1f}")
                acceleration_metric.metric("Acceleration", f"{acceleration:.1f}")
                left_valgus_metric.metric("L Valgus", f"{left_valgus:.1f}")
                right_valgus_metric.metric("R Valgus", f"{right_valgus:.1f}")
                
                st.session_state.history_data.append({
                    "Left Knee": left_angle,
                    "Right Knee": right_angle,
                    "Velocity": velocity,
                    "Acceleration": acceleration,
                    "Risk": risk
                })
                
                df_chart = pd.DataFrame(st.session_state.history_data[-100:])
                if not df_chart.empty and all(col in df_chart.columns for col in ["Left Knee", "Right Knee", "Risk"]):
                    chart_placeholder.line_chart(df_chart[["Left Knee", "Right Knee", "Risk"]])
                
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
        cap.release()
        if tfile is not None:
            try: os.unlink(tfile.name)
            except: pass