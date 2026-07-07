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

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="ACL Biomechanics Intelligence System",
    layout="wide"
)

# =====================================
# CSS
# =====================================
st.markdown("""
<style>
.stApp{
    background:#0f172a;
}
h1,h2,h3{
    color:white;
}
[data-testid="stMetric"]{
    background:#1e293b;
    border:1px solid #334155;
    border-radius:15px;
    padding:15px;
}
section[data-testid="stSidebar"]{
    background:#111827;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================
st.title("ACL Biomechanics Intelligence System")
st.markdown("Real-Time ACL Risk Analysis using MediaPipe and Biomechanics")

# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("Athlete Profile")
age = st.sidebar.number_input("Age", 10, 100, 18)
gender = st.sidebar.selectbox("Gender", ["Male","Female"])
weight = st.sidebar.number_input("Weight (kg)", 20.0, 200.0, 60.0)
height = st.sidebar.number_input("Height (cm)", 100.0, 250.0, 170.0)

# =====================================
# KNOWLEDGE PANEL
# =====================================
tips = [
    "Landing with stiff knees increases ACL loading.",
    "Female athletes have higher ACL injury rates.",
    "Hamstring strength can help protect ACL.",
    "Poor landing mechanics increase injury risk.",
    "Large left-right asymmetry may indicate risk."
]
st.info(tips[int(time.time()) % len(tips)])

# =====================================
# START & INITIALIZE STATE
# =====================================
run = st.checkbox("Start Camera")
logger = SessionLogger()

if 'history_data' not in st.session_state:
    st.session_state.history_data = []

# =====================================
# PDF GENERATION FUNCTION (2 หน้าสมบูรณ์)
# =====================================
def generate_pdf_report(df_history, age, gender, weight, height):
    pdf = FPDF()
    
    # --- หน้าที่ 1: ตารางสรุปพารามิเตอร์สูงสุดตามมาตรฐานใบงาน ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 10, "ACL Biomechanics Report", ln=True, align="C")
    pdf.ln(10)
    
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

        # วาดตารางข้อมูลพารามิเตอร์
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

        # --- หน้าที่ 2: บังคับขึ้นหน้าใหม่เพื่อแสดงกราฟทั้งหมด ---
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Kinematics & Analytics Visualizations", ln=True, align="L")
        pdf.ln(5)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 8))
        
        # กราฟบน: มุมเข่ากับความเสี่ยง
        ax1.plot(df_history["Left Knee"].values, label="Left Knee Angle", color="blue", linewidth=1.5)
        ax1.plot(df_history["Right Knee"].values, label="Right Knee Angle", color="green", linewidth=1.5)
        ax1.plot(df_history["Risk"].values, label="ACL Risk (%)", color="red", linestyle="--", linewidth=1.5)
        ax1.set_title("Knee Kinematics & Injury Risk Over Time", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Degrees / Percentage", fontsize=9)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", fontsize=8)
        
        # กราฟล่าง: ความเร็วและความเร่ง
        ax2.plot(df_history["Velocity"].values, label="Velocity (omega)", color="darkorange", linewidth=1.5)
        ax2.plot(df_history["Acceleration"].values, label="Acceleration (alpha)", color="purple", linestyle="-.", linewidth=1.2)
        ax2.set_title("Knee Angular Velocity & Acceleration Over Time", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Frames", fontsize=9)
        ax2.set_ylabel("Velocity (deg/s) / Accel (deg/s²)", fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left", fontsize=8)
        
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            plt.savefig(tmp_img.name, dpi=180)
            plt.close()
            pdf.image(tmp_img.name, x=10, y=25, w=190)
            try: os.unlink(tmp_img.name)
            except: pass
                
    return bytes(pdf.output())

# ปุ่มสร้างและดาวน์โหลดรายงานความยาว 2 หน้า
save_data = st.button("Save Session & Generate Report")
if save_data:
    logger.save_csv()
    if st.session_state.history_data:
        df_report = pd.DataFrame(st.session_state.history_data)
        pdf_bytes = generate_pdf_report(df_report, age, gender, weight, height)
        st.success("Session saved successfully!")
        st.download_button(
            label="📥 Download 2-Page PDF Report (Table & Multi-Graphs)",
            data=pdf_bytes,
            file_name=f"ACL_Biomechanics_Report_{int(time.time())}.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("No data captured yet. Please check 'Start Camera' and move first.")

st.markdown("---")

# =====================================
# METRICS PLACEMENT
# =====================================
col1, col2, col3, col4 = st.columns(4)
left_metric = col1.empty()
right_metric = col2.empty()
risk_metric = col3.empty()
asymmetry_metric = col4.empty()

col5, col6, col7, col8 = st.columns(4)
velocity_metric = col5.empty()
acceleration_metric = col6.empty()
left_valgus_metric = col7.empty()
right_valgus_metric = col8.empty()

frame_placeholder = st.empty()
chart_placeholder = st.empty()

# =====================================
# CAMERA LOOP
# =====================================
if run:
    pose = create_pose()
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    
    fps = 30.0
    dt = 1.0 / fps
    
    prev_left_angle = None
    prev_velocity = 0
    st.session_state.history_data = []

    risk_history = []
    left_angle_history = []
    right_angle_history = []
    velocity_history = []
    acceleration_history = []

    while True:
        success, frame = cap.read()
        if not success:
            st.error("Cannot access camera")
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
            
            left_angle_history.append(left_angle)
            right_angle_history.append(right_angle)
            if len(left_angle_history) > 5:
                left_angle_history.pop(0)
                right_angle_history.pop(0)
            left_angle = sum(left_angle_history) / len(left_angle_history)
            right_angle = sum(right_angle_history) / len(right_angle_history)
            
            asymmetry = calculate_asymmetry(left_angle, right_angle)
            left_valgus = calculate_knee_valgus(left_hip, left_knee, left_ankle)
            right_valgus = calculate_knee_valgus(right_hip, right_knee, right_ankle)
            
            # แก้ไขบั๊กการคำนวณความเร็วและความเร่งเชิงมุมไม่ให้เป็นศูนย์ตายตัว
            if prev_left_angle is not None:
                raw_velocity = calculate_velocity(left_angle, prev_left_angle, dt)
                raw_acceleration = calculate_acceleration(raw_velocity, prev_velocity, dt)
            else:
                raw_velocity, raw_acceleration = 0, 0
                
            prev_left_angle = left_angle
            prev_velocity = raw_velocity
            
            velocity_history.append(raw_velocity)
            acceleration_history.append(raw_acceleration)
            if len(velocity_history) > 5:
                velocity_history.pop(0)
                acceleration_history.pop(0)
                
            velocity = sum(velocity_history) / len(velocity_history)
            acceleration = sum(acceleration_history) / len(acceleration_history)
            
            raw_risk = acl_risk_score(left_angle, right_angle, asymmetry, left_valgus, right_valgus, velocity, acceleration, age, gender, landing=False)
            
            # แก้อาการคะแนนค้างที่ 25% ด้วยการตรวจสอบการยืนตรงนิ่งๆ
            avg_angle = (left_angle + right_angle) / 2
            if avg_angle > 165 and abs(velocity) < 50:
                risk = 0
            else:
                risk = raw_risk
                
            risk_history.append(risk)
            if len(risk_history) > 10:
                risk_history.pop(0)
            risk = round(sum(risk_history) / len(risk_history))
            
            logger.add_record(time.time(), left_angle, right_angle, asymmetry, velocity, risk)
            status = "LOW" if risk < 30 else "MODERATE" if risk < 60 else "HIGH"
            
            draw_leg(frame, left_hip, left_knee, left_ankle)
            draw_leg(frame, right_hip, right_knee, right_ankle)
            cv2.putText(frame, f"Risk: {risk}%", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame, status, (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            
            left_metric.metric("Left Knee", f"{left_angle:.1f}°")
            right_metric.metric("Right Knee", f"{right_angle:.1f}°")
            risk_metric.metric("ACL Risk", f"{risk}%")
            asymmetry_metric.metric("Asymmetry", f"{asymmetry:.1f}%")
            velocity_metric.metric("Velocity", f"{velocity:.1f}")
            acceleration_metric.metric("Acceleration", f"{acceleration:.1f}")
            left_valgus_metric.metric("L Valgus", f"{left_valgus:.1f}")
            right_valgus_metric.metric("R Valgus", f"{right_valgus:.1f}")
            
            # จัดเก็บข้อมูลประวัติลง Session State สำหรับนำไปดึงค่าพล็อตเป็นรูปภาพกราฟในรายงาน PDF
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