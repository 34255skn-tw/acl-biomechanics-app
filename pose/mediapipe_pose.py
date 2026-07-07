import mediapipe as mp

# แนะนำให้เรียกใช้ผ่านการอ้างอิงตรงจากโมดูลย่อย เพื่อรองรับ MediaPipe เวอร์ชันใหม่ๆ
from mediapipe.python.solutions import pose as mp_pose

def create_pose():
    # เรียกใช้งาน Pose Instance ตามโครงสร้างเดิมของคุณ
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
