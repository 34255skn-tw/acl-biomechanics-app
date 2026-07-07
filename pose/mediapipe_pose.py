import mediapipe as mp
# เปลี่ยนวิธีดึงข้อมูลย่อยเป็นแบบเรียกตรง เพื่อรองรับ MediaPipe ทุกเวอร์ชัน
from mediapipe.python.solutions import pose as mp_pose

def create_pose():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
