import mediapipe as mp

def create_pose():
    # เรียกจากตัวฐานรากตรงๆ โครงสร้างนี้รองรับ MediaPipe เวอร์ชันใหม่บนคลาวด์ 100%
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
