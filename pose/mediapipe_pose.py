import mediapipe as mp

mp_pose = mp.solutions.pose

def create_pose():

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return pose