def get_pose_points(
    landmarks,
    mp_pose
):

    data = {

        "left_hip":[
            landmarks[
                mp_pose.PoseLandmark.LEFT_HIP.value
            ].x,

            landmarks[
                mp_pose.PoseLandmark.LEFT_HIP.value
            ].y
        ],

        "left_knee":[
            landmarks[
                mp_pose.PoseLandmark.LEFT_KNEE.value
            ].x,

            landmarks[
                mp_pose.PoseLandmark.LEFT_KNEE.value
            ].y
        ],

        "left_ankle":[
            landmarks[
                mp_pose.PoseLandmark.LEFT_ANKLE.value
            ].x,

            landmarks[
                mp_pose.PoseLandmark.LEFT_ANKLE.value
            ].y
        ],

        "right_hip":[
            landmarks[
                mp_pose.PoseLandmark.RIGHT_HIP.value
            ].x,

            landmarks[
                mp_pose.PoseLandmark.RIGHT_HIP.value
            ].y
        ],

        "right_knee":[
            landmarks[
                mp_pose.PoseLandmark.RIGHT_KNEE.value
            ].x,

            landmarks[
                mp_pose.PoseLandmark.RIGHT_KNEE.value
            ].y
        ],

        "right_ankle":[
            landmarks[
                mp_pose.PoseLandmark.RIGHT_ANKLE.value
            ].x,

            landmarks[
                mp_pose.PoseLandmark.RIGHT_ANKLE.value
            ].y
        ]

    }

    return data