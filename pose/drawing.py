import cv2

def draw_leg(
    frame,
    hip,
    knee,
    ankle
):

    h,w,_ = frame.shape

    hip = (
        int(hip[0]*w),
        int(hip[1]*h)
    )

    knee = (
        int(knee[0]*w),
        int(knee[1]*h)
    )

    ankle = (
        int(ankle[0]*w),
        int(ankle[1]*h)
    )

    cv2.line(
        frame,
        hip,
        knee,
        (0,255,0),
        6
    )

    cv2.line(
        frame,
        knee,
        ankle,
        (0,255,0),
        6
    )

    cv2.circle(
        frame,
        hip,
        8,
        (255,255,255),
        -1
    )

    cv2.circle(
        frame,
        knee,
        8,
        (255,255,255),
        -1
    )

    cv2.circle(
        frame,
        ankle,
        8,
        (255,255,255),
        -1
    )

    return frame