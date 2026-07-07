import numpy as np

def calculate_knee_valgus(
    hip,
    knee,
    ankle
):

    hip = np.array(hip)
    knee = np.array(knee)
    ankle = np.array(ankle)

    thigh = hip - knee
    shank = ankle - knee

    cosine = np.dot(
        thigh,
        shank
    ) / (
        np.linalg.norm(thigh)
        *
        np.linalg.norm(shank)
    )

    cosine = np.clip(
        cosine,
        -1,
        1
    )

    angle = np.degrees(
        np.arccos(cosine)
    )

    valgus = abs(
        180 - angle
    )

    return valgus