import numpy as np

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(
        ba,
        bc
    ) / (
        np.linalg.norm(ba)
        *
        np.linalg.norm(bc)
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    angle = np.degrees(
        np.arccos(cosine)
    )

    return angle