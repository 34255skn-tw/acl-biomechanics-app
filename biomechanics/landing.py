def detect_landing(
    previous_ankle_y,
    current_ankle_y,
    threshold=0.03
):

    movement = (
        current_ankle_y -
        previous_ankle_y
    )

    if movement > threshold:
        return True

    return False