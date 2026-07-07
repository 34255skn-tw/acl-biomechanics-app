def detect_jump(
    previous_ankle_y,
    current_ankle_y,
    threshold=0.03
):

    movement = (
        previous_ankle_y -
        current_ankle_y
    )

    if movement > threshold:
        return True

    return False