def calculate_asymmetry(
    left_angle,
    right_angle
):

    average = (
        left_angle +
        right_angle
    ) / 2

    if average == 0:
        return 0

    asymmetry = (
        abs(
            left_angle -
            right_angle
        )
        /
        average
    ) * 100

    return asymmetry