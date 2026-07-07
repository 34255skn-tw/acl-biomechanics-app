def calculate_acceleration(
    current_velocity,
    previous_velocity,
    dt
):

    if dt <= 0:
        return 0

    return (
        current_velocity -
        previous_velocity
    ) / dt