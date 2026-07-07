def calculate_velocity(

    current_angle,

    previous_angle,

    dt

):

    if dt <= 0:

        return 0

    return (

        current_angle -

        previous_angle

    ) / dt