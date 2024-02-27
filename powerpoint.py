import numpy as np


def generate_m(num_points=100, width=1):
    left_vertical = distribute_points_around_line(
        np.array([1, 1]), np.array([1, 9]), num_points // 4, width
    )
    middle_to_top = distribute_points_around_line(
        np.array([1, 9]), np.array([2.5, 5]), num_points // 4, width
    )
    middle_to_bottom = distribute_points_around_line(
        np.array([2.5, 5]), np.array([4, 9]), num_points // 4, width
    )
    right_vertical = distribute_points_around_line(
        np.array([4, 9]), np.array([4, 1]), num_points // 4, width
    )
    return np.hstack((left_vertical, middle_to_top, middle_to_bottom, right_vertical))


def generate_a(num_points=100, width=1):
    left_slant = distribute_points_around_line(
        np.array([4.5, 1]), np.array([5.5, 9]), num_points // 3, width
    )
    right_slant = distribute_points_around_line(
        np.array([5.5, 9]), np.array([6.5, 1]), num_points // 3, width
    )
    cross_bar = distribute_points_around_line(
        np.array([4.75, 5]), np.array([6.25, 5]), num_points // 3, width
    )
    return np.hstack((left_slant, right_slant, cross_bar))


def generate_r(num_points=100, width=1):
    vertical_line = distribute_points_around_line(
        np.array([7, 1]), np.array([7, 9]), num_points // 3, width
    )
    semi_circle = distribute_points_around_circle(
        np.array([7, 7.5]), 1.5, -90, 90, num_points // 3, width
    )
    leg = distribute_points_around_line(
        np.array([7, 6]), np.array([8.5, 1]), num_points // 3, width
    )
    return np.hstack((vertical_line, semi_circle, leg))


# Adjusting positions for "C" and "O" for the word "MARCO"
def generate_c(num_points=100, width=1):
    return (
        distribute_points_around_circle(
            np.array([11.5, 5]), 2, 90, 270, num_points, width
        )
        * np.array([1, 2])[..., None]
        - np.array([0, 5])[..., None]
    )


def generate_o(num_points=100, width=1):
    return (
        distribute_points_around_circle(np.array([14, 5]), 2, 0, 360, num_points, width)
        * np.array([1, 2])[..., None]
        - np.array([0, 5])[..., None]
    )


def generate_n(num_points_per_letter, width):
    return np.hstack(
        [
            distribute_points_around_line(
                np.array([5, 9]), np.array([5, 1]), num_points_per_letter // 3, width
            ),
            distribute_points_around_line(
                np.array([5, 9]), np.array([7, 1]), num_points_per_letter // 3, width
            ),
            distribute_points_around_line(
                np.array([7, 1]), np.array([7, 9]), num_points_per_letter // 3, width
            ),
        ]
    )


def generate_i(num_points_per_letter, width):
    return distribute_points_around_line(
        np.array([8.5, 1]), np.array([8.5, 9]), num_points_per_letter, width
    )


def distribute_points_around_line(start, end, num_points, width=1):
    """Distribute points around a line to simulate stroke width."""
    x_values = np.linspace(start[0], end[0], num_points)
    y_values = np.linspace(start[1], end[1], num_points)
    # Adding randomness to simulate the stroke width
    x_values += np.random.uniform(-width / 2, width / 2, num_points)
    y_values += np.random.uniform(-width / 2, width / 2, num_points)
    return np.vstack((x_values, y_values))


def distribute_points_around_circle(
    center, radius, start_angle, end_angle, num_points, width=1
):
    """Distribute points around a part of a circle to simulate stroke width."""
    angles = np.linspace(np.radians(start_angle), np.radians(end_angle), num_points)
    x_values = (
        center[0]
        + radius * np.cos(angles)
        + np.random.uniform(-width / 2, width / 2, num_points)
    )
    y_values = (
        center[1]
        + radius * np.sin(angles)
        + np.random.uniform(-width / 2, width / 2, num_points)
    )
    return np.vstack((x_values, y_values))
