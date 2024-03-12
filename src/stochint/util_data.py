"""Data-generation utilities."""

import jax
import jax.numpy as jnp


def generate_m(key, width):
    l, m, m_, r, key_idx = jax.random.split(key, num=5)

    left_vertical = distribute_points_around_line(
        l, jnp.array([1, 1]), jnp.array([1, 9]), width
    )
    middle_to_top = distribute_points_around_line(
        m, jnp.array([1, 9]), jnp.array([2.5, 5]), width
    )
    middle_to_bottom = distribute_points_around_line(
        m_, jnp.array([2.5, 5]), jnp.array([4, 9]), width
    )
    right_vertical = distribute_points_around_line(
        r, jnp.array([4, 9]), jnp.array([4, 1]), width
    )
    full_letter = jnp.stack(
        (left_vertical, middle_to_top, middle_to_bottom, right_vertical), axis=0
    )
    idx = jax.random.choice(key_idx, jnp.arange(0, 4, step=1))
    return full_letter[idx]


def generate_a(key, width):
    idx = jax.random.choice(key, jnp.arange(0, 3, step=1))

    left_slant = distribute_points_around_line(
        key, jnp.array([4.5, 1]), jnp.array([5.5, 9]), width
    )
    right_slant = distribute_points_around_line(
        key, jnp.array([5.5, 9]), jnp.array([6.5, 1]), width
    )
    cross_bar = distribute_points_around_line(
        key, jnp.array([4.75, 5]), jnp.array([6.25, 5]), width
    )
    return jnp.stack((left_slant, right_slant, cross_bar), axis=0)[idx]


def generate_r(key, width):
    idx = jax.random.choice(key, jnp.arange(0, 3, step=1))

    vertical_line = distribute_points_around_line(
        key, jnp.array([7, 1]), jnp.array([7, 9]), width
    )
    semi_circle = distribute_points_around_circle(
        key, jnp.array([7, 7.5]), 1.5, -90, 90, width
    )
    leg = distribute_points_around_line(
        key, jnp.array([7, 6]), jnp.array([8.5, 1]), width
    )
    return jnp.stack((vertical_line, semi_circle, leg), axis=0)[idx]


# Adjusting positions for "C" and "O" for the word "MARCO"
def generate_c(key, width):
    return distribute_points_around_circle(
        key, jnp.array([11.5, 5]), 2, 90, 270, width
    ) * jnp.array([1, 2]) - jnp.array([0, 5])


def generate_o(key, width):
    return distribute_points_around_circle(
        key, jnp.array([14, 5]), 2, 0, 360, width
    ) * jnp.array([1, 2]) - jnp.array([0, 5])


def generate_n(key, width):
    key_, a, b, c = jax.random.split(key, num=4)
    idx = jax.random.choice(key_, jnp.arange(0, 3, step=1))

    return jnp.stack(
        [
            distribute_points_around_line(
                a, jnp.array([5, 9]), jnp.array([5, 1]), width
            ),
            distribute_points_around_line(
                b, jnp.array([5, 9]), jnp.array([7, 1]), width
            ),
            distribute_points_around_line(
                c, jnp.array([7, 1]), jnp.array([7, 9]), width
            ),
        ],
        axis=0,
    )[idx]


def generate_i(key, width):
    return distribute_points_around_line(
        key, jnp.array([8.5, 1]), jnp.array([8.5, 9]), width
    )


def distribute_points_around_line(key, start, end, width):
    """Distribute points around a line to simulate stroke width."""
    # x_values = jnp.linspace(start[0], end[0], num_points)
    # y_values = jnp.linspace(start[1], end[1], num_points)
    # Adding randomness to simulate the stroke width
    key_x1, key_x2, key_y1, key_y2 = jax.random.split(key, num=4)

    base = jax.random.uniform(key_x1, shape=())
    base_x = base * (end[0] - start[0]) + start[0]
    base_y = base * (end[1] - start[1]) + start[1]

    x_values = base_x + jax.random.uniform(key_x2, shape=()) * width - width / 2
    y_values = base_y + jax.random.uniform(key_y2, shape=()) * width - width / 2
    return jnp.stack((x_values, y_values))


def distribute_points_around_circle(key, center, radius, start_angle, end_angle, width):
    """Distribute points around a part of a circle to simulate stroke width."""
    # angles = jnp.linspace(jnp.radians(start_angle), jnp.radians(end_angle), num_points)

    key_x, key_y, key_base = jax.random.split(key, num=3)
    base = jax.random.uniform(key_base, shape=())
    angles = base * (jnp.radians(end_angle) - jnp.radians(start_angle)) + jnp.radians(
        start_angle
    )

    x_values = center[0] + radius * jnp.cos(angles)
    y_values = center[1] + radius * jnp.sin(angles)

    # Adding randomness to simulate the stroke width
    x_values += jax.random.uniform(key_x, shape=()) * width - width / 2
    y_values += jax.random.uniform(key_y, shape=()) * width - width / 2

    return jnp.stack((x_values, y_values))
