
import jax.numpy as jnp

import jax 
def generate_m(key, num_points, width):
    left_vertical = distribute_points_around_line(
        key,
        jnp.array([1, 1]), jnp.array([1, 9]), num_points // 4, width
    )
    middle_to_top = distribute_points_around_line(
        key,
        jnp.array([1, 9]), jnp.array([2.5, 5]), num_points // 4, width
    )
    middle_to_bottom = distribute_points_around_line(
        key,
        jnp.array([2.5, 5]), jnp.array([4, 9]), num_points // 4, width
    )
    right_vertical = distribute_points_around_line(
        key,
        jnp.array([4, 9]), jnp.array([4, 1]), num_points // 4, width
    )
    return jnp.hstack((left_vertical, middle_to_top, middle_to_bottom, right_vertical))


def generate_a(key, num_points, width):
    left_slant = distribute_points_around_line(key,
        jnp.array([4.5, 1]), jnp.array([5.5, 9]), num_points // 3, width
    )
    right_slant = distribute_points_around_line(key,
        jnp.array([5.5, 9]), jnp.array([6.5, 1]), num_points // 3, width
    )
    cross_bar = distribute_points_around_line(key,
        jnp.array([4.75, 5]), jnp.array([6.25, 5]), num_points // 3, width
    )
    return jnp.hstack((left_slant, right_slant, cross_bar))


def generate_r(key, num_points, width):
    vertical_line = distribute_points_around_line(key,
        jnp.array([7, 1]), jnp.array([7, 9]), num_points // 3, width
    )
    semi_circle = distribute_points_around_circle(key,
        jnp.array([7, 7.5]), 1.5, -90, 90, num_points // 3, width
    )
    leg = distribute_points_around_line(key,
        jnp.array([7, 6]), jnp.array([8.5, 1]), num_points // 3, width
    )
    return jnp.hstack((vertical_line, semi_circle, leg))


# Adjusting positions for "C" and "O" for the word "MARCO"
def generate_c(key, num_points, width):
    return (
        distribute_points_around_circle(key,
            jnp.array([11.5, 5]), 2, 90, 270, num_points, width
        )
        * jnp.array([1, 2])[..., None]
        - jnp.array([0, 5])[..., None]
    )


def generate_o(key, num_points, width):
    return (
        distribute_points_around_circle(key,jnp.array([14, 5]), 2, 0, 360, num_points, width)
        * jnp.array([1, 2])[..., None]
        - jnp.array([0, 5])[..., None]
    )


def generate_n(key, num_points, width):
    return jnp.hstack(
        [
            distribute_points_around_line(key,
                jnp.array([5, 9]), jnp.array([5, 1]), num_points // 3, width
            ),
            distribute_points_around_line(key,
                jnp.array([5, 9]), jnp.array([7, 1]), num_points // 3, width
            ),
            distribute_points_around_line(key,
                jnp.array([7, 1]), jnp.array([7, 9]), num_points // 3, width
            ),
        ]
    )


def generate_i(key, num_points, width):
    return distribute_points_around_line(key,
        jnp.array([8.5, 1]), jnp.array([8.5, 9]), num_points, width
    )


def distribute_points_around_line(key, start, end, num_points, width):
    """Distribute points around a line to simulate stroke width."""
    x_values = jnp.linspace(start[0], end[0], num_points)
    y_values = jnp.linspace(start[1], end[1], num_points)

    # Adding randomness to simulate the stroke width
    key_x, key_y = jax.random.split(key, num=2)
    x_values += jax.random.uniform(key_x, shape=(num_points,))*width - width/2
    y_values += jax.random.uniform(key_y, shape=(num_points,))*width - width/2
    return jnp.vstack((x_values, y_values))


def distribute_points_around_circle(
   key, center, radius, start_angle, end_angle, num_points, width
):
    """Distribute points around a part of a circle to simulate stroke width."""
    angles = jnp.linspace(jnp.radians(start_angle), jnp.radians(end_angle), num_points)
    x_values = (
        center[0]
        + radius * jnp.cos(angles)
    )
    y_values = (
        center[1]
        + radius * jnp.sin(angles)
    )

    # Adding randomness to simulate the stroke width
    key_x, key_y = jax.random.split(key, num=2)
    x_values += jax.random.uniform(key_x, shape=(num_points,))*width - width/2
    y_values += jax.random.uniform(key_y, shape=(num_points,))*width - width/2

    return jnp.vstack((x_values, y_values))
