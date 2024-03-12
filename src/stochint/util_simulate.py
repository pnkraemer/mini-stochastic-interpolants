"""Simulation utilities. Mainly ODE & SDE solvers."""


import jax
import jax.numpy as jnp


def solve_ode(x0, *, b, dt):
    t = 0.0
    x = x0
    xs = [x]
    for _ in range(int(1 / dt)):
        t += dt
        x += dt * b(t, x)
        xs.append(x)
    return jnp.asarray(xs), x


def solve_sde(x0, key, *, b, s, dt, epsilon_const):
    t = 0.0
    x = x0
    xs = [x]

    epsilon = lambda t: epsilon_const
    noises = jnp.sqrt(dt) * jax.random.normal(key, shape=(int(1 / dt),))

    def step(carry, noise):
        t, x = carry
        t = t + dt

        b_f_eval = b(t, x) + epsilon(t) * s(t, x)
        x = x + dt * b_f_eval + jnp.sqrt(2 * epsilon(t)) * noise
        return (t, x), (t, x)

    (t, x), (ts, xs) = jax.lax.scan(step, init=(t, x), xs=noises)

    ts = jnp.concatenate([jnp.zeros((1,)), ts])
    xs = jnp.concatenate([x0[None, ...], xs])
    return (ts, xs)
