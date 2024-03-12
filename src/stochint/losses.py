"""Loss functions for stochastic interpolants."""

from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def make_loss_b(*, big_i, gamma, b_parametrized, sample_rho0, sample_rho1):
    def loss(prng_key, params):
        key_0, key_1, key_z, key_t = jax.random.split(prng_key, num=4)

        x0 = sample_rho0(key_0)
        x1 = sample_rho1(key_1)

        shift = jnp.finfo(jnp.dtype(x1)).eps * 100

        t = jax.random.uniform(key_t, shape=()) * (1 - 2 * shift) + shift
        z = jax.random.normal(key_z, shape=x0.shape)

        loss_b_pointwise = _get_loss_b_pointwise(
            b_parametrized=b_parametrized,
            big_i=big_i,
            gamma=gamma,
            big_i_deriv=jax.jacfwd(big_i, argnums=0),
            gamma_deriv=jax.jacfwd(gamma),
            x_init=x0,
            x_final=x1,
            z=z,
        )
        return loss_b_pointwise(t=t, params=params)

    return loss


def _get_loss_b_pointwise(
    *,
    b_parametrized: Callable,
    big_i: Callable,
    big_i_deriv: Callable,
    gamma: Callable,
    gamma_deriv: Callable,
    x_init: jax.Array,
    x_final: jax.Array,
    z: jax.Array,
):
    def loss_b_pointwise(*, t, params):
        x_t = big_i(t, x_init, x_final) + gamma(t) * z
        b_eval = b_parametrized(t, x_t, params)

        term_one = 0.5 * jnp.dot(b_eval, b_eval)

        big_i_deriv_eval = big_i_deriv(t, x_init, x_final)
        gamma_deriv_eval = gamma_deriv(t)

        assert jnp.shape(big_i_deriv_eval) == jnp.shape(x_init)
        assert jnp.shape(z) == jnp.shape(x_init)

        term_two = jnp.dot(big_i_deriv_eval + gamma_deriv_eval * z, b_eval)
        return term_one - term_two

    return loss_b_pointwise


def make_loss_s(*, big_i, gamma, s_parametrized, sample_rho0, sample_rho1):
    def loss(prng_key, params):
        key_0, key_1, key_z, key_t = jax.random.split(prng_key, num=4)

        x0 = sample_rho0(key_0)
        x1 = sample_rho1(key_1)

        shift = jnp.finfo(jnp.dtype(x1)).eps * 100

        t = jax.random.uniform(key_t, shape=()) * (1 - 2 * shift) + shift
        z = jax.random.normal(key_z, shape=x0.shape)

        loss_s_pointwise = _get_loss_s_pointwise(
            s_parametrized=s_parametrized,
            big_i=big_i,
            gamma=gamma,
            x_init=x0,
            x_final=x1,
            z=z,
        )
        return loss_s_pointwise(t=t, params=params)

    return loss


def _get_loss_s_pointwise(
    *,
    s_parametrized: Callable,
    big_i: Callable,
    gamma: Callable,
    x_init: jax.Array,
    x_final: jax.Array,
    z: jax.Array,
):
    def loss_s_pointwise(*, t, params):
        x_t = big_i(t, x_init, x_final) + gamma(t) * z
        s_eval = s_parametrized(t, x_t, params)
        assert jnp.shape(z) == jnp.shape(s_eval)

        term_one = 0.5 * jnp.dot(s_eval, s_eval)
        term_two = jnp.dot(z / gamma(t), s_eval)
        return term_one + term_two

    return loss_s_pointwise


def make_loss_s_antithetic(*, big_i, gamma, s_parametrized, sample_rho0, sample_rho1):
    def loss(prng_key, params):
        key_0, key_1, key_z, key_t = jax.random.split(prng_key, num=4)

        x0 = sample_rho0(key_0)
        x1 = sample_rho1(key_1)
        t = jax.random.uniform(key_t, shape=())
        z = jax.random.normal(key_z, shape=())

        loss_s_pointwise = _get_loss_s_pointwise_antithetic(
            s_parametrized=s_parametrized,
            big_i=big_i,
            gamma=gamma,
            x_init=x0,
            x_final=x1,
            z=z,
        )
        return loss_s_pointwise(t=t, params=params)

    return loss


def _get_loss_s_pointwise_antithetic(
    *,
    s_parametrized: Callable,
    big_i: Callable,
    gamma: Callable,
    x_init: jax.Array,
    x_final: jax.Array,
    z: jax.Array,
):
    def loss_s_pointwise(*, t, params):
        x_t = big_i(t, x_init, x_final) + gamma(t) * z
        s_eval = s_parametrized(t, x_t, params)

        term_one = 0.5 * jnp.dot(s_eval, s_eval)
        term_two = jnp.dot(1.0 / gamma(t) * z, s_eval)
        xx = term_one + term_two

        # Repeat with x_init, x_final, -z
        x_t = big_i(t, x_init, x_final) + gamma(t) * (-z)
        s_eval = s_parametrized(t, x_t, params)

        term_one = 0.5 * jnp.dot(s_eval, s_eval)
        term_two = jnp.dot(1.0 / gamma(t) * (-z), s_eval)
        xxxx = term_one + term_two
        return (xx + xxxx) / 2

    return loss_s_pointwise


def slider(fun, init_frequency, *, name):
    ts, trajectories = fun(init_frequency)
    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    lines = ax.plot(ts, trajectories, linewidth=0.5, color="black", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("State")

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the frequency.
    axfreq = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    freq_slider = Slider(
        ax=axfreq,
        label=name,
        valmin=-3.0,
        valmax=3.0,
        valinit=init_frequency,
        orientation="vertical",
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        _ts, ys_new = fun(freq_slider.val)
        for l, y in zip(lines, ys_new.T):
            l.set_ydata(y)
        fig.canvas.draw_idle()

    # register the update function with each slider
    freq_slider.on_changed(update)
