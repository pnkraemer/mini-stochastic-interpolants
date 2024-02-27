from typing import Callable

import jax
import jax.numpy as jnp
import optax

import flax.linen
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


def make_loss_b(*, big_i, gamma, b_parametrized, sample_rho0, sample_rho1):
    def loss(prng_key, params):
        key_0, key_1, key_z, key_t = jax.random.split(prng_key, num=4)

        x0 = sample_rho0(key_0)
        x1 = sample_rho1(key_1)
        t = jax.random.uniform(key_t, shape=())
        z = jax.random.normal(key_z, shape=())

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

        term_one = 0.5 * jnp.abs(b_eval) ** 2
        term_two = (big_i_deriv(t, x_init, x_final) + gamma_deriv(t) * z) * b_eval
        return term_one - term_two

    return loss_b_pointwise


def make_loss_s(*, big_i, gamma, s_parametrized, sample_rho0, sample_rho1):
    def loss(prng_key, params):
        key_0, key_1, key_z, key_t = jax.random.split(prng_key, num=4)

        x0 = sample_rho0(key_0)
        x1 = sample_rho1(key_1)
        t = jax.random.uniform(key_t, shape=())
        z = jax.random.normal(key_z, shape=())

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

        term_one = 0.5 * jnp.abs(s_eval) ** 2
        term_two = (z / gamma(t)) * s_eval
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

        term_one = 0.5 * jnp.abs(s_eval) ** 2
        term_two = (1.0 / (gamma(t)) * z) * s_eval
        xx = term_one + term_two

        # Repeat with x_init, x_final, -z
        x_t = big_i(t, x_init, x_final) + gamma(t) * (-z)
        s_eval = s_parametrized(t, x_t, params)

        term_one = 0.5 * jnp.abs(s_eval) ** 2
        term_two = (1.0 / (gamma(t)) * (-z)) * s_eval
        xxxx = term_one + term_two
        return (xx + xxxx) / 2

    return loss_s_pointwise


def train_step(*, list_of_keys, model, loss, params, opt_state, optimizer):
    ret, grads = jax.value_and_grad(loss, argnums=1)(list_of_keys, params)
    params_update, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, params_update)
    return params, opt_state, ret


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


class MLP(flax.linen.Module):
    act_fn: callable
    output_dim: int
    hidden_dim: int = 64
    num_layers: int = 3

    @flax.linen.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for _ in range(self.num_layers):
            x = flax.linen.Dense(self.hidden_dim)(x)
            x = self.act_fn(x)
        x = flax.linen.Dense(self.output_dim)(x)
        return x

class Transformer(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.MultiHeadDotProductAttention(num_heads=8, qkv_features=24)(x)
        x = flax.linen.SelfAttention(num_heads=8, qkv_features=24)(x)
        return jnp.sum(x)
