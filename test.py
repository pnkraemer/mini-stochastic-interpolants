import functools
from typing import Callable

import flax.linen
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tqdm

import imports

# Training parameters
num_samples = 1_000
num_epochs_b = 100_000
num_epochs_s = 100_000
learning_rate_b = 0.01
learning_rate_s = 0.01


# Generation parameters
num_generates = 100
dt = 0.01


# The one and only magic key
prng_key = jax.random.PRNGKey(1)


# Set up the problem: sample from mixtures of Gaussians

x_shape = (1,)


def sample_from_mean(key, *, mean):
    key_idx, key_g = jax.random.split(key, num=2)
    idx = jax.random.choice(key_idx, jnp.arange(0, len(mean), step=1))
    return jax.random.normal(key_g, shape=x_shape) + mean[idx]


m0 = jnp.asarray([[-10], [0], [10]])
m1 = jnp.asarray([[-5], [5]])
sample_rho0 = functools.partial(sample_from_mean, mean=m0)
sample_rho1 = functools.partial(sample_from_mean, mean=m1)


model_b = imports.MLP(
    output_dim=x_shape[0],
    num_layers=2,
    hidden_dim=10,
    act_fn=jax.nn.tanh,
)
model_s = imports.MLP(
    output_dim=x_shape[0],
    num_layers=2,
    hidden_dim=10,
    act_fn=jax.nn.tanh,
)

# model_b = imports.Transformer()
# model_s = imports.Transformer()


def b_parametrized(t, x, p):
    t_and_x = jnp.concatenate([t[None], x.reshape((-1,))])[None, ...]
    return model_b.apply(p, t_and_x).reshape(x_shape)


def s_parametrized(t, x, p):
    t_and_x = jnp.concatenate([t[None], x.reshape((-1,))])[None, ...]
    return model_s.apply(p, t_and_x).reshape(x_shape)


# Initialize the model parameters

t_and_x_like = jnp.concatenate([jnp.zeros((1,)), jnp.zeros(x_shape).reshape((-1,))])[
    None, ...
]
prng_key_b, prng_key_s, prng_key = jax.random.split(prng_key, num=3)
params_b = model_b.init(prng_key_b, t_and_x_like)
params_s = model_s.init(prng_key_s, t_and_x_like)


# Define losses aka choose type of stochastic interpolation


def big_i(t, x_0, x_1):
    return (1 - t) * x_0 + t * x_1


def gamma(t):
    alpha = 1.0
    return jnp.sqrt(alpha * t * (1 - t))


loss_b = imports.make_loss_b(
    big_i=big_i,
    gamma=gamma,
    b_parametrized=b_parametrized,
    sample_rho0=sample_rho0,
    sample_rho1=sample_rho1,
)
loss_s = imports.make_loss_s(
    big_i=big_i,
    gamma=gamma,
    s_parametrized=s_parametrized,
    sample_rho0=sample_rho0,
    sample_rho1=sample_rho1,
)


# todo: make these four lines a bit less horrible


def loss_b_eval(*a, **kw):
    loss_b_vmapped = jax.vmap(loss_b, in_axes=(0, None))
    loss_b_per_sample = loss_b_vmapped(*a, **kw)
    return jnp.mean(loss_b_per_sample, axis=0)


def loss_s_eval(*a, **kw):
    loss_s_vmapped = jax.vmap(loss_s, in_axes=(0, None))
    loss_s_per_sample = loss_s_vmapped(*a, **kw)
    return jnp.mean(loss_s_per_sample, axis=0)


# Set up an optimizer and optimize b

optimizer_b = optax.adam(learning_rate_b)
opt_state_b = optimizer_b.init(params_b)
step_b_nonjit = functools.partial(
    imports.train_step,
    loss=loss_b_eval,
    model=model_b,
    optimizer=optimizer_b,
)
step_b = jax.jit(step_b_nonjit)

pbar = tqdm.tqdm(range(num_epochs_b))
for epoch in pbar:
    try:
        prng_key, _ = jax.random.split(prng_key, num=2)
        keys_b_all = jax.random.split(prng_key, num=num_samples)

        params_b, opt_state_b, loss_value_b = step_b(
            params=params_b,
            opt_state=opt_state_b,
            list_of_keys=keys_b_all,
        )
        pbar.set_description(f"Loss (b): {loss_value_b:.3f}")
    except KeyboardInterrupt:
        break


# Set up an optimizer and optimize s


optimizer_s = optax.adam(learning_rate_s)
opt_state_s = optimizer_s.init(params_s)
step_s_nonjit = functools.partial(
    imports.train_step,
    loss=loss_s_eval,
    model=model_s,
    optimizer=optimizer_s,
)
step_s = jax.jit(step_s_nonjit)

pbar = tqdm.tqdm(range(num_epochs_s))
for epoch in pbar:
    try:
        prng_key, _ = jax.random.split(prng_key, num=2)
        keys_s_all = jax.random.split(prng_key, num=num_samples)
        params_s, opt_state_s, loss_value_s = step_s(
            params=params_s,
            opt_state=opt_state_s,
            list_of_keys=keys_s_all,
        )
        pbar.set_description(f"Loss (s): {loss_value_s:.3f}")
    except KeyboardInterrupt:
        break

# Fix the optimized parameters

b = functools.partial(b_parametrized, p=params_b)
s = functools.partial(s_parametrized, p=params_s)


# Prepare the SDE solver
prng_key_init_x0s, prng_key_sde, prng_key = jax.random.split(prng_key, num=3)
keys_init_x0s = jax.random.split(prng_key_init_x0s, num=num_generates)
keys_sde = jax.random.split(prng_key_sde, num_generates)
simulate_sde_single = functools.partial(
    imports.solve_sde, dt=dt, b=b, s=s, epsilon_const=0.0
)
simulate_sde = jax.vmap(simulate_sde_single, out_axes=(None, 0))

# Sample the initial states and simulate the SDE
x0s = jax.vmap(sample_rho0)(keys_init_x0s)
(t_trajectories, x1_trajectories) = simulate_sde(x0s, keys_sde)


# Plot the results
plt.plot(t_trajectories, x1_trajectories[:, :, 0].T, color="black", alpha=0.2)
plt.show()


# Slider time


def f(log_epsilon, /):
    epsilon = 10**log_epsilon
    simulate_sde = functools.partial(
        imports.solve_sde, dt=dt, b=b, s=s, epsilon_const=epsilon
    )
    trajectories = jax.vmap(simulate_sde, out_axes=(None, 0))(x0s, keys_sde)
    # shape (num_generate, num_timesteps, num_x_dim)
    (t_trajectories, x1_trajectories) = trajectories
    return t_trajectories, x1_trajectories[:, :, 0].T


imports.slider(f, 0.0, name="log - Universal attention transport constant")
plt.show()

print("Boomshakalaka")
