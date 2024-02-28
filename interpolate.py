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
num_samples = 10
num_epochs_b = 10
num_epochs_s = 10
learning_rate_b = 0.001
learning_rate_s = 0.001


# Generation parameters
num_generates = 10
dt = 0.01
epsilon = 1.0


# The one and only magic key
prng_key = jax.random.PRNGKey(1)


# Set up the problem: sample from mixtures of Gaussians

x_shape = (2,)


def sample_from_mean(key, *, mean):
    key_idx, key_g = jax.random.split(key, num=2)
    idx = jax.random.choice(key_idx, jnp.arange(0, len(mean), step=1))
    return jax.random.normal(key_g, shape=x_shape) + mean[idx]


m0 = jnp.asarray([[-10, -10], [0, 0], [10, 10]])
m1 = jnp.asarray([[-5, 5], [5, -5]])
sample_rho0 = functools.partial(sample_from_mean, mean=m0)
sample_rho1 = functools.partial(sample_from_mean, mean=m1)


model_b = imports.MLP(
    output_dim=x_shape[0],
    num_layers=2,
    hidden_dim=20,
    act_fn=jax.nn.tanh,
)
model_s = imports.MLP(
    output_dim=x_shape[0],
    num_layers=2,
    hidden_dim=20,
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


inputs = jax.vmap(sample_rho0)(jax.random.split(prng_key, num=100))
outputs = jax.vmap(sample_rho1)(jax.random.split(prng_key, num=100))


@jax.jit
def mse(params, x_batched, y_batched):
    # Define the squared loss for a single pair (x,y)
    def squared_error(x, y):
        pred = model.apply(params, x)
        return jnp.inner(y - pred, y - pred) / 2.0

    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)


print(inputs.shape, outputs.shape)
