import functools
from typing import Callable

import flax.linen
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tqdm
import powerpoint

import imports

# Training parameters
num_samples = 100_000
num_epochs_b = 10000
learning_rate_b = 0.001
batch_size = 1_000

# Generation parameters
num_generates = 10_000
dt = 0.01
epsilon = 1.0


# The one and only magic key
prng_key = jax.random.PRNGKey(2)


# Set up the problem: sample from mixtures of Gaussians

x_shape = (2,)



def sample_marco(key, *, width):
    m, a, r, c, o, key_idx = jax.random.split(key, num=6)

    # Generate points for each letter with adjusted parameters
    marco_m_points = powerpoint.generate_m(m, width)
    marco_a_points = powerpoint.generate_a(a, width)
    marco_r_points = powerpoint.generate_r(r, width)
    marco_c_points = powerpoint.generate_c(c, width)
    marco_o_points = powerpoint.generate_o(o, width)

    idx = jax.random.choice(key, jnp.arange(0, 5, step=1))
    word = jnp.stack(
        [
            marco_m_points,
            marco_a_points,
            marco_r_points,
            marco_c_points,
            marco_o_points,
        ],
        axis=0,
    )
    return word[idx]


def sample_nico(key, *, width):
    n, i, c, o, key_idx = jax.random.split(key, num=5)

    # Generate points for each letter with adjusted parameters
    nico_n_points = powerpoint.generate_n(n, width)
    nico_i_points = powerpoint.generate_i(i, width)
    nico_c_points = powerpoint.generate_c(c, width)
    nico_o_points = powerpoint.generate_o(o, width)

    idx = jax.random.choice(key, jnp.arange(0, 4, step=1))
    word = jnp.stack(
        [nico_n_points, nico_i_points, nico_c_points, nico_o_points], axis=0
    )
    return word[idx]


sample_rho1 = functools.partial(sample_nico, width=0.25)
sample_rho0 = functools.partial(sample_marco, width=0.25)



# m0 = jnp.asarray([[-10, -10], [0, 0], [10, 10]])
# m1 = jnp.asarray([[-5, 5], [5, -5]])
# sample_rho0 = functools.partial(sample_from_mean, mean=m0)
# sample_rho1 = functools.partial(sample_from_mean, mean=m1)


model_b = imports.MLP(
    output_dim=1,
    num_layers=1,
    hidden_dim=100,
    act_fn=jax.nn.relu,
)



# Initialize the model parameters

prng_key, prng_key_in, prng_key_out = jax.random.split(prng_key, num=3)
params_b = model_b.init(prng_key, jnp.ones((2,)))




inputs = jax.vmap(sample_rho0)(jax.random.split(prng_key_in, num=num_samples))
outputs = jax.vmap(sample_rho1)(jax.random.split(prng_key_out, num=num_samples))

@jax.jit
def mse(params, x_batched, y_batched):
    # Define the squared loss for a single pair (x,y)
    def squared_error(x, y):
        pred = model_b.apply(params, x).reshape(x_shape)
        return jnp.inner(y - pred, y - pred) / 2.0

    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)



optimizer_b = optax.adam(learning_rate_b)
opt_state_b = optimizer_b.init(params_b)
pbar = tqdm.tqdm(range(num_epochs_b))
for epoch in pbar:
    try:
        prng_key, _ = jax.random.split(prng_key, num=2)
        batch = jax.random.choice(prng_key, jnp.arange(0, len(inputs)), shape=(batch_size,), replace=False )
        ret, grads = jax.value_and_grad(mse, argnums=0)(params_b, inputs[batch], outputs[batch])
        params_update, opt_state_b = optimizer_b.update(grads, opt_state_b)
        params_b = optax.apply_updates(params_b, params_update)

        pbar.set_description(f"Loss: {ret:.1f}")
    except KeyboardInterrupt:
        break 



prng_key, _ = jax.random.split(prng_key, num=2)
inputs = jax.vmap(sample_rho0)(jax.random.split(prng_key, num=num_generates))
outputs = jax.vmap(model_b.apply, in_axes=(None, 0))(params_b, inputs)


plt.scatter(inputs[:, 0], inputs[:, 1])
plt.scatter(outputs[:, 0] + 20, outputs[:, 1])
plt.show()