"""Training utilities. Mainly MLPs and optimization steps."""

from typing import Callable

import flax.linen
import jax
import jax.numpy as jnp
import optax


class MLP(flax.linen.Module):
    act_fn: Callable
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


def train_step(*, list_of_keys, model, loss, params, opt_state, optimizer):
    ret, grads = jax.value_and_grad(loss, argnums=1)(list_of_keys, params)
    params_update, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, params_update)
    return params, opt_state, ret
