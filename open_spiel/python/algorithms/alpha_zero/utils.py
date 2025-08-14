import jax.numpy as jnp
import chex

def api_selector(api_version):
  if api_version == "nnx":
    from open_spiel.python.algorithms.alpha_zero import model_jax as model_lib
  elif api_version == "linen":
    from open_spiel.python.algorithms.alpha_zero import model_nnx as model_lib
  else:
    raise ValueError("Only `linen` and `nnx` APIs are implmented")
  
  return model_lib


def flatten(x):
  return x.reshape((x.shape[0], -1))

@chex.dataclass(frozen=True)
class TrainInput: 
  """Inputs of the model: o_t, mask_t, π(a_t|o_t), v_t
  """
  observation: chex.Array
  legals_mask: chex.Array 
  policy: chex.Array 
  value: chex.Array

  @staticmethod
  def stack(train_inputs):

    observation, legals_mask, policy, value = zip(*[
      (ti.observation, ti.legals_mask, ti.policy, ti.value) for ti in train_inputs
    ])
    return TrainInput(
        observation=jnp.array(observation, dtype=jnp.float32),
        legals_mask=jnp.array(legals_mask, dtype=jnp.bool),
        policy=jnp.array(policy, dtype=jnp.float32),
        value=jnp.expand_dims(jnp.array(value, dtype=jnp.float32), 1))

@chex.dataclass(frozen=True)
class Losses:
  """Losses: policy, value, L2
  """
  policy: chex.Array
  value: chex.Array 
  l2: chex.Array 
  
  @property
  def total(self):
    return self.policy + self.value + self.l2

  def __str__(self):
    return ("Losses(total: {:.3f}, policy: {:.3f}, value: {:.3f}, "
            "l2: {:.3f})").format(self.total, self.policy, self.value, self.l2)

  def __add__(self, other):
    return Losses(
      policy=self.policy + other.policy,
      value=self.value + other.value,
      l2=self.l2 + other.l2
    )

  def __truediv__(self, n):
    return Losses(policy=self.policy / n, value=self.value / n, l2=self.l2 / n)
