import itertools

import flax.nnx as nn
import pyspiel
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from open_spiel.python.jax.nes import nes
from open_spiel.python.jax.nes import deb
from open_spiel.python.jax.nes import networks
from open_spiel.python.jax.nes import utils