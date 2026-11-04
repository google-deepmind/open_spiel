# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The gin card-token layout: how CardTokenTorso reads its input vector.

The vector is written by the gin card-token encoder
(`encoders.gin_card_tokens`, which imports these constants and asserts that
its own derived sizes agree, so they exist once) and read by
`networks.gin_tfv2.CardTokenTorso`. The torso needs only these constants and
`split`.

Layout: [card block 52 x card_fields | global block global_fields |
touch block TOUCH_ROWS x TOUCH_FIELDS], each touch row (card + 1, kind,
t + 1).
"""

import jax.numpy as jnp

NUM_CARDS = 52
EVENTS_MAX = 160
TOUCH_ROWS = 189
TOUCH_FIELDS = 3
KIND_PAD = 0
NUM_KINDS = 12
OPP_KINDS = (2, 4, 6, 8, 10)
NUM_TOUCH_CARD_CODES = NUM_CARDS + 1  # 0 = no card (knock)
NUM_TIME_CODES = EVENTS_MAX + 1
CARD_FIELD_OPP_KNOWN = 2
CARD_FIELD_UPCARD = 3


def dim(card_fields: int, global_fields: int) -> int:
  """The vector's width for a card and global field count.

  Args:
    card_fields: the fields per card.
    global_fields: the global fields.

  Returns:
    The width.
  """
  return NUM_CARDS * card_fields + global_fields + TOUCH_ROWS * TOUCH_FIELDS


def split(obs, card_fields: int, global_fields: int):
  """Splits [..., dim] f32 vectors into their three blocks.

  Args:
    obs: f32[..., dim(card_fields, global_fields)].
    card_fields: the fields per card.
    global_fields: the global fields.

  Returns:
    (cards f32[..., 52, card_fields], globals f32[..., global_fields],
    touch rows i32[..., 189, 3]).

  Raises:
    ValueError: the width is not the layout's.
  """
  want = dim(card_fields, global_fields)
  if obs.shape[-1] != want:
    raise ValueError(
        f"card tokens: got width {obs.shape[-1]}, layout "
        f"(card_fields={card_fields}, global_fields={global_fields}) is {want}"
    )
  lead = obs.shape[:-1]
  c = NUM_CARDS * card_fields
  cards = obs[..., :c].reshape(lead + (NUM_CARDS, card_fields))
  glob = obs[..., c : c + global_fields]
  touch = obs[..., c + global_fields :].reshape(
      lead + (TOUCH_ROWS, TOUCH_FIELDS)
  )
  return cards, glob, jnp.round(touch).astype(jnp.int32)
