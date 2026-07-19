# Copyright 2019 DeepMind Technologies Limited
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

"""Hand-shape, yaku, and scoring logic for Riichi Mahjong.

This module is deliberately independent of the game/state machinery in
mahjong_riichi.py so that hand evaluation can be unit-tested against known
reference hands on its own.

Tiles are represented throughout as integer *types* in [0, 34):
  0-8:   1m-9m  (man / characters)
  9-17:  1p-9p  (pin / circles)
  18-26: 1s-9s  (sou / bamboo)
  27-30: East, South, West, North winds
  31-33: White, Green, Red dragons

A "counts" array is a length-34 list/array of how many of each type are
present in some multiset of tiles (a hand, a wall, etc).

Scope note: this implements the standard scoring yaku used by the large
majority of modern Riichi rulesets (Tenhou-style), including all common
1-6 han yaku and the standard yakuman. A handful of rare local-rule yaku
(e.g. renhou, daisharin, paarenchan) are intentionally not included.
"""

import collections

NUM_TILE_TYPES = 34

MAN, PIN, SOU, HONOR = 0, 9, 18, 27
EAST, SOUTH, WEST, NORTH = 27, 28, 29, 30
WHITE, GREEN, RED = 31, 32, 33

WINDS = (EAST, SOUTH, WEST, NORTH)
DRAGONS = (WHITE, GREEN, RED)

TERMINALS = frozenset({0, 8, 9, 17, 18, 26})
HONORS = frozenset(range(27, 34))
TERMINALS_AND_HONORS = TERMINALS | HONORS
# The 13 "orphan" types used by kokushi musou (thirteen orphans).
ORPHANS = TERMINALS_AND_HONORS

_SUIT_NAMES = ("m", "p", "s")
_HONOR_NAMES = ("East", "South", "West", "North", "White", "Green", "Red")


def is_suit_tile(t):
  return t < 27


def suit_of(t):
  """Returns 0/1/2 for man/pin/sou, or None for honors."""
  return t // 9 if t < 27 else None


def number_of(t):
  """Returns 1-9 for suit tiles, or None for honors."""
  return (t % 9) + 1 if t < 27 else None


def is_terminal(t):
  return t in TERMINALS


def is_honor(t):
  return t in HONORS


def is_simple(t):
  """"Simples" (tanyao tiles): not a terminal and not an honor."""
  return t not in TERMINALS_AND_HONORS


def tile_name(t):
  if t < 27:
    return f"{number_of(t)}{_SUIT_NAMES[suit_of(t)]}"
  return _HONOR_NAMES[t - 27]


def counts_from_tiles(tiles):
  counts = [0] * NUM_TILE_TYPES
  for t in tiles:
    counts[t] += 1
  return counts


def tiles_from_counts(counts):
  tiles = []
  for t, c in enumerate(counts):
    tiles.extend([t] * c)
  return tiles


def dora_from_indicator(indicator):
  """Returns the dora tile type corresponding to a dora-indicator tile."""
  if indicator < 27:
    suit_base = (indicator // 9) * 9
    num = indicator % 9
    return suit_base + (num + 1) % 9
  elif indicator < 31:
    return EAST + (indicator - EAST + 1) % 4
  else:
    return WHITE + (indicator - WHITE + 1) % 3


# A meld is (kind, tile, is_concealed), where kind is one of "triplet",
# "sequence", "kan". For a sequence, `tile` is the *lowest* type in the run.
Meld = collections.namedtuple("Meld", ["kind", "tile", "is_concealed"])


def _decompose_sets(counts):
  """Yields tuples of ("triplet"|"sequence", type) that exactly consume counts.

  `counts` is mutated and restored in place (standard backtracking); the
  caller must not rely on it during iteration.
  """
  idx = 0
  while idx < NUM_TILE_TYPES and counts[idx] == 0:
    idx += 1
  if idx == NUM_TILE_TYPES:
    yield ()
    return
  if counts[idx] >= 3:
    counts[idx] -= 3
    for rest in _decompose_sets(counts):
      yield (("triplet", idx),) + rest
    counts[idx] += 3
  if (idx < 27 and (idx % 9) <= 6 and counts[idx + 1] > 0 and
      counts[idx + 2] > 0):
    counts[idx] -= 1
    counts[idx + 1] -= 1
    counts[idx + 2] -= 1
    for rest in _decompose_sets(counts):
      yield (("sequence", idx),) + rest
    counts[idx] += 1
    counts[idx + 1] += 1
    counts[idx + 2] += 1


def decompose_standard(concealed_counts):
  """Returns all (pair_type, sets) decompositions of a standard 4-sets+pair
  shape for the given concealed tile counts (sets is a tuple of
  ("triplet"|"sequence", type)).
  """
  counts = list(concealed_counts)
  results = []
  for pair_t in range(NUM_TILE_TYPES):
    if counts[pair_t] >= 2:
      counts[pair_t] -= 2
      for sets in _decompose_sets(counts):
        results.append((pair_t, sets))
      counts[pair_t] += 2
  return results


def is_chiitoitsu_shape(counts_14):
  """Seven pairs: exactly 7 distinct types, each with count == 2."""
  nonzero = [c for c in counts_14 if c != 0]
  return len(nonzero) == 7 and all(c == 2 for c in nonzero)


def is_kokushi_shape(counts_14):
  """Thirteen orphans: all 13 orphan types present, one of them doubled."""
  if sum(counts_14) != 14:
    return False
  for t in range(NUM_TILE_TYPES):
    if t in ORPHANS:
      if counts_14[t] == 0:
        return False
    elif counts_14[t] != 0:
      return False
  return sum(1 for t in ORPHANS if counts_14[t] == 2) == 1


def is_winning_shape(concealed_counts, open_melds):
  """True if concealed_counts (+ open_melds already removed) forms a win.

  Chiitoitsu/kokushi are only possible with zero open melds (both require a
  fully concealed hand).
  """
  if not open_melds and (is_chiitoitsu_shape(concealed_counts) or
                          is_kokushi_shape(concealed_counts)):
    return True
  return bool(decompose_standard(concealed_counts))


def get_waits(concealed_counts_13, open_melds):
  """Returns the set of tile types that would complete this (13-tile-
  equivalent) hand into a winning hand.
  """
  waits = set()
  for t in range(NUM_TILE_TYPES):
    if concealed_counts_13[t] >= 4:
      continue
    concealed_counts_13[t] += 1
    if is_winning_shape(concealed_counts_13, open_melds):
      waits.add(t)
    concealed_counts_13[t] -= 1
  return waits


class WinContext:
  """All information needed to score a winning hand."""

  # pylint: disable=too-many-arguments
  def __init__(self,
               concealed_counts,
               open_melds,
               winning_tile,
               is_tsumo,
               seat_wind,
               round_wind,
               is_riichi=False,
               is_double_riichi=False,
               is_ippatsu=False,
               is_haitei=False,
               is_houtei=False,
               is_rinshan=False,
               is_chankan=False,
               is_tenhou=False,
               is_chiihou=False,
               dora_indicators=(),
               ura_dora_indicators=(),
               aka_dora_count=0):
    self.concealed_counts = list(concealed_counts)
    self.open_melds = list(open_melds)
    self.winning_tile = winning_tile
    self.is_tsumo = is_tsumo
    self.seat_wind = seat_wind
    self.round_wind = round_wind
    self.is_riichi = is_riichi
    self.is_double_riichi = is_double_riichi
    self.is_ippatsu = is_ippatsu
    self.is_haitei = is_haitei
    self.is_houtei = is_houtei
    self.is_rinshan = is_rinshan
    self.is_chankan = is_chankan
    self.is_tenhou = is_tenhou
    self.is_chiihou = is_chiihou
    self.dora_indicators = list(dora_indicators)
    self.ura_dora_indicators = list(ura_dora_indicators)
    self.aka_dora_count = aka_dora_count

  @property
  def is_menzen(self):
    """Fully concealed: no open melds (closed kans do not break menzen)."""
    return all(m.is_concealed for m in self.open_melds)


class ScoreResult:
  """The outcome of scoring a winning hand."""

  def __init__(self, yaku, han, fu, is_yakuman, yakuman_multiplier=0):
    self.yaku = yaku  # list of (name, han) pairs, han==0 marks dora entries
    self.han = han
    self.fu = fu
    self.is_yakuman = is_yakuman
    self.yakuman_multiplier = yakuman_multiplier

  def __repr__(self):
    return (f"ScoreResult(yaku={self.yaku}, han={self.han}, fu={self.fu}, "
            f"is_yakuman={self.is_yakuman})")


# -----------------------------------------------------------------------
# Wait-type / fu computation for a single (pair, sets) decomposition.
# -----------------------------------------------------------------------


def _wait_role(pair_t, sets, winning_tile):
  """Classifies how `winning_tile` completes this decomposition.

  Returns (wait_fu, role), where role is one of:
    ("pair", None)                        -- tanki
    ("triplet", tile)                     -- shanpon
    ("sequence", seq_start, "kanchan")
    ("sequence", seq_start, "penchan")
    ("sequence", seq_start, "ryanmen")
  """
  if winning_tile == pair_t:
    return 2, ("pair", None)
  for kind, t in sets:
    if kind == "triplet" and t == winning_tile:
      return 0, ("triplet", t)
    if kind == "sequence" and t <= winning_tile <= t + 2:
      rel = winning_tile - t
      if rel == 1:
        return 2, ("sequence", t, "kanchan")
      elif rel == 0:
        is_edge = number_of(t + 2) == 9
        return (2, ("sequence", t, "penchan")) if is_edge else (
            0, ("sequence", t, "ryanmen"))
      else:  # rel == 2
        is_edge = number_of(t) == 1
        return (2, ("sequence", t, "penchan")) if is_edge else (
            0, ("sequence", t, "ryanmen"))
  raise ValueError(
      f"winning tile {winning_tile} not present in decomposition "
      f"(pair={pair_t}, sets={sets})")


def _meld_fu(kind, tile, is_concealed):
  """Fu contributed by one triplet/kan; sequences contribute 0."""
  if kind == "sequence":
    return 0
  honor_or_terminal = is_terminal(tile) or is_honor(tile)
  if kind == "triplet":
    base = 4 if honor_or_terminal else 2
  else:
    assert kind == "kan"
    base = 16 if honor_or_terminal else 8
  return base * (2 if is_concealed else 1)


def _is_pinfu(pair_t, sets, open_melds, wait_role, seat_wind, round_wind):
  if open_melds:
    return False
  if any(kind == "triplet" for kind, _ in sets):
    return False
  if pair_t in DRAGONS or pair_t == seat_wind or pair_t == round_wind:
    return False
  return wait_role[0] == "sequence" and wait_role[2] == "ryanmen"


def compute_fu(pair_t, sets, open_melds, winning_tile, is_tsumo, seat_wind,
                round_wind):
  """Computes fu for one (pair, sets) decomposition of a standard-shape win."""
  wait_fu, role = _wait_role(pair_t, sets, winning_tile)
  is_menzen = all(m.is_concealed for m in open_melds)
  pinfu = _is_pinfu(pair_t, sets, open_melds, role, seat_wind, round_wind)

  if pinfu and is_tsumo:
    return 20, role, True
  if pinfu and not is_tsumo:
    return 30, role, True

  fu = 20 + wait_fu
  if is_tsumo:
    fu += 2
  elif is_menzen:
    fu += 10  # Menzen ron bonus.

  if pair_t in DRAGONS:
    fu += 2
  if pair_t == seat_wind:
    fu += 2
  if pair_t == round_wind:
    fu += 2

  for kind, t in sets:
    if kind != "triplet":
      continue
    if role[0] == "triplet" and role[1] == t and not is_tsumo:
      concealed = False  # Shanpon completed by ron: counted as open (minko).
    else:
      concealed = True  # A pre-existing concealed triplet (ankou).
    fu += _meld_fu("triplet", t, concealed)

  for m in open_melds:
    fu += _meld_fu(m.kind, m.tile, m.is_concealed)

  return fu, role, pinfu


def _round_fu(fu):
  return ((fu + 9) // 10) * 10


# -----------------------------------------------------------------------
# Yaku detection.
# -----------------------------------------------------------------------


def _all_melds(sets, open_melds):
  """Combines concealed `sets` and `open_melds` into one list of
  (kind, tile, is_concealed) triples, for checks that don't care about the
  concealed/open distinction beyond what's already encoded."""
  combined = [(kind, t, True) for kind, t in sets]
  combined.extend((m.kind, m.tile, m.is_concealed) for m in open_melds)
  return combined


def _sequence_starts(sets, open_melds):
  starts = [t for kind, t in sets if kind == "sequence"]
  starts.extend(m.tile for m in open_melds if m.kind == "sequence")
  return starts


def _triplet_or_kan_tiles(sets, open_melds):
  tiles = [t for kind, t in sets if kind == "triplet"]
  tiles.extend(m.tile for m in open_melds if m.kind in ("triplet", "kan"))
  return tiles


def _all_hand_tiles(pair_t, sets, open_melds):
  """All 14 logical tile-type occurrences (kans counted as a triplet)."""
  tiles = [pair_t, pair_t]
  for kind, t in sets:
    if kind == "sequence":
      tiles.extend([t, t + 1, t + 2])
    else:
      tiles.extend([t, t, t])
  for m in open_melds:
    if m.kind == "sequence":
      tiles.extend([m.tile, m.tile + 1, m.tile + 2])
    else:
      tiles.extend([m.tile, m.tile, m.tile])
  return tiles


_GREEN_TILES = frozenset({19, 20, 21, 23, 25, GREEN})  # 2s,3s,4s,6s,8s,green


def _check_yaku(pair_t, sets, wait_role, pinfu, context):
  """Returns a list of (name, han) for one decomposition, excluding yakuman
  and dora (handled by the caller)."""
  open_melds = context.open_melds
  is_menzen = context.is_menzen
  yaku = []

  if context.is_double_riichi:
    yaku.append(("Double Riichi", 2))
  elif context.is_riichi:
    yaku.append(("Riichi", 1))
  if context.is_riichi and context.is_ippatsu:
    yaku.append(("Ippatsu", 1))
  if context.is_tsumo and is_menzen:
    yaku.append(("Menzen Tsumo", 1))
  if pinfu:
    yaku.append(("Pinfu", 1))
  if context.is_haitei:
    yaku.append(("Haitei Raoyue", 1))
  if context.is_houtei:
    yaku.append(("Houtei Raoyui", 1))
  if context.is_rinshan:
    yaku.append(("Rinshan Kaihou", 1))
  if context.is_chankan:
    yaku.append(("Chankan", 1))

  all_tiles = _all_hand_tiles(pair_t, sets, open_melds)
  if all(is_simple(t) for t in all_tiles):
    yaku.append(("Tanyao", 1))

  if is_menzen:
    seq_starts = _sequence_starts(sets, ())
    counts = collections.Counter(seq_starts)
    if any(c >= 2 for c in counts.values()):
      yaku.append(("Iipeikou", 1))

  for t in _triplet_or_kan_tiles(sets, open_melds):
    if t in DRAGONS:
      yaku.append((f"Yakuhai ({tile_name(t)})", 1))
    if t == context.seat_wind:
      yaku.append((f"Yakuhai (seat {tile_name(t)})", 1))
    if t == context.round_wind:
      yaku.append((f"Yakuhai (round {tile_name(t)})", 1))

  all_seq_starts = _sequence_starts(sets, open_melds)
  relative_starts = {s % 9 for s in all_seq_starts if s < 27}
  if any(
      all((suit * 9 + rel) in all_seq_starts for suit in range(3))
      for rel in relative_starts):
    yaku.append(("Sanshoku Doujun", 1 if open_melds else 2))
  for suit in range(3):
    base = suit * 9
    needed = (base, base + 3, base + 6)
    if all(n in all_seq_starts for n in needed):
      yaku.append(("Ittsu", 1 if open_melds else 2))

  groups_tiles = [[pair_t, pair_t]]
  for kind, t in sets:
    groups_tiles.append([t, t + 1, t + 2] if kind == "sequence" else [t, t, t])
  for m in open_melds:
    groups_tiles.append(
        [m.tile, m.tile + 1, m.tile + 2]
        if m.kind == "sequence" else [m.tile, m.tile, m.tile])
  all_have_terminal_or_honor = all(
      any(is_terminal(t) or is_honor(t) for t in g) for g in groups_tiles)
  all_have_honor = any(is_honor(t) for g in groups_tiles for t in g)
  if all_have_terminal_or_honor:
    if all_have_honor:
      yaku.append(("Chanta", 1 if open_melds else 2))
    else:
      yaku.append(("Junchan", 2 if open_melds else 3))

  num_sequences = (sum(1 for kind, _ in sets if kind == "sequence") +
                   sum(1 for m in open_melds if m.kind == "sequence"))
  if num_sequences == 0:
    yaku.append(("Toitoi", 2))

  ankou_count = 0
  for kind, t in sets:
    if kind == "triplet":
      if not (wait_role[0] == "triplet" and wait_role[1] == t and
              not context.is_tsumo):
        ankou_count += 1
  ankou_count += sum(
      1 for m in open_melds if m.kind == "kan" and m.is_concealed)
  if ankou_count >= 3:
    yaku.append(("Sanankou", 2))

  kan_count = sum(1 for m in open_melds if m.kind == "kan")
  if kan_count == 3:
    yaku.append(("Sankantsu", 2))

  if num_sequences == 0 and all(t in TERMINALS_AND_HONORS
                                 for g in groups_tiles for t in g):
    yaku.append(("Honroutou", 2))

  dragon_triplets = [t for t in _triplet_or_kan_tiles(sets, open_melds)
                      if t in DRAGONS]
  if len(dragon_triplets) == 2 and pair_t in DRAGONS:
    yaku.append(("Shousangen", 2))

  suits_present = {suit_of(t) for t in all_tiles if is_suit_tile(t)}
  has_honor_tile = any(is_honor(t) for t in all_tiles)
  if len(suits_present) == 1:
    if has_honor_tile:
      yaku.append(("Honitsu", 2 if open_melds else 3))
    else:
      yaku.append(("Chinitsu", 5 if open_melds else 6))

  return yaku


def _check_yakuman(pair_t, sets, open_melds, all_tiles, context, num_kans,
                    role):
  """Returns a list of (name, multiplier) yakuman achieved by a standard-shape
  decomposition (kokushi is handled separately by the caller)."""
  results = []

  ankou_count = 0
  for kind, t in sets:
    if kind != "triplet":
      continue
    completed_by_ron_here = (
        role[0] == "triplet" and role[1] == t and not context.is_tsumo)
    if not completed_by_ron_here:
      ankou_count += 1
  ankou_count += sum(
      1 for m in open_melds if m.kind == "kan" and m.is_concealed)
  if ankou_count >= 4:
    results.append(("Suuankou", 1))

  dragon_triplets = [t for t in _triplet_or_kan_tiles(sets, open_melds)
                      if t in DRAGONS]
  if len(dragon_triplets) == 3:
    results.append(("Daisangen", 1))

  wind_triplets = [t for t in _triplet_or_kan_tiles(sets, open_melds)
                    if t in WINDS]
  if len(wind_triplets) == 4:
    results.append(("Daisuushii", 1))
  elif len(wind_triplets) == 3 and pair_t in WINDS:
    results.append(("Shousuushii", 1))

  if all(is_honor(t) for t in all_tiles):
    results.append(("Tsuuiisou", 1))
  num_sequences = (sum(1 for kind, _ in sets if kind == "sequence") +
                   sum(1 for m in open_melds if m.kind == "sequence"))
  if num_sequences == 0 and all(is_terminal(t) for t in all_tiles):
    results.append(("Chinroutou", 1))
  if all(t in _GREEN_TILES for t in all_tiles):
    results.append(("Ryuuiisou", 1))
  if num_kans == 4:
    results.append(("Suukantsu", 1))

  if context.is_tenhou:
    results.append(("Tenhou", 1))
  if context.is_chiihou:
    results.append(("Chiihou", 1))

  return results


def _chuuren_poutou(concealed_counts, open_melds):
  if open_melds:
    return False
  suits_present = {suit_of(t) for t in range(NUM_TILE_TYPES)
                    if concealed_counts[t] > 0 and is_suit_tile(t)}
  has_honor = any(concealed_counts[t] > 0 for t in HONORS)
  if has_honor or len(suits_present) != 1:
    return False
  suit = next(iter(suits_present))
  base = suit * 9
  required = [3, 1, 1, 1, 1, 1, 1, 1, 3]
  diff = [concealed_counts[base + i] - required[i] for i in range(9)]
  if any(d < 0 for d in diff):
    return False
  extra = [i for i, d in enumerate(diff) if d == 1]
  return sum(diff) == 1 and len(extra) == 1


def _count_dora(tiles, indicators):
  """Counts dora among `tiles` (a flat list of tile-type occurrences), one
  indicator at a time so that multiple indicators pointing at the same dora
  tile each count separately, as per the standard rule."""
  total = 0
  for indicator in indicators:
    dora_tile = dora_from_indicator(indicator)
    total += tiles.count(dora_tile)
  return total


def _evaluate_chiitoitsu(context):
  """Scores a seven-pairs (chiitoitsu) win. Always fully concealed."""
  all_tiles = tiles_from_counts(context.concealed_counts)
  yaku = [("Chiitoitsu", 2)]
  if context.is_double_riichi:
    yaku.append(("Double Riichi", 2))
  elif context.is_riichi:
    yaku.append(("Riichi", 1))
  if context.is_riichi and context.is_ippatsu:
    yaku.append(("Ippatsu", 1))
  if context.is_tsumo:
    yaku.append(("Menzen Tsumo", 1))
  if context.is_haitei:
    yaku.append(("Haitei Raoyue", 1))
  if context.is_houtei:
    yaku.append(("Houtei Raoyui", 1))
  if context.is_tenhou:
    yaku.append(("Tenhou", 1))
  if context.is_chiihou:
    yaku.append(("Chiihou", 1))
  if all(is_simple(t) for t in all_tiles):
    yaku.append(("Tanyao", 1))
  if all(t in TERMINALS_AND_HONORS for t in all_tiles):
    yaku.append(("Honroutou", 2))
  suits_present = {suit_of(t) for t in all_tiles if is_suit_tile(t)}
  has_honor = any(is_honor(t) for t in all_tiles)
  if len(suits_present) == 1:
    yaku.append(("Honitsu", 3) if has_honor else ("Chinitsu", 6))

  han = sum(h for _, h in yaku)
  dora_count = _count_dora(all_tiles, context.dora_indicators)
  ura_count = (_count_dora(all_tiles, context.ura_dora_indicators)
               if context.is_riichi else 0)
  if dora_count:
    yaku.append(("Dora", dora_count))
  if ura_count:
    yaku.append(("Ura Dora", ura_count))
  if context.aka_dora_count:
    yaku.append(("Aka Dora", context.aka_dora_count))
  han += dora_count + ura_count + context.aka_dora_count
  return ScoreResult(yaku=yaku, han=han, fu=25, is_yakuman=False)


def evaluate_hand(context):
  """Scores a winning hand described by `context` (a WinContext).

  Returns a ScoreResult, or None if the hand (as described) is not actually
  a valid win (wrong shape, or a shape with no yaku -- e.g. an unlucky
  non-ryanmen all-sequence hand that narrowly misses pinfu).
  """
  counts14 = list(context.concealed_counts)
  open_melds = context.open_melds
  winning_tile = context.winning_tile

  if not open_melds and is_kokushi_shape(counts14):
    is_thirteen_wait = counts14[winning_tile] == 2 and sum(
        1 for t in ORPHANS if counts14[t] == 2 and t != winning_tile) == 0
    del is_thirteen_wait  # Double-yakuman 13-wait variant not implemented.
    return ScoreResult(
        yaku=[("Kokushi Musou", 1)], han=13, fu=0, is_yakuman=True,
        yakuman_multiplier=1)

  if not open_melds and is_chiitoitsu_shape(counts14):
    return _evaluate_chiitoitsu(context)

  decomps = decompose_standard(counts14)
  if not decomps:
    return None

  num_kans = sum(1 for m in open_melds if m.kind == "kan")
  best = None
  best_is_yakuman = False

  for pair_t, sets in decomps:
    fu, role, pinfu = compute_fu(pair_t, sets, open_melds, winning_tile,
                                  context.is_tsumo, context.seat_wind,
                                  context.round_wind)
    all_tiles = _all_hand_tiles(pair_t, sets, open_melds)

    yakuman_list = _check_yakuman(pair_t, sets, open_melds, all_tiles,
                                   context, num_kans, role)
    if _chuuren_poutou(context.concealed_counts, open_melds):
      yakuman_list.append(("Chuuren Poutou", 1))

    if yakuman_list:
      mult = sum(m for _, m in yakuman_list)
      if not best_is_yakuman or mult > best.yakuman_multiplier:
        best = ScoreResult(yaku=yakuman_list, han=13 * mult, fu=0,
                            is_yakuman=True, yakuman_multiplier=mult)
        best_is_yakuman = True
      continue
    if best_is_yakuman:
      continue

    yaku_list = _check_yaku(pair_t, sets, role, pinfu, context)
    if not yaku_list:
      continue  # A hand needs at least one yaku to be a valid win.

    han = sum(h for _, h in yaku_list)
    dora_count = _count_dora(all_tiles, context.dora_indicators)
    ura_count = (_count_dora(all_tiles, context.ura_dora_indicators)
                 if context.is_riichi else 0)
    full_yaku = list(yaku_list)
    if dora_count:
      full_yaku.append(("Dora", dora_count))
    if ura_count:
      full_yaku.append(("Ura Dora", ura_count))
    if context.aka_dora_count:
      full_yaku.append(("Aka Dora", context.aka_dora_count))
    total_han = han + dora_count + ura_count + context.aka_dora_count
    rounded_fu = _round_fu(fu)

    if best is None or (total_han, rounded_fu) > (best.han, best.fu):
      best = ScoreResult(yaku=full_yaku, han=total_han, fu=rounded_fu,
                          is_yakuman=False)

  return best


# -----------------------------------------------------------------------
# Points and payments.
# -----------------------------------------------------------------------


def base_points(han, fu):
  """Base points before the tsumo/ron payment multiplier is applied."""
  if han >= 11:
    return 6000
  if han >= 8:
    return 4000
  if han >= 6:
    return 3000
  return min(fu * (2**(2 + han)), 2000)


def _round_up_100(x):
  return -(-int(x) // 100) * 100


def compute_payments(score_result, is_dealer, is_tsumo, honba=0):
  """Returns a dict describing how points change hands for this win.

  For a tsumo win: {"dealer": amount_from_dealer, "non_dealer":
  amount_from_each_non_dealer} if the winner isn't the dealer, or
  {"each": amount_from_each_of_the_three_others} if the winner is the
  dealer.
  For a ron win: {"loser": amount_paid_by_the_discarder}.
  """
  if score_result.is_yakuman:
    base = 8000 * score_result.yakuman_multiplier
  else:
    base = base_points(score_result.han, score_result.fu)

  if is_tsumo:
    if is_dealer:
      return {"each": _round_up_100(base * 2) + honba * 100}
    else:
      return {
          "dealer": _round_up_100(base * 2) + honba * 100,
          "non_dealer": _round_up_100(base) + honba * 100,
      }
  else:
    multiplier = 6 if is_dealer else 4
    return {"loser": _round_up_100(base * multiplier) + honba * 300}
