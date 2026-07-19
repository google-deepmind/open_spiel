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

"""Tests for mahjong_riichi_utils.py, against known reference hands."""

from absl.testing import absltest

from open_spiel.python.games import mahjong_riichi_utils as u


def _evaluate(tiles14, winning_tile, is_tsumo, seat=u.EAST, round_=u.EAST,
              open_melds=(), **kwargs):
  concealed = u.counts_from_tiles(tiles14)
  context = u.WinContext(concealed, list(open_melds), winning_tile, is_tsumo,
                          seat, round_, **kwargs)
  return u.evaluate_hand(context)


def _yaku_names(result):
  return {name for name, _ in result.yaku}


class HandShapeTest(absltest.TestCase):

  def test_decompose_standard_unique(self):
    tiles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 18, 19, 20]
    decomps = u.decompose_standard(u.counts_from_tiles(tiles))
    self.assertLen(decomps, 1)
    pair, sets = decomps[0]
    self.assertEqual(pair, 9)
    self.assertCountEqual(
        sets, [("sequence", 0), ("sequence", 3), ("sequence", 6),
               ("sequence", 18)])

  def test_chiitoitsu_shape(self):
    tiles = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    self.assertTrue(u.is_chiitoitsu_shape(u.counts_from_tiles(tiles)))

  def test_kokushi_shape(self):
    tiles = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 33]
    self.assertTrue(u.is_kokushi_shape(u.counts_from_tiles(tiles)))

  def test_non_winning_shape_rejected(self):
    tiles = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 27, 28, 29, 30]
    self.assertFalse(u.is_winning_shape(u.counts_from_tiles(tiles), []))

  def test_ryanmen_waits(self):
    tiles13 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 19, 20]
    waits = u.get_waits(u.counts_from_tiles(tiles13), [])
    self.assertEqual(waits, {18, 21})  # 1s or 4s.


class ScoringTest(absltest.TestCase):

  def test_no_yaku_hand_is_rejected(self):
    """All-sequence, kanchan-wait (breaks pinfu), ron (no menzen tsumo),
    non-yakuhai pair, with terminals present (breaks tanyao) -- genuinely
    zero yaku, so this must return None: a hand needs >=1 yaku to win."""
    tiles = [0, 1, 2, 24, 25, 26, 10, 10, 12, 13, 14, 3, 4, 5]
    result = _evaluate(tiles, 4, is_tsumo=False)
    self.assertIsNone(result)

  def test_pinfu_tsumo_tanyao(self):
    tiles = [1, 2, 3, 5, 6, 7, 12, 13, 14, 16, 16, 19, 20, 21]
    result = _evaluate(tiles, 21, is_tsumo=True)
    self.assertIsNotNone(result)
    self.assertEqual(_yaku_names(result),
                      {"Menzen Tsumo", "Pinfu", "Tanyao"})
    self.assertEqual(result.han, 3)
    self.assertEqual(result.fu, 20)

  def test_pinfu_ron_is_30_fu(self):
    tiles = [1, 2, 3, 5, 6, 7, 12, 13, 14, 16, 16, 19, 20, 21]
    result = _evaluate(tiles, 21, is_tsumo=False)
    self.assertEqual(result.fu, 30)

  def test_yakuhai_kanchan_ron(self):
    tiles = [1, 1, 11, 12, 13, 23, 24, 25, 31, 31, 31, 3, 4, 5]
    result = _evaluate(tiles, 4, is_tsumo=False)
    self.assertEqual(_yaku_names(result), {"Yakuhai (White)"})
    self.assertEqual(result.han, 1)
    self.assertEqual(result.fu, 40)

  def test_double_wind_pair_fu(self):
    """Seat wind == round wind: the East pair should score fu twice (2+2)."""
    tiles = [1, 2, 3, 13, 14, 15, 19, 20, 21, 27, 27, 6, 7, 8]
    result = _evaluate(tiles, 8, is_tsumo=False, seat=u.EAST, round_=u.EAST,
                        is_riichi=True)
    # 20 base + 0 (ryanmen wait) + 10 (menzen ron) + 4 (double-wind pair),
    # rounded up to the nearest 10.
    self.assertEqual(result.fu, 40)

  def test_iipeikou_requires_menzen(self):
    tiles = [1, 2, 3, 1, 2, 3, 12, 13, 14, 16, 16, 19, 20, 21]
    result = _evaluate(tiles, 21, is_tsumo=True)
    self.assertIn("Iipeikou", _yaku_names(result))

  def test_sanshoku_doujun_any_start(self):
    """Regression test: sanshoku must be detected for non-123 starts too."""
    tiles = [1, 2, 3, 10, 11, 12, 19, 20, 21, 16, 16, 3, 4, 5]
    result = _evaluate(tiles, 4, is_tsumo=False)
    self.assertIn("Sanshoku Doujun", _yaku_names(result))
    self.assertEqual(dict(result.yaku)["Sanshoku Doujun"], 2)  # closed value

  def test_ittsu(self):
    tiles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 16, 19, 20, 21]
    result = _evaluate(tiles, 21, is_tsumo=True)
    self.assertIn("Ittsu", _yaku_names(result))
    self.assertNotIn("Tanyao", _yaku_names(result))  # 1m/9m are terminals.

  def test_toitoi(self):
    # Ron (not tsumo) on the shanpon pair-of-20s: this downgrades that one
    # triplet to "open" for ankou-counting, so it's Toitoi + Sanankou (3
    # concealed triplets), not all the way up to Suuankou.
    tiles = [1, 1, 1, 5, 5, 5, 12, 12, 12, 20, 20, 20, 16, 16]
    result = _evaluate(tiles, 20, is_tsumo=False)
    self.assertIn("Toitoi", _yaku_names(result))
    self.assertIn("Sanankou", _yaku_names(result))
    self.assertFalse(result.is_yakuman)

  def test_suuankou_yakuman_via_tsumo(self):
    # Same shape, but tsumo: even the shanpon-completed triplet counts as
    # concealed, giving four ankou -> Suuankou (which then suppresses the
    # normal Toitoi/Sanankou yaku, per standard rules).
    tiles = [1, 1, 1, 5, 5, 5, 12, 12, 12, 20, 20, 20, 16, 16]
    result = _evaluate(tiles, 20, is_tsumo=True)
    self.assertTrue(result.is_yakuman)
    self.assertEqual(_yaku_names(result), {"Suuankou"})

  def test_honitsu_and_chinitsu(self):
    honitsu_tiles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 27, 27, 31, 31]
    result = _evaluate(honitsu_tiles, 31, is_tsumo=True)
    self.assertIn("Honitsu", _yaku_names(result))

    chinitsu_tiles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 1, 1, 1]
    result = _evaluate(chinitsu_tiles, 1, is_tsumo=True)
    self.assertIn("Chinitsu", _yaku_names(result))

  def test_chiitoitsu_scoring(self):
    tiles = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    result = _evaluate(tiles, 6, is_tsumo=True)
    self.assertEqual(result.fu, 25)
    self.assertIn("Chiitoitsu", _yaku_names(result))
    self.assertIn("Chinitsu", _yaku_names(result))  # all man tiles.

  def test_kokushi_yakuman(self):
    tiles = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 33]
    result = _evaluate(tiles, 33, is_tsumo=True)
    self.assertTrue(result.is_yakuman)
    self.assertEqual(result.yakuman_multiplier, 1)

  def test_daisangen_yakuman(self):
    tiles = [31, 31, 31, 32, 32, 32, 33, 33, 33, 1, 1, 5, 6, 7]
    result = _evaluate(tiles, 7, is_tsumo=True)
    self.assertTrue(result.is_yakuman)
    self.assertIn("Daisangen", _yaku_names(result))

  def test_dora_counts_winning_tile(self):
    tiles = [1, 2, 3, 5, 6, 7, 12, 13, 14, 16, 16, 19, 20, 21]
    result = _evaluate(tiles, 21, is_tsumo=True, dora_indicators=[20])
    self.assertEqual(dict(result.yaku)["Dora"], 1)
    self.assertEqual(result.han, 4)  # Pinfu+Tsumo+Tanyao(3) + Dora(1).

  def test_multiple_indicators_same_dora_stack(self):
    tiles = [1, 2, 3, 5, 6, 7, 12, 13, 14, 16, 16, 19, 20, 21]
    result = _evaluate(
        tiles, 21, is_tsumo=True, dora_indicators=[20, 20])
    self.assertEqual(dict(result.yaku)["Dora"], 2)

  def test_open_hand_downgrades_han_and_breaks_menzen_yaku(self):
    open_melds = [u.Meld("triplet", 31, False)]  # Called pon of white dragon.
    tiles = [1, 2, 3, 5, 6, 7, 12, 13, 14, 16, 16]  # 11-tile concealed part.
    result = _evaluate(tiles, 14, is_tsumo=False, open_melds=open_melds)
    self.assertIsNotNone(result)
    self.assertNotIn("Pinfu", _yaku_names(result))
    self.assertIn("Yakuhai (White)", _yaku_names(result))


class PaymentTest(absltest.TestCase):

  def test_base_points_table(self):
    self.assertEqual(u.base_points(1, 30), 30 * 8)
    self.assertEqual(u.base_points(4, 30), min(30 * 2**6, 2000))
    self.assertEqual(u.base_points(6, 30), 3000)
    self.assertEqual(u.base_points(8, 30), 4000)
    self.assertEqual(u.base_points(11, 30), 6000)

  def test_non_dealer_ron_payment(self):
    result = u.ScoreResult(yaku=[("Riichi", 1)], han=3, fu=30,
                            is_yakuman=False)
    payments = u.compute_payments(result, is_dealer=False, is_tsumo=False)
    base = 30 * 2**5  # 960
    self.assertEqual(payments["loser"], -(-(base * 4) // 100) * 100)

  def test_dealer_tsumo_payment_is_symmetric(self):
    result = u.ScoreResult(yaku=[("Riichi", 1)], han=4, fu=30,
                            is_yakuman=False)
    payments = u.compute_payments(result, is_dealer=True, is_tsumo=True)
    self.assertIn("each", payments)

  def test_yakuman_payment(self):
    result = u.ScoreResult(yaku=[("Kokushi Musou", 1)], han=13, fu=0,
                            is_yakuman=True, yakuman_multiplier=1)
    payments = u.compute_payments(result, is_dealer=False, is_tsumo=False)
    self.assertEqual(payments["loser"], 32000)

  def test_honba_adds_to_ron_payment(self):
    result = u.ScoreResult(yaku=[("Yakuhai", 1)], han=1, fu=30,
                            is_yakuman=False)
    no_honba = u.compute_payments(result, False, False, honba=0)
    with_honba = u.compute_payments(result, False, False, honba=1)
    self.assertEqual(with_honba["loser"] - no_honba["loser"], 300)


if __name__ == "__main__":
  absltest.main()
