"""Phase 6 tests: peer-group-neutral construction. Offline, synthetic panels."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant.models.cross_sectional import (
    CrossSectionalModel,
    backtest_xs,
    build_weights,
    compute_scores,
)
from quant.universes import (
    DEFAULT_PEER_GROUP,
    build_group_map,
    groups_to_members,
    peer_group_of,
)
from validate_cross_sectional import group_neutral_kwargs, make_xs_strategy


# Eight names spanning four peer groups (two per group).
_TICKERS = ['MU', 'SNDK', 'AMAT', 'LRCX', 'TSM', 'INTC', 'SNPS', 'CDNS']
_GROUPS = {
    'MU': 'memory_storage', 'SNDK': 'memory_storage',
    'AMAT': 'equipment', 'LRCX': 'equipment',
    'TSM': 'foundry', 'INTC': 'foundry',
    'SNPS': 'eda_ip', 'CDNS': 'eda_ip',
}


def _panel(n_days: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.009, size=n_days)
    cols, t = {}, np.arange(n_days)
    for i, tk in enumerate(_TICKERS):
        phase = 2.0 * np.pi * i / len(_TICKERS)
        rotate = 0.0008 * np.sin(2.0 * np.pi * t / 70.0 + phase)
        noise = rng.normal(0.0, 0.012, size=n_days)
        r = market * (0.9 + 0.03 * i) + rotate + noise
        cols[tk] = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=pd.bdate_range('2022-01-01', periods=n_days))


class TestPeerGroupMap(unittest.TestCase):
    def test_known_and_unknown_tickers(self):
        self.assertEqual(peer_group_of('MU'), 'memory_storage')
        self.assertEqual(peer_group_of('amat'), 'equipment')
        self.assertEqual(peer_group_of('ZZZZ'), DEFAULT_PEER_GROUP)

    def test_thin_groups_merge_into_fallback(self):
        # TXN is the only analog_power name here -> should merge into fallback.
        tickers = ['MU', 'SNDK', 'TXN', 'AMAT', 'LRCX']
        gm = build_group_map(tickers, min_group_names=2)
        self.assertEqual(gm['TXN'], DEFAULT_PEER_GROUP)
        self.assertEqual(gm['MU'], 'memory_storage')
        self.assertEqual(gm['AMAT'], 'equipment')

    def test_min_group_names_one_keeps_singletons(self):
        gm = build_group_map(['MU', 'TXN'], min_group_names=1)
        self.assertEqual(gm['TXN'], 'analog_power')

    def test_invalid_min_group_names(self):
        with self.assertRaises(ValueError):
            build_group_map(['MU'], min_group_names=0)


class TestGroupNeutralWeights(unittest.TestCase):
    def test_weights_sum_to_one(self):
        panel = _panel()
        scores = compute_scores(panel, mode='momentum', lookback=63, skip=0)
        w = build_weights(
            panel, scores, top_frac=0.5, rebalance=5, market_neutral=False,
            group_neutral=True, group_map=_GROUPS,
        )
        sums = w.sum(axis=1)
        invested = sums[sums > 1e-9]
        np.testing.assert_allclose(invested.to_numpy(), 1.0, atol=1e-9)
        self.assertTrue((w.to_numpy() >= -1e-12).all())

    def test_equal_capital_across_groups(self):
        """With one name picked per group, each active group gets 1/G capital."""
        panel = _panel()
        scores = compute_scores(panel, mode='momentum', lookback=63, skip=0)
        w = build_weights(
            panel, scores, top_frac=0.25, rebalance=5, market_neutral=False,
            group_neutral=True, group_map=_GROUPS, group_top_frac=0.25,
        )
        last = w.iloc[-1]
        held = last[last > 1e-9]
        # 4 groups, 1 name each at group_top_frac=0.25 (round(0.25*2)=0 -> max 1).
        self.assertEqual(len(held), 4)
        np.testing.assert_allclose(held.to_numpy(), 0.25, atol=1e-9)
        # Each held name is from a distinct group.
        groups = {_GROUPS[t] for t in held.index}
        self.assertEqual(len(groups), 4)

    def test_not_concentrated_in_one_group(self):
        """Group-neutral spreads across groups; plain top-k can pile into one."""
        panel = _panel()
        scores = compute_scores(panel, mode='momentum', lookback=63, skip=0)
        gn = build_weights(
            panel, scores, top_frac=0.25, rebalance=5, market_neutral=False,
            group_neutral=True, group_map=_GROUPS,
        )
        held = gn.iloc[-1][gn.iloc[-1] > 1e-9]
        per_group = {}
        for t, wt in held.items():
            per_group[_GROUPS[t]] = per_group.get(_GROUPS[t], 0.0) + wt
        # No single group exceeds half the book.
        self.assertLessEqual(max(per_group.values()), 0.5 + 1e-9)

    def test_market_neutral_with_group_neutral_raises(self):
        panel = _panel()
        scores = compute_scores(panel, mode='momentum', lookback=63, skip=0)
        with self.assertRaises(ValueError):
            build_weights(panel, scores, market_neutral=True,
                          group_neutral=True, group_map=_GROUPS)

    def test_group_neutral_requires_map(self):
        panel = _panel()
        scores = compute_scores(panel, mode='momentum', lookback=63, skip=0)
        with self.assertRaises(ValueError):
            build_weights(panel, scores, market_neutral=False,
                          group_neutral=True, group_map=None)


class TestGroupNeutralPlumbing(unittest.TestCase):
    def test_default_off_is_backward_compatible(self):
        panel = _panel()
        params = dict(mode='momentum', lookback=63, skip=0, top_frac=0.25,
                      rebalance=5, market_neutral=False)
        base = backtest_xs(panel, **params)[0]
        explicit_off = backtest_xs(panel, group_neutral=False, **params)[0]
        pd.testing.assert_frame_equal(base, explicit_off)

    def test_backtest_xs_threads_group_neutral(self):
        panel = _panel()
        df, _ = backtest_xs(
            panel, mode='momentum', lookback=63, skip=0, top_frac=0.5,
            rebalance=5, market_neutral=False,
            group_neutral=True, group_map=_GROUPS,
        )
        self.assertIn('strat_net', df.columns)
        self.assertTrue(np.isfinite(df['strat_net']).all())

    def test_current_weights_group_neutral(self):
        panel = _panel()
        w = CrossSectionalModel().current_weights(
            panel, mode='momentum', lookback=63, skip=0, top_frac=0.25,
            market_neutral=False, group_neutral=True, group_map=_GROUPS,
        )
        self.assertGreater(len(w), 0)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)
        self.assertTrue((w > 0).all())

    def test_kwargs_helper(self):
        self.assertEqual(group_neutral_kwargs(False), {})
        out = group_neutral_kwargs(True, group_map=_GROUPS, group_top_frac=0.3)
        self.assertTrue(out['group_neutral'])
        self.assertEqual(out['group_map'], _GROUPS)
        self.assertAlmostEqual(out['group_top_frac'], 0.3)
        with self.assertRaises(ValueError):
            group_neutral_kwargs(True, group_map=None)

    def test_no_lookahead_future_perturbation(self):
        """Group-neutral weights at date t cannot move when a strictly-later
        price is perturbed."""
        panel = _panel()
        scores = compute_scores(panel, mode='momentum', lookback=63, skip=0)
        w_base = build_weights(
            panel, scores, top_frac=0.5, rebalance=5, market_neutral=False,
            group_neutral=True, group_map=_GROUPS,
        )
        bumped = panel.copy()
        bumped.iloc[-10:] *= 1.25
        scores_b = compute_scores(bumped, mode='momentum', lookback=63, skip=0)
        w_bumped = build_weights(
            bumped, scores_b, top_frac=0.5, rebalance=5, market_neutral=False,
            group_neutral=True, group_map=_GROUPS,
        )
        cutoff = panel.index[-11]
        pd.testing.assert_frame_equal(
            w_base.loc[:cutoff], w_bumped.loc[:cutoff],
        )


if __name__ == '__main__':
    unittest.main()
