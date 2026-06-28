"""Unit tests for covariance-based risk sizing."""

from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from quant.risk_model import (
    estimate_covariance,
    report_sizing,
    risk_contributions,
    risk_parity_weights,
    size_long_leg,
)


def synthetic_returns(n_days: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    corr = np.array([
        [1.00, 0.55, 0.35, 0.25, 0.20],
        [0.55, 1.00, 0.45, 0.30, 0.25],
        [0.35, 0.45, 1.00, 0.40, 0.30],
        [0.25, 0.30, 0.40, 1.00, 0.50],
        [0.20, 0.25, 0.30, 0.50, 1.00],
    ])
    vols = np.array([0.10, 0.16, 0.22, 0.30, 0.42]) / np.sqrt(252.0)
    cov = corr * np.outer(vols, vols)
    data = rng.multivariate_normal(np.zeros(5), cov, size=n_days)
    return pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])


class TestRiskModel(unittest.TestCase):
    def test_sizing_methods_sum_to_one(self):
        returns = synthetic_returns()
        for method in ['equal', 'inverse_vol', 'risk_parity', 'min_variance']:
            weights = size_long_leg(returns, method=method)
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
            self.assertTrue((weights >= 0).all())

    def test_risk_parity_equalizes_risk_contributions(self):
        returns = synthetic_returns()
        cov = estimate_covariance(returns)
        weights = risk_parity_weights(cov)
        rc = risk_contributions(weights, cov, annualize=False)
        expected = np.full(len(rc), 1.0 / len(rc))
        np.testing.assert_allclose(rc['pct_of_total'].to_numpy(), expected, rtol=1e-5, atol=1e-8)

    def test_risk_contributions_sum_to_portfolio_variance(self):
        returns = synthetic_returns()
        cov = estimate_covariance(returns)
        weights = size_long_leg(returns, method='inverse_vol')
        rc = risk_contributions(weights, cov, annualize=True)
        self.assertAlmostEqual(
            float(rc['risk_contribution'].sum()),
            float(rc.attrs['portfolio_variance']),
            places=14,
        )

    def test_report_sizing_prints_pct_of_total_as_percent(self):
        returns = synthetic_returns()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_sizing(returns)
        out = buf.getvalue()
        self.assertIn('=== Long-Leg Sizing Weights ===', out)
        self.assertIn('=== Resulting Annualized Volatility ===', out)
        self.assertIn('=== Equal-Weight Risk Contributions ===', out)
        self.assertIn('pct_of_total', out)
        self.assertIn('%', out.split('=== Equal-Weight Risk Contributions ===', 1)[1])


if __name__ == '__main__':
    unittest.main()
