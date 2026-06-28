"""Covariance-based position sizing for cross-sectional long books."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252.0
WEIGHTING_METHODS = ('equal', 'inverse_vol', 'risk_parity', 'min_variance')


def _validate_square_frame(cov: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(cov, pd.DataFrame):
        raise TypeError('cov must be a pandas DataFrame.')
    if cov.empty:
        raise ValueError('cov must not be empty.')
    if cov.shape[0] != cov.shape[1]:
        raise ValueError('cov must be square.')
    if list(cov.index) != list(cov.columns):
        raise ValueError('cov index and columns must match in the same order.')
    return cov.astype(float)


def _normalize_positive(raw: pd.Series) -> pd.Series:
    raw = raw.astype(float)
    if raw.isna().any():
        raise ValueError('weights contain NaN values.')
    if (raw < 0).any():
        raise ValueError('weights must be non-negative.')
    total = float(raw.sum())
    if total <= 0:
        raise ValueError('weights sum must be positive.')
    return raw / total


def estimate_covariance(returns: pd.DataFrame, shrink: float = 0.2) -> pd.DataFrame:
    """Estimate a shrinkage covariance matrix from daily returns.

    This is the manual analog of Ledoit-Wolf shrinkage: sample covariances are
    noisy, so pulling noisy off-diagonal covariances toward zero keeps the
    matrix well-conditioned and invertible. sklearn's LedoitWolf can replace
    this for the optimal intensity, but the fixed-shrink version needs no extra
    dependency.

    The diagonal target keeps each asset's sample variance and zeros all
    off-diagonal covariances. ``shrink`` must be in ``[0, 1]``.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError('returns must be a pandas DataFrame.')
    if returns.empty or returns.shape[1] == 0:
        raise ValueError('returns must contain at least one asset.')
    if not 0.0 <= shrink <= 1.0:
        raise ValueError('shrink must be in [0, 1].')

    sample = returns.astype(float).cov()
    if sample.isna().any().any():
        raise ValueError('covariance contains NaN values; check return history.')
    diagonal = np.diag(np.diag(sample.to_numpy(dtype=float)))
    target = pd.DataFrame(diagonal, index=sample.index, columns=sample.columns)
    return (1.0 - shrink) * sample + shrink * target


def inverse_vol_weights(cov: pd.DataFrame) -> pd.Series:
    """Weight assets in proportion to inverse standalone volatility."""
    cov = _validate_square_frame(cov)
    variances = pd.Series(np.diag(cov.to_numpy(dtype=float)), index=cov.index)
    if variances.isna().any() or (variances <= 0).any():
        raise ValueError('covariance diagonal must contain positive variances.')
    raw = 1.0 / np.sqrt(variances)
    return _normalize_positive(raw)


def risk_parity_weights(
    cov: pd.DataFrame,
    iters: int = 5000,
    tol: float = 1e-12,
) -> pd.Series:
    """Solve equal-risk-contribution weights with fixed-point iteration.

    The iteration uses ``w_i proportional to 1 / (Sigma w)_i``. This matters:
    using ``w_i / (Sigma w)_i`` instead would converge toward minimum variance,
    not equal risk contribution.
    """
    cov = _validate_square_frame(cov)
    if iters <= 0:
        raise ValueError('iters must be positive.')
    if tol <= 0:
        raise ValueError('tol must be positive.')

    sigma = cov.to_numpy(dtype=float)
    w = inverse_vol_weights(cov).to_numpy(dtype=float)

    for _ in range(iters):
        marginal = sigma @ w
        if np.any(marginal <= 0) or np.any(~np.isfinite(marginal)):
            raise ValueError('risk parity iteration requires positive marginal risks.')
        next_w = 1.0 / marginal
        next_w = next_w / next_w.sum()
        if float(np.max(np.abs(next_w - w))) < tol:
            w = next_w
            break
        w = next_w

    marginal = sigma @ w
    risk_contrib = w * marginal
    target = float(risk_contrib.sum()) / len(w)
    assert np.allclose(risk_contrib, target, rtol=1e-6, atol=1e-12)
    return pd.Series(w, index=cov.index, name='risk_parity')


def min_variance_weights(cov: pd.DataFrame, long_only: bool = True) -> pd.Series:
    """Compute minimum-variance weights with a linear solve.

    The unconstrained solution is ``w proportional to solve(Sigma, ones)`` and
    avoids forming an explicit inverse for numerical stability. When
    ``long_only`` is true, negative weights are clipped to zero and the
    remaining weights are renormalized. That clipping is only an approximation:
    exact long-only minimum variance needs a quadratic-programming solver.
    """
    cov = _validate_square_frame(cov)
    sigma = cov.to_numpy(dtype=float)
    ones = np.ones(len(cov), dtype=float)
    raw = pd.Series(np.linalg.solve(sigma, ones), index=cov.index)
    total = float(raw.sum())
    if abs(total) <= 1e-15:
        raise ValueError('minimum-variance weights have near-zero total exposure.')
    weights = raw / total
    if long_only:
        weights = weights.clip(lower=0.0)
        weights = _normalize_positive(weights)
    return weights.rename('min_variance')


def risk_contributions(
    weights: pd.Series | dict[str, float],
    cov: pd.DataFrame,
    annualize: bool = True,
) -> pd.DataFrame:
    """Return marginal and total variance contributions by ticker.

    If ``annualize`` is true, covariance inputs are scaled by 252 before
    computing marginal contribution, risk contribution, total variance, and
    portfolio volatility. This assumes the input covariance matrix was estimated
    from daily returns.
    """
    cov = _validate_square_frame(cov)
    w = pd.Series(weights, dtype=float).reindex(cov.index)
    if w.isna().any():
        raise ValueError('weights must cover every covariance ticker.')

    scale = TRADING_DAYS if annualize else 1.0
    sigma = cov.to_numpy(dtype=float) * scale
    wv = w.to_numpy(dtype=float)
    marginal = sigma @ wv
    risk_contrib = wv * marginal
    portfolio_variance = float(wv @ sigma @ wv)
    assert np.isclose(float(risk_contrib.sum()), portfolio_variance, rtol=1e-10, atol=1e-14)

    pct = risk_contrib / portfolio_variance if portfolio_variance != 0 else np.nan
    out = pd.DataFrame({
        'weight': wv,
        'marginal_contribution': marginal,
        'risk_contribution': risk_contrib,
        'pct_of_total': pct,
    }, index=cov.index)
    out.attrs['portfolio_variance'] = portfolio_variance
    out.attrs['portfolio_volatility'] = float(np.sqrt(portfolio_variance))
    out.attrs['annualize'] = annualize
    return out


def size_long_leg(
    returns: pd.DataFrame,
    method: str = 'inverse_vol',
    shrink: float = 0.2,
) -> pd.Series:
    """Dispatch cross-sectional long-leg position sizing."""
    cov = estimate_covariance(returns, shrink=shrink)
    key = method.lower().replace('-', '_')
    if key == 'equal':
        return pd.Series(1.0 / len(cov), index=cov.index, name='equal')
    if key == 'inverse_vol':
        return inverse_vol_weights(cov).rename('inverse_vol')
    if key == 'risk_parity':
        return risk_parity_weights(cov).rename('risk_parity')
    if key == 'min_variance':
        return min_variance_weights(cov).rename('min_variance')
    raise ValueError("method must be one of: 'equal', 'inverse_vol', 'risk_parity', 'min_variance'.")


def _portfolio_vol(weights: pd.Series, cov: pd.DataFrame, annualize: bool) -> float:
    rc = risk_contributions(weights, cov, annualize=annualize)
    return float(rc.attrs['portfolio_volatility'])


def report_sizing(
    returns: pd.DataFrame,
    annualize: bool = True,
    label: str = 'Long-Leg',
) -> dict:
    """Print weights, volatility, and equal-weight risk concentration."""
    cov = estimate_covariance(returns)
    methods = list(WEIGHTING_METHODS)
    weights: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}
    for method in methods:
        try:
            weights[method] = size_long_leg(returns, method=method)
        except (AssertionError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
            errors[method] = str(exc)

    weight_table = pd.DataFrame(index=cov.index)
    for method in methods:
        weight_table[method] = weights.get(method, pd.Series(np.nan, index=cov.index))
    vols = pd.Series(index=methods, dtype=float, name='portfolio_volatility')
    for method, w in weights.items():
        vols[method] = _portfolio_vol(w, cov, annualize=annualize)
    equal_rc = risk_contributions(weights['equal'], cov, annualize=annualize)

    vol_label = 'annualized volatility' if annualize else 'daily volatility'
    print(f'\n=== {label} Sizing Weights ===')
    print(weight_table.to_string(float_format=lambda x: f'{x:0.4f}', na_rep='n/a'))
    print(f'\n=== Resulting {vol_label.title()} ===')
    print(vols.to_frame().to_string(float_format=lambda x: f'{x:0.2%}', na_rep='n/a'))
    if errors:
        print('\nUnavailable sizing methods:')
        for method in methods:
            if method in errors:
                print(f'  {method}: {errors[method]}')
    print('\n=== Equal-Weight Risk Contributions ===')
    print(equal_rc.to_string(formatters={
        'weight': lambda x: f'{x:0.4f}',
        'marginal_contribution': lambda x: f'{x:0.6f}',
        'risk_contribution': lambda x: f'{x:0.6f}',
        'pct_of_total': lambda x: f'{x:0.2%}',
    }))
    return {
        'weights': weight_table,
        'portfolio_volatility': vols,
        'equal_risk_contributions': equal_rc,
        'errors': errors,
    }


# Integration note:
# In the cross-sectional long leg, replace the 1/k equal weight by calling
# size_long_leg(trailing_returns_of_selected_longs, method=...). Expose it as a
# weighting parameter defaulting to 'equal' so existing behavior is unchanged
# and backward-compatible.
