"""
American exercise: binomial pricing and implied-volatility inversion.

Why this module exists
----------------------
Every listed US equity option is American, and the rest of this project prices
them with Black-76, which assumes European exercise. For out-of-the-money
options on a low-dividend index the error is small — but it is not zero, and it
is systematically **one-signed**: an American option is never worth less than
its European twin, so pricing it as European and inverting gives an implied
volatility that is too *high*. The bias grows with the early-exercise value:
deep in-the-money puts, high rates, and dividends before expiry.

Method
------
A Cox-Ross-Rubinstein binomial tree, **vectorised across contracts** rather than
across tree nodes. That inversion matters: a chain has ~1,500 quotes and each
implied-vol solve needs tens of tree builds, so a per-contract Python loop is
minutes and a per-contract-vectorised tree is seconds. Every contract gets its
own S, K, T, sigma and exercise style, and the whole chain marches down the tree
together in numpy.

The tree is used rather than an analytic approximation (Barone-Adesi-Whaley,
Bjerksund-Stensland) because it is exact in the limit, easy to verify against
the European closed form, and fast enough once vectorised.

On accuracy, measured rather than asserted — run `convergence_check()`. CRR
converges at O(1/steps): the error halves each time the step count doubles. At
`steps=256` a one-year at-the-money option on a $100 underlying prices to about
**1 cent** of the closed form, so on a $500 underlying expect a few cents. That
is well inside a typical bid/ask, but it is *not* negligible against a
penny-wide SPY market — which is why the default inversion path stays European
and American pricing is opt-in.

Dividends
---------
Handled as a continuous yield `q`, which the pipeline takes from the
**put-call-parity forward** it already extracts: q = r − ln(F/S)/T. That keeps
the American pricer consistent with the same forward the European path uses, so
any difference between the two is early exercise and not a different dividend
assumption. A discrete-dividend tree would be more accurate for a single name
with one large known dividend; the continuous yield is the right call for an
index and for a chain where the dividend is inferred rather than known.
"""
from __future__ import annotations

import numpy as np

from .black_scholes import bs_price_spot

_EPS = 1e-12


def binomial_price(
    S, K, T, sigma, r=0.0, q=0.0, is_call=True, steps: int = 256,
    american: bool = True,
) -> np.ndarray:
    """
    Cox-Ross-Rubinstein price, vectorised across contracts.

    All of S, K, T, sigma, r, q, is_call broadcast against each other. Returns
    an array shaped like the broadcast inputs.

    `american=False` prices the same tree with European exercise, which is the
    control that makes the early-exercise premium measurable rather than
    assumed.
    """
    S, K, T, sigma, r_a, q_a, call = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, sigma, r, q)),
        np.asarray(is_call, dtype=bool),
    )
    shape = S.shape
    flat = [np.ravel(a).astype(float, copy=True) for a in (S, K, T, sigma, r_a, q_a)]
    S_f, K_f, T_f, sig_f, r_f, q_f = flat
    call_f = np.ravel(call).copy()
    n = S_f.size

    out = np.full(n, np.nan)
    ok = (np.isfinite(S_f) & np.isfinite(K_f) & (S_f > 0) & (K_f > 0)
          & np.isfinite(sig_f) & (sig_f > 0) & np.isfinite(T_f) & (T_f > 0))

    # Degenerate contracts collapse to intrinsic — no tree needed, and building
    # one with dt = 0 would divide by zero.
    dead = ~ok & np.isfinite(S_f) & np.isfinite(K_f)
    if dead.any():
        out[dead] = np.where(call_f[dead], np.maximum(S_f[dead] - K_f[dead], 0.0),
                             np.maximum(K_f[dead] - S_f[dead], 0.0))
    if not ok.any():
        return out.reshape(shape) if shape else float(out[0])

    s, k, t = S_f[ok], K_f[ok], T_f[ok]
    sg, rr, qq, cc = sig_f[ok], r_f[ok], q_f[ok], call_f[ok]
    m = s.size

    dt = t / steps
    u = np.exp(sg * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-rr * dt)
    # Risk-neutral probability. Clipped into [0,1]: with a very short maturity
    # and a large carry, the CRR probability can leave the unit interval, which
    # would produce a negative "price" rather than an obvious failure.
    p = np.clip((np.exp((rr - qq) * dt) - d) / np.maximum(u - d, _EPS), 0.0, 1.0)

    # Terminal layer: S * u^j * d^(steps-j) for j = 0..steps.
    j = np.arange(steps + 1)
    prices = s[:, None] * u[:, None] ** j[None, :] * d[:, None] ** (steps - j)[None, :]
    sign = np.where(cc, 1.0, -1.0)[:, None]
    values = np.maximum(sign * (prices - k[:, None]), 0.0)

    # March backwards. At each step the node's spot is the previous layer's
    # divided by d, which avoids recomputing powers.
    for step in range(steps - 1, -1, -1):
        prices = prices[:, : step + 1] / d[:, None]
        values = disc[:, None] * (p[:, None] * values[:, 1: step + 2]
                                  + (1.0 - p[:, None]) * values[:, : step + 1])
        if american:
            intrinsic = np.maximum(sign * (prices - k[:, None]), 0.0)
            values = np.maximum(values, intrinsic)

    out[ok] = values[:, 0]
    return out.reshape(shape) if shape else float(out[0])


def early_exercise_premium(S, K, T, sigma, r=0.0, q=0.0, is_call=True,
                           steps: int = 256) -> np.ndarray:
    """
    American price minus European price, on the *same tree*.

    Differencing two tree prices rather than tree-minus-Black-Scholes cancels
    the discretisation error almost exactly, so what is left is the early
    exercise value and not the lattice's own bias.
    """
    american = binomial_price(S, K, T, sigma, r, q, is_call, steps, american=True)
    european = binomial_price(S, K, T, sigma, r, q, is_call, steps, american=False)
    return american - european


def implied_vol_american(
    price, S, K, T, r=0.0, q=0.0, is_call=True, steps: int = 128,
    lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-6, max_iter: int = 60,
) -> np.ndarray:
    """
    Implied volatility under American exercise, by vectorised bisection.

    Bisection rather than Newton: a binomial price has no closed-form vega, and
    a finite-difference vega would double the tree cost per iteration for a
    derivative that is only used to take a step. Price is monotone in sigma, so
    bisection is unconditionally convergent — 60 iterations narrows the bracket
    to about 1e-16 of its width, and the loop exits far earlier on tolerance.

    Returns NaN where the price is outside the no-arbitrage bounds, matching
    the European inverter's contract of never returning a guessed root.
    """
    price, S, K, T, r_a, q_a, call = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (price, S, K, T, r, q)),
        np.asarray(is_call, dtype=bool),
    )
    shape = price.shape
    p_f = np.ravel(price).astype(float)
    S_f, K_f, T_f = (np.ravel(x).astype(float) for x in (S, K, T))
    r_f, q_f = (np.ravel(x).astype(float) for x in (r_a, q_a))
    call_f = np.ravel(call)

    # American bounds: never below intrinsic (exercisable now), never above the
    # underlying (call) or the strike (put).
    intrinsic = np.where(call_f, np.maximum(S_f - K_f, 0.0),
                         np.maximum(K_f - S_f, 0.0))
    upper = np.where(call_f, S_f, K_f)
    valid = (np.isfinite(p_f) & (T_f > 0) & (S_f > 0) & (K_f > 0)
             & (p_f > intrinsic + 1e-10) & (p_f < upper - 1e-12))

    out = np.full(p_f.shape, np.nan)
    if not valid.any():
        return out.reshape(shape) if shape else float(out[0])

    idx = np.flatnonzero(valid)
    lo_v = np.full(idx.size, lo)
    hi_v = np.full(idx.size, hi)
    args = (S_f[idx], K_f[idx], T_f[idx], r_f[idx], q_f[idx], call_f[idx])

    for _ in range(max_iter):
        mid = 0.5 * (lo_v + hi_v)
        modelled = binomial_price(args[0], args[1], args[2], mid, args[3],
                                  args[4], args[5], steps=steps, american=True)
        too_high = modelled > p_f[idx]
        hi_v = np.where(too_high, mid, hi_v)
        lo_v = np.where(too_high, lo_v, mid)
        if np.all(hi_v - lo_v < tol):
            break

    sigma = 0.5 * (lo_v + hi_v)
    # Reject roots pinned to the bracket: those did not converge, they ran out
    # of room, and reporting them as a volatility would be a lie.
    converged = (sigma > lo * 1.5) & (sigma < hi * 0.99)
    out[idx] = np.where(converged, sigma, np.nan)
    return out.reshape(shape) if shape else float(out[0])


def convergence_check(S: float = 100.0, K: float = 100.0, T: float = 1.0,
                      sigma: float = 0.25, r: float = 0.05, q: float = 0.02,
                      is_call: bool = True,
                      step_counts=(16, 32, 64, 128, 256, 512, 1024)):
    """
    European tree price against the closed form as the step count rises.

    Run this rather than trusting a step count: it reports the actual error, in
    price units, for the parameters you care about.
    """
    import pandas as pd

    exact = float(bs_price_spot(S, K, T, sigma, r=r, q=q, is_call=is_call))
    rows = []
    for steps in step_counts:
        tree = float(binomial_price(S, K, T, sigma, r, q, is_call,
                                    steps=steps, american=False))
        rows.append({"steps": steps, "tree": tree, "closed_form": exact,
                     "abs_error": abs(tree - exact)})
    return pd.DataFrame(rows)
