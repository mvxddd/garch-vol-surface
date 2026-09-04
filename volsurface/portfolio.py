"""
Position-level risk against a calibrated surface.

Give it a list of option and underlying positions and it prices them on the
surface, aggregates the Greeks, breaks vega down by tenor and by strike, and
runs a spot/vol stress grid. This is the layer that turns a research surface
into something a desk would actually look at in the morning.

Two modelling choices that matter more than anything else here
--------------------------------------------------------------
**How the smile moves when spot moves.** Shocking spot by −5% is meaningless
until you say what happens to the smile. Two conventions:

* *sticky moneyness* (a.k.a. sticky delta, the default) — the smile travels
  with the forward, so an option keeps its implied vol for a given `log(K/F)`.
  This is the right default for index options over short horizons and the
  convention most desks quote in.
* *sticky strike* — each strike keeps its vol as spot moves. Closer to how
  single-name equity options behave between re-marks.

The two give materially different gamma/vanna P&L on a large move, so the
convention is an explicit parameter and is printed in every report.

**Vega is reported per 1 volatility point** (0.01), not per 1.00 of vol,
because that is the unit a trader thinks in: "I am short 40k vega" means 40,000
currency units per vol point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from .config import CALENDAR_DAYS
from .models.black_scholes import black76_price, greeks
from .utils import get_logger

LOG = get_logger("volsurface.portfolio")

StickyMode = Literal["moneyness", "strike"]


@dataclass
class Position:
    """
    One line of a book.

    quantity is in contracts and may be negative (short). `multiplier` is the
    contract size — 100 for US listed equity options, 1 for the underlying.
    """

    instrument: Literal["call", "put", "underlying"]
    quantity: float
    strike: float | None = None
    expiry: pd.Timestamp | None = None
    multiplier: float = 100.0
    label: str = ""

    def __post_init__(self) -> None:
        self.instrument = str(self.instrument).lower().strip()  # type: ignore[assignment]
        if self.instrument in {"c", "call"}:
            self.instrument = "call"                            # type: ignore[assignment]
        elif self.instrument in {"p", "put"}:
            self.instrument = "put"                             # type: ignore[assignment]
        elif self.instrument in {"u", "underlying", "stock", "spot"}:
            self.instrument = "underlying"                      # type: ignore[assignment]
            self.multiplier = self.multiplier if self.multiplier != 100.0 else 1.0
        else:
            raise ValueError(f"Unknown instrument {self.instrument!r}")

        if self.instrument != "underlying":
            if self.strike is None or not np.isfinite(self.strike) or self.strike <= 0:
                raise ValueError(f"Option position needs a positive strike: {self}")
            if self.expiry is None:
                raise ValueError(f"Option position needs an expiry: {self}")
            self.expiry = pd.Timestamp(self.expiry).normalize()
        if not np.isfinite(self.quantity):
            raise ValueError(f"Position quantity must be finite: {self}")


@dataclass
class Portfolio:
    """A named collection of positions."""

    positions: list[Position] = field(default_factory=list)
    name: str = "book"

    def __len__(self) -> int:
        return len(self.positions)

    @classmethod
    def from_records(cls, records: Iterable[dict], name: str = "book") -> "Portfolio":
        return cls([Position(**r) for r in records], name=name)

    @classmethod
    def from_csv(cls, path: str | Path, name: str | None = None) -> "Portfolio":
        """
        Load from a CSV with columns:
        instrument, quantity, strike, expiry[, multiplier, label]

        Unknown columns are ignored, so a broker export can usually be fed in
        after renaming a few headers.
        """
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"instrument", "quantity"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}")

        keep = ["instrument", "quantity", "strike", "expiry", "multiplier", "label"]
        records = []
        for row in df[[c for c in keep if c in df.columns]].to_dict("records"):
            records.append({k: v for k, v in row.items()
                            if not (isinstance(v, float) and np.isnan(v))})
        LOG.info("Loaded %d positions from %s", len(records), path)
        return cls.from_records(records, name=name or Path(path).stem)


# --------------------------------------------------------------------------- #
# Pricing and Greeks
# --------------------------------------------------------------------------- #
def price_portfolio(portfolio: Portfolio, surface, spot: float | None = None,
                    vol_shift: float = 0.0, spot_shift: float = 0.0,
                    sticky: StickyMode = "moneyness") -> pd.DataFrame:
    """
    Price every position on the surface and return a per-position risk frame.

    Parameters
    ----------
    vol_shift  : parallel shift of the whole surface, in vol **points**
                 (0.01 = +1 vol point).
    spot_shift : relative move of the underlying (-0.05 = −5%).
    sticky     : how the smile follows spot — see the module docstring.

    Greeks are per position (already multiplied by quantity and multiplier):
    `delta` in shares, `vega` per vol point, `theta` per calendar day.
    """
    base_spot = float(spot if spot is not None else surface.spot)
    new_spot = base_spot * (1.0 + spot_shift)
    asof = pd.Timestamp(surface.asof).normalize()
    r = float(getattr(surface, "r", 0.0))

    rows = []
    for pos in portfolio.positions:
        if pos.instrument == "underlying":
            value = new_spot * pos.quantity * pos.multiplier
            rows.append({
                "label": pos.label or "underlying", "instrument": "underlying",
                "quantity": pos.quantity, "strike": np.nan, "expiry": pd.NaT,
                "days": np.nan, "T": np.nan, "forward": np.nan, "k": np.nan,
                "iv": np.nan, "price": new_spot, "value": value,
                "delta": pos.quantity * pos.multiplier, "gamma": 0.0, "vega": 0.0,
                "theta": 0.0, "notional": abs(value),
            })
            continue

        T = max((pos.expiry - asof).days, 0) / CALENDAR_DAYS
        if T <= 0:
            LOG.warning("Position %s expires on or before the surface date — "
                        "valued at intrinsic.", pos.label or pos.strike)

        # Forward under the shocked spot: the surface's own forward curve
        # carries the carry, so scale it by the spot move.
        base_forward = float(surface.forward(max(T, 1e-9)))
        forward = base_forward * (1.0 + spot_shift)

        if sticky == "moneyness":
            # Smile travels with the forward: same log-moneyness, same vol.
            k = np.log(pos.strike / forward)
        else:
            # Smile pinned to strikes: read the vol at the *unshocked* moneyness.
            k = np.log(pos.strike / base_forward)

        iv = float(surface.iv(k, max(T, 1e-9))) + vol_shift
        iv = max(iv, 1e-4)
        is_call = pos.instrument == "call"

        if T <= 0:
            price = float(max(forward - pos.strike, 0.0) if is_call
                          else max(pos.strike - forward, 0.0))
            g = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        else:
            price = float(black76_price(forward, pos.strike, T, iv, r=r,
                                        is_call=is_call))
            g = greeks(forward, pos.strike, T, iv, r=r, q=0.0, is_call=is_call,
                       S=new_spot)

        scale = pos.quantity * pos.multiplier
        rows.append({
            "label": pos.label or f"{pos.instrument} {pos.strike:g} "
                                  f"{pd.Timestamp(pos.expiry).date()}",
            "instrument": pos.instrument, "quantity": pos.quantity,
            "strike": pos.strike, "expiry": pos.expiry,
            "days": int(round(T * CALENDAR_DAYS)), "T": T, "forward": forward,
            "k": float(k), "iv": iv, "price": price, "value": price * scale,
            "delta": float(g["delta"]) * scale,
            "gamma": float(g["gamma"]) * scale,
            # Per vol point, and per calendar day: the units traders quote.
            "vega": float(g["vega"]) * scale / 100.0,
            "theta": float(g["theta"]) * scale,
            "notional": abs(pos.strike * scale),
        })

    df = pd.DataFrame(rows)
    df.attrs.update(spot=new_spot, base_spot=base_spot, vol_shift=vol_shift,
                    spot_shift=spot_shift, sticky=sticky)
    return df


def aggregate_risk(priced: pd.DataFrame) -> dict[str, float]:
    """Book-level totals, plus the dollar move implied by a 1% spot shock."""
    if priced.empty:
        return {}
    spot = float(priced.attrs.get("spot", np.nan))
    delta = float(priced["delta"].sum())
    gamma = float(priced["gamma"].sum())
    return {
        "value": float(priced["value"].sum()),
        "delta_shares": delta,
        "delta_cash": delta * spot,
        "gamma_shares_per_pct": gamma * spot * 0.01,
        "vega_per_vol_point": float(priced["vega"].sum()),
        "theta_per_day": float(priced["theta"].sum()),
        "gross_notional": float(priced["notional"].sum()),
        "n_positions": int(len(priced)),
        # Dollar-gamma: what a 1% move does to delta, the number that tells you
        # how fast the hedge goes stale.
        "delta_change_per_1pct": gamma * spot * spot * 0.01,
    }


def vega_ladder(priced: pd.DataFrame, by: Literal["tenor", "strike"] = "tenor",
                n_buckets: int = 6) -> pd.DataFrame:
    """
    Vega broken down by maturity bucket or by moneyness bucket.

    A book can be flat total vega and still be badly exposed — long the front,
    short the back. The ladder is what makes that visible, and it is the
    standard way vol risk is shown on a desk.
    """
    opts = priced[priced["instrument"] != "underlying"].copy()
    if opts.empty:
        return pd.DataFrame(columns=["bucket", "vega", "n"])

    if by == "tenor":
        edges = [0, 30, 60, 90, 180, 365, np.inf]
        labels = ["≤30d", "31-60d", "61-90d", "91-180d", "181-365d", ">365d"]
        opts["bucket"] = pd.cut(opts["days"], bins=edges, labels=labels,
                                right=True, include_lowest=True)
    else:
        edges = [-np.inf, -0.15, -0.07, -0.02, 0.02, 0.07, np.inf]
        labels = ["<-15%", "-15..-7%", "-7..-2%", "ATM ±2%", "+2..+7%", ">+7%"]
        opts["bucket"] = pd.cut(opts["k"], bins=edges, labels=labels)

    out = (opts.groupby("bucket", observed=False)
               .agg(vega=("vega", "sum"), n=("vega", "size"))
               .reset_index())
    out["bucket"] = out["bucket"].astype(str)
    return out


def stress_grid(portfolio: Portfolio, surface,
                spot_shocks: Sequence[float] = (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10),
                vol_shocks: Sequence[float] = (-0.05, -0.02, 0.0, 0.02, 0.05),
                sticky: StickyMode = "moneyness") -> pd.DataFrame:
    """
    P&L across a grid of simultaneous spot and volatility shocks.

    Returns a frame indexed by vol shock (rows) and spot shock (columns), in
    currency, relative to the unshocked book value. This is the single most
    useful risk artefact for an options book: it captures gamma and vanna,
    which a delta/vega summary cannot.
    """
    base = price_portfolio(portfolio, surface, sticky=sticky)
    base_value = float(base["value"].sum())

    data = {}
    for ds in spot_shocks:
        col = []
        for dv in vol_shocks:
            shocked = price_portfolio(portfolio, surface, vol_shift=dv,
                                      spot_shift=ds, sticky=sticky)
            col.append(float(shocked["value"].sum()) - base_value)
        data[ds] = col

    grid = pd.DataFrame(data, index=list(vol_shocks))
    grid.index.name = "vol_shock"
    grid.columns.name = "spot_shock"
    grid.attrs.update(base_value=base_value, sticky=sticky)
    return grid


def risk_report(portfolio: Portfolio, surface, sticky: StickyMode = "moneyness"
                ) -> dict[str, object]:
    """Everything at once: positions, totals, both ladders, and the stress grid."""
    priced = price_portfolio(portfolio, surface, sticky=sticky)
    report = {
        "positions": priced,
        "totals": aggregate_risk(priced),
        "vega_by_tenor": vega_ladder(priced, by="tenor"),
        "vega_by_strike": vega_ladder(priced, by="strike"),
        "stress": stress_grid(portfolio, surface, sticky=sticky),
        "sticky": sticky,
    }
    tot = report["totals"]
    LOG.info("Book %s: value %.0f · delta %.0f sh · vega %.0f per vol pt · "
             "theta %.0f per day (sticky %s)",
             portfolio.name, tot.get("value", 0.0), tot.get("delta_shares", 0.0),
             tot.get("vega_per_vol_point", 0.0), tot.get("theta_per_day", 0.0),
             sticky)
    return report


def example_portfolio(surface, spot: float | None = None) -> Portfolio:
    """
    A small, realistic book for demos and tests: a short front-month strangle
    financed against a long longer-dated put, plus a delta hedge. Exactly the
    shape of position whose risk is invisible without a stress grid.
    """
    spot = float(spot if spot is not None else surface.spot)
    mats = list(surface.maturities)
    near = mats[min(1, len(mats) - 1)]
    far = mats[min(len(mats) - 1, max(0, len(mats) // 2))]
    asof = pd.Timestamp(surface.asof).normalize()

    def expiry_of(T: float) -> pd.Timestamp:
        return asof + pd.Timedelta(days=int(round(T * CALENDAR_DAYS)))

    def round_strike(x: float) -> float:
        return float(np.round(x / 5.0) * 5.0)

    return Portfolio([
        Position("put", -10, round_strike(spot * 0.93), expiry_of(near),
                 label="short front put"),
        Position("call", -10, round_strike(spot * 1.05), expiry_of(near),
                 label="short front call"),
        Position("put", 6, round_strike(spot * 0.88), expiry_of(far),
                 label="long back put"),
        Position("underlying", 250, label="delta hedge"),
    ], name="example strangle")
