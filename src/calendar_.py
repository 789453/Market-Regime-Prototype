"""Exchange-local session calendar and data-driven session specification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


CME_ROOTS = frozenset({"ES", "NQ", "RTY"})
HK_ROOTS = frozenset({"HSI", "HTI"})
OFFICIAL_CME_SHORT_SESSIONS_2026 = frozenset(
    {date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3)}
)
CME_EXPECTED_5M_BARS_2026 = {
    date(2026, 4, 3): 183,
    date(2026, 5, 25): 228,
    date(2026, 6, 19): 228,
    date(2026, 7, 3): 228,
}
ROOT_TIMEZONES = {
    "ES": "America/New_York",
    "NQ": "America/New_York",
    "RTY": "America/New_York",
    "HSI": "Asia/Hong_Kong",
    "HTI": "Asia/Hong_Kong",
}

CME_PHASES = (
    ("GLOBEX_OPEN", 18 * 60, 20 * 60),
    ("ASIA", 20 * 60, 2 * 60),
    ("EU_OPEN", 2 * 60, 4 * 60),
    ("EU_MORN", 4 * 60, 8 * 60),
    ("US_PRE", 8 * 60, 9 * 60 + 30),
    ("US_OPEN", 9 * 60 + 30, 10 * 60 + 30),
    ("US_MORN", 10 * 60 + 30, 12 * 60),
    ("US_LUNCH", 12 * 60, 14 * 60),
    ("US_PM", 14 * 60, 15 * 60 + 30),
    ("US_CLOSE", 15 * 60 + 30, 16 * 60 + 15),
    ("POST", 16 * 60 + 15, 17 * 60),
)
HK_PHASES = (
    ("HK_NIGHT_OPEN", 17 * 60, 18 * 60),
    ("HK_EU", 18 * 60, 21 * 60 + 30),
    ("HK_US_PRE", 21 * 60 + 30, 22 * 60 + 30),
    ("HK_US_MORN", 22 * 60 + 30, 60),
    ("HK_LATE", 60, 3 * 60),
)


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    start_minute: int
    end_minute: int


@dataclass(frozen=True)
class SessionSpec:
    root: str
    tz: str
    valid_from: str
    valid_to: str
    segments: list[dict[str, str]]
    phases: list[PhaseSpec]


def _contains(minute: int | np.ndarray, start: int, end: int) -> Any:
    if start < end:
        return (minute >= start) & (minute < end)
    return (minute >= start) | (minute < end)


def phase_for_minute(root: str, minute: int) -> str:
    phases = CME_PHASES if root in CME_ROOTS else HK_PHASES
    for name, start, end in phases:
        if bool(_contains(minute, start, end)):
            return name
    return "MAINTENANCE" if root in CME_ROOTS else "OUTSIDE_SESSION"


def _provisional_session_id(local_dates: pd.Series, minutes: pd.Series, market: str) -> pd.Series:
    dates = pd.to_datetime(local_dates)
    if market == "CME":
        dates = dates + pd.to_timedelta((minutes >= 18 * 60).astype(int), unit="D")
    else:
        dates = dates - pd.to_timedelta((minutes < 12 * 60).astype(int), unit="D")
    return pd.Series(dates.dt.date, index=local_dates.index)


def add_calendar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach exchange-local session coordinates using timezone databases."""
    result = frame.copy()
    result["local_minute"] = np.int16(-1)
    result["local_date"] = None
    result["session_id"] = None
    result["session_phase"] = None

    for root, timezone_name in ROOT_TIMEZONES.items():
        mask = result["root"].eq(root)
        if not mask.any():
            continue
        local = result.loc[mask, "timestamp"].dt.tz_convert(timezone_name)
        minutes = (local.dt.hour * 60 + local.dt.minute).astype("int16")
        local_dates = local.dt.date
        market = "CME" if root in CME_ROOTS else "HKEX"
        session_ids = _provisional_session_id(local_dates, minutes, market)
        phases = np.full(mask.sum(), "OUTSIDE_SESSION", dtype=object)
        definitions = CME_PHASES if market == "CME" else HK_PHASES
        minute_values = minutes.to_numpy()
        for phase, start, end in definitions:
            phases[_contains(minute_values, start, end)] = phase

        result.loc[mask, "local_minute"] = minutes.to_numpy()
        result.loc[mask, "local_date"] = local_dates.to_numpy()
        result.loc[mask, "session_id"] = session_ids.to_numpy()
        result.loc[mask, "session_phase"] = phases

    result["local_minute"] = result["local_minute"].astype("int16")
    return result


def _parse_clock(value: str) -> int:
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)


def add_session_positions(frame: pd.DataFrame, specs: list[SessionSpec] | None = None) -> pd.DataFrame:
    result = frame.sort_values(["root", "timestamp"], kind="stable").reset_index(drop=True)
    grouped = result.groupby(["root", "session_id"], sort=False, observed=True)
    result["bar_in_session"] = grouped.cumcount().astype("int16")
    expected_bars = np.where(result["root"].isin(CME_ROOTS), 276, 120)
    if specs is not None:
        session_dates = pd.to_datetime(result["session_id"])
        for spec in specs:
            segment = spec.segments[0]
            duration = (_parse_clock(segment["end"]) - _parse_clock(segment["start"])) % (24 * 60)
            mask = (
                result["root"].eq(spec.root)
                & session_dates.between(spec.valid_from, spec.valid_to)
            )
            expected_bars = np.where(mask, duration // 5, expected_bars)
    for session_date, bars in CME_EXPECTED_5M_BARS_2026.items():
        expected_bars = np.where(
            result["root"].isin(CME_ROOTS) & result["session_id"].eq(session_date), bars, expected_bars
        )
    result["phi"] = (result["bar_in_session"] / (expected_bars - 1)).clip(upper=1.0)
    result["is_half_day"] = result["root"].isin(CME_ROOTS) & result["session_id"].isin(
        OFFICIAL_CME_SHORT_SESSIONS_2026
    )

    last_session = result.groupby("root", observed=True)["session_id"].transform("max")
    complete = result["session_id"] < last_session
    cme = result["root"].isin(CME_ROOTS)
    anchor = np.where(cme, 18 * 60, 17 * 60)
    elapsed = (result["local_minute"].to_numpy() - anchor) % (24 * 60)
    result["_session_elapsed"] = elapsed
    starts = grouped["_session_elapsed"].transform("min")
    ends = grouped["_session_elapsed"].transform("max")
    symbol_last_session = result.groupby(["root", "local_symbol"], observed=True)["session_id"].transform("max")
    root_last_symbol = result.groupby("root", observed=True)["local_symbol"].transform("last")
    is_truncated_roll_session = result["session_id"].eq(symbol_last_session) & result["local_symbol"].ne(root_last_symbol)
    expected_end = np.where(cme, 23 * 60 - 1, 10 * 60 - 1)
    result["is_truncated_session"] = (
        complete
        & ~result["is_half_day"]
        & ~is_truncated_roll_session
        & (ends.to_numpy() < expected_end - 5)
    )
    return result.drop(columns="_session_elapsed")


def _clock(minute: int) -> str:
    minute %= 24 * 60
    return f"{minute // 60:02d}:{minute % 60:02d}"


def infer_session_specs(frame: pd.DataFrame) -> tuple[list[SessionSpec], pd.DataFrame]:
    """Infer dated session segments from observed first/last local bars."""
    annotated = add_calendar_columns(frame)
    annotated["session_elapsed"] = np.where(
        annotated["root"].isin(CME_ROOTS),
        (annotated["local_minute"] - 18 * 60) % (24 * 60),
        (annotated["local_minute"] - 17 * 60) % (24 * 60),
    )
    daily = (
        annotated.groupby(["root", "session_id"], observed=True)
        .agg(start_elapsed=("session_elapsed", "min"), end_elapsed=("session_elapsed", "max"), bars=("timestamp", "size"))
        .reset_index()
    )
    daily["anchor"] = np.where(daily["root"].isin(CME_ROOTS), 18 * 60, 17 * 60)
    daily["start_minute"] = (daily["anchor"] + daily["start_elapsed"]) % (24 * 60)
    daily["end_minute"] = (daily["anchor"] + daily["end_elapsed"]) % (24 * 60)
    specs: list[SessionSpec] = []
    for root, root_daily in daily.groupby("root", sort=True):
        root_daily = root_daily.sort_values("session_id").copy()
        normal_open = root_daily["start_elapsed"].le(30)
        opening = (root_daily.loc[normal_open, "start_elapsed"].div(15).round() * 15).astype(int)
        segment_starts = [0]
        if opening.nunique() > 1:
            final_open = int(opening.tail(min(10, len(opening))).mode().iloc[0])
            quantized = (root_daily["start_elapsed"].div(15).round() * 15).astype(int)
            candidates = root_daily.index[normal_open & quantized.eq(final_open)]
            for candidate in candidates:
                tail = root_daily.loc[candidate:]
                tail = (tail.loc[tail["start_elapsed"].le(30), "start_elapsed"].div(15).round() * 15).astype(int)
                if len(tail) >= 5 and tail.eq(final_open).mean() >= 0.9:
                    segment_starts.append(root_daily.index.get_loc(candidate))
                    break
        segment_starts = sorted(set(segment_starts))
        for position, start_idx in enumerate(segment_starts):
            end_idx = segment_starts[position + 1] - 1 if position + 1 < len(segment_starts) else len(root_daily) - 1
            segment = root_daily.iloc[start_idx : end_idx + 1]
            normal_segment = segment.loc[segment["start_elapsed"].le(30)]
            start_elapsed = int(normal_segment["start_elapsed"].median())
            end_elapsed = int(segment["end_elapsed"].quantile(0.95, interpolation="nearest"))
            anchor = 18 * 60 if root in CME_ROOTS else 17 * 60
            start_minute = (anchor + start_elapsed) % (24 * 60)
            end_minute = (anchor + end_elapsed) % (24 * 60)
            market = "CME" if root in CME_ROOTS else "HKEX"
            phases = CME_PHASES if market == "CME" else HK_PHASES
            specs.append(
                SessionSpec(
                    root=root,
                    tz=ROOT_TIMEZONES[root],
                    valid_from=str(segment["session_id"].iloc[0]),
                    valid_to=str(segment["session_id"].iloc[-1]),
                    segments=[{"start": _clock(start_minute), "end": _clock((end_minute + 1) % (24 * 60))}],
                    phases=[PhaseSpec(name, start, end) for name, start, end in phases],
                )
            )
    return specs, daily


def write_session_specs(specs: list[SessionSpec], path: str | Path) -> None:
    payload = {"session_defs": []}
    for spec in specs:
        item = asdict(spec)
        item["phases"] = [asdict(phase) for phase in spec.phases]
        payload["session_defs"].append(item)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
