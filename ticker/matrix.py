import os
from datetime import time

import numpy as np
import pandas as pd


def _minute_frame(df, contract, use_mid=False, mode="perfect_foresight",
                  log_returns=False, session_filter=True, lookback=1):
    """
    Expensive part, run ONCE per contract: ticks -> 1-min bars -> forward return
    + signal. Returns a long frame with columns:
        day, tod, fwd, signal_move (points), signal_ret, pos (raw sign).
    The dead-band/threshold is NOT applied here (that is cheap and varies).
    """
    d = df.loc[df["code"] == contract].copy()
    if d.empty:
        raise ValueError(
            f"No rows for code={contract!r}. "
            f"Available codes: {sorted(df['code'].unique())}"
        )

    d["ts"] = pd.to_datetime(d["m_nDatetime"])
    d["px"] = (d["m_nBidPrice"] + d["m_nAskPrice"]) / 2.0 if use_mid else d["m_nPrice"]

    s = d.set_index("ts").sort_index()["px"]
    bars = s.resample("1min").last().dropna().rename("close").reset_index()
    bars["day"] = bars["ts"].dt.normalize()
    bars["tod"] = bars["ts"].dt.time

    morning   = (bars["tod"] >= time(9, 30)) & (bars["tod"] <= time(11, 30))
    afternoon = (bars["tod"] >= time(13, 0)) & (bars["tod"] <= time(15, 0))
    if session_filter:
        bars = bars[morning | afternoon].copy()
    bars["session"] = np.where(bars["tod"] <= time(11, 30), "AM", "PM")

    g = bars.groupby(["day", "session"])
    bars["px_next"] = g["close"].shift(-1)
    bars["px_prev"] = g["close"].shift(lookback)

    if log_returns:
        bars["fwd"] = np.log(bars["px_next"]) - np.log(bars["close"])
    else:
        bars["fwd"] = bars["px_next"] / bars["close"] - 1.0

    if mode == "perfect_foresight":
        bars["signal_move"] = bars["px_next"] - bars["close"]      # future move
    elif mode == "momentum":
        bars["signal_move"] = bars["close"] - bars["px_prev"]      # past move
    else:
        raise ValueError("mode must be 'perfect_foresight' or 'momentum'")

    bars["signal_ret"] = bars["signal_move"] / bars["close"]
    bars["pos"] = np.sign(bars["signal_move"])
    return bars[["day", "tod", "fwd", "signal_move", "signal_ret", "pos"]]


def apply_threshold(frame, threshold=0.0, threshold_unit="return", tick_size=0.2):
    """
    Cheap part, run PER threshold: dead-band on the signal, then pivot to a
    (trading_day x minute_of_day) matrix. Small signals -> flat (0).
    """
    f = frame
    if threshold_unit == "tick":
        keep = f["signal_move"].abs() > threshold * tick_size
    elif threshold_unit == "sigma":
        day_sigma = f["signal_ret"].groupby(f["day"]).transform("std")
        keep = f["signal_ret"].abs() > threshold * day_sigma
    else:  # "return"
        keep = f["signal_ret"].abs() > threshold

    r = f["pos"].where(keep, 0.0) * f["fwd"]
    out = f.assign(r=r)
    R = out.pivot(index="day", columns="tod", values="r").sort_index(axis=1)
    return R


def minute_return_matrix(df, contract, use_mid=False, mode="perfect_foresight",
                         log_returns=False, threshold=0.0, threshold_unit="return",
                         tick_size=0.2, session_filter=True, lookback=1):
    """Convenience wrapper: build the frame then apply one threshold."""
    frame = _minute_frame(df, contract, use_mid, mode, log_returns,
                          session_filter, lookback)
    return apply_threshold(frame, threshold, threshold_unit, tick_size)


def split_by_month(R):
    """Split a day x minute matrix into {'2023-01': df, '2023-02': df, ...}."""
    return {str(p): sub for p, sub in R.groupby(R.index.to_period("M"))}


########################
# Usage (single contract demo)
########################
if __name__ == "__main__":
    df = pd.read_csv(
        "/Users/zhuisabella/xn/ticker/IC_IF_IH_IM_20230104_20230304.csv",
        dtype={"code": "string"}, parse_dates=["m_nDatetime"],
    )
    R = minute_return_matrix(df, "IF0000", use_mid=True, mode="momentum",
                             threshold=10, threshold_unit="tick")
    print(R.shape)
