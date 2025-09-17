# Simplified backtester for Delta-hedged Gamma-Scalping
# Requirements: pandas, numpy, requests, scipy, pytz

import pandas as pd
import numpy as np
from io import StringIO
import requests
from datetime import datetime, timedelta
import pytz
from scipy.stats import norm
from scipy.optimize import brentq

# --------- Insert your API details ----------
API_OHLCV = "https://api.futprint.in/api/historical-data-ohlcv-oi"
HEADERS = {"authorization": "Bearer <YOUR_TOKEN>", "x-admin-api-key": "FutPrintIN", "accept":"*/*"}

# --------- Utility BS & IV (vectorizable approach) ----------
def bs_price_vec(opt_type, S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt_type == "CE":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

# scalar iv solver (used per-row)
def implied_vol_scalar(opt_type, price, S, K, T, r, q=0.0):
    if price <= 0 or S <= 0 or T <= 0:
        return np.nan
    def f(sig): return bs_price_vec(opt_type, S, K, T, r, sig, q) - price
    try:
        return brentq(f, 1e-6, 5.0, maxiter=200)
    except:
        return np.nan

# bs greeks scalar
def bs_greeks_scalar(opt_type, S, K, T, r, sigma, q=0.0):
    if T<=0 or sigma<=0 or S<=0:
        return (np.nan,)*5
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    pdf = norm.pdf(d1)
    if opt_type=="CE":
        delta = np.exp(-q*T)*norm.cdf(d1)
        theta = (-S*pdf*sigma*np.exp(-q*T)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2) + q*S*np.exp(-q*T)*norm.cdf(d1))
        rho = K*T*np.exp(-r*T)*norm.cdf(d2)
    else:
        delta = -np.exp(-q*T)*norm.cdf(-d1)
        theta = (-S*pdf*sigma*np.exp(-q*T)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2) - q*S*np.exp(-q*T)*norm.cdf(-d1))
        rho = -K*T*np.exp(-r*T)*norm.cdf(-d2)
    gamma = pdf*np.exp(-q*T)/(S*sigma*np.sqrt(T))
    vega = S*np.exp(-q*T)*pdf*np.sqrt(T)
    return delta, gamma, theta/365.0, vega, rho

# --------- fetch OHLCV helper ----------
def fetch_ohlcv(symbol, start_iso, end_iso):
    resp = requests.get(API_OHLCV, params={"symbol":symbol,"start_time":start_iso,"end_time":end_iso,"interval":"1m"}, headers=HEADERS)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), parse_dates=["interval_start"])
    df = df.sort_values("interval_start").reset_index(drop=True)
    df["interval_start"] = pd.to_datetime(df["interval_start"], utc=True)
    return df

# --------- simplified pipeline to compute per-interval greeks for one option ----------
def prepare_option_greeks(und_df, opt_df, option_symbol, r=0.06, q=0.0, expiry_time="15:30:00"):
    # merge
    merged = pd.merge(opt_df, und_df[["interval_start","close"]].rename(columns={"close":"underlying"}), on="interval_start", how="inner")
    # parse strike/opt type
    base = option_symbol[:5]; expiry_raw = option_symbol[5:11]; strike = int(option_symbol[11:-2]); opt_type = option_symbol[-2:]
    expiry_date = datetime.strptime(expiry_raw,"%y%m%d").date()
    hh,mm,ss = map(int, expiry_time.split(":"))
    expiry_local = pytz.timezone("Asia/Kolkata").localize(datetime(expiry_date.year,expiry_date.month,expiry_date.day,hh,mm,ss)).astimezone(pytz.UTC)
    # compute T and price
    merged["T"] = (expiry_local - merged["interval_start"]).dt.total_seconds().clip(lower=0) / (365*24*3600)
    merged["option_price"] = merged["close"]  # use close
    # compute IV sequentially (vectorizing robustly is complex; do row-wise)
    ivs=[]; deltas=[]; gammas=[]; thetas=[]; vegas=[]; rhos=[]
    for _,row in merged.iterrows():
        S = float(row["underlying"]); price = float(row["option_price"]); T=float(row["T"])
        iv = implied_vol_scalar(opt_type, price, S, strike, T, r, q)
        if np.isnan(iv):
            ivs.append(np.nan); deltas.append(np.nan); gammas.append(np.nan); thetas.append(np.nan); vegas.append(np.nan); rhos.append(np.nan)
        else:
            ivs.append(iv)
            d,g,th,v,rho = bs_greeks_scalar(opt_type,S,strike,T,r,iv,q)
            deltas.append(d); gammas.append(g); thetas.append(th); vegas.append(v); rhos.append(rho)
    merged["iv"]=ivs; merged["delta"]=deltas; merged["gamma"]=gammas; merged["theta"]=thetas; merged["vega"]=vegas; merged["rho"]=rhos
    return merged

# --------- Backtest engine for Gamma-Scalp ----------
def backtest_gamma_scalp(und_df, opt_df, option_symbol, rebalance_minutes=1, entry_qty=1, max_notional=1_000_000,
                         fee_per_contract=10.0, slippage_ticks=1, tick_size=1.0, r=0.06):
    """
    - Buy 'entry_qty' option contracts at time 0 (first interval) and delta-hedge immediately.
    - Rebalance underlying hedge every 'rebalance_minutes'.
    - Track P&L from options (marked to market) and underlying hedge trading.
    """
    # prepare greeks
    merged = prepare_option_greeks(und_df,opt_df,option_symbol,r=r)
    if merged.empty:
        return None
    merged = merged.set_index("interval_start")
    # initial position: buy entry_qty contracts
    qty_opts = entry_qty  # positive means long options
    lot_multiplier = 50  # example NIFTY lot size; replace with correct
    contracts = qty_opts
    # bookkeeping
    cash = 0.0
    underlying_pos = 0.0
    pnl_hist=[]
    last_rebalance_time = None
    for t, row in merged.iterrows():
        # mark option position
        mtm_options = contracts * row["option_price"] * lot_multiplier * 1.0  # option price in points * multiplier
        # compute desired hedge to neutralize delta: hedge units = contracts * delta * lot_multiplier
        if np.isnan(row["delta"]):
            desired_hedge = 0.0
        else:
            desired_hedge = - contracts * row["delta"] * lot_multiplier  # negative: short underlying
        # rebalancing rule: time-based
        if last_rebalance_time is None or (t - last_rebalance_time) >= pd.Timedelta(minutes=rebalance_minutes):
            # trade underlying to desired_hedge
            trade_size = desired_hedge - underlying_pos
            if abs(trade_size) > 0:
                # trade price: approximate using underlying close price with slippage
                trade_price = row["underlying"] + slippage_ticks * tick_size * np.sign(trade_size)
                trade_cost = trade_price * trade_size
                # fees
                fees = fee_per_contract * abs(trade_size)/lot_multiplier  # scale fee to contracts
                cash -= trade_cost + fees
                underlying_pos = desired_hedge
            last_rebalance_time = t
        # portfolio mark-to-market
        mtm_underlying = underlying_pos * row["underlying"]
        total_mv = mtm_options + mtm_underlying + cash
        # store metrics
        pnl_hist.append({
            "timestamp": t, "option_price": row["option_price"], "iv": row["iv"],
            "contracts": contracts, "delta": row["delta"], "gamma": row["gamma"],
            "underlying_pos": underlying_pos, "mv_options": mtm_options, "mv_underlying": mtm_underlying,
            "portfolio_value": total_mv
        })
    df_pnl = pd.DataFrame(pnl_hist).set_index("timestamp")
    # compute returns
    df_pnl["pnl_change"] = df_pnl["portfolio_value"].diff().fillna(0)
    total_pnl = df_pnl["pnl_change"].sum()
    return {"pnl_timeseries": df_pnl, "total_pnl": total_pnl}

# --------- Example run ----------
# if __name__=="__main__":
#     START = "2025-09-05T03:45:00.000Z";
#     END   = "2025-09-06T04:15:00.000Z";
#     und = fetch_ohlcv("NIFTY-I", START, END)
#     opt = fetch_ohlcv("NIFTY25090924900PE", START, END)
#     res = backtest_gamma_scalp(und,opt,"NIFTY25090924900PE", rebalance_minutes=1, entry_qty=1)
#     print("Total P&L:", res["total_pnl"])
#     print(res["pnl_timeseries"])



import matplotlib.pyplot as plt

# --------- Backtest engine for Gamma-Scalp (unchanged above) ---------

def backtest_gamma_scalp(und_df, opt_df, option_symbol, rebalance_minutes=1, entry_qty=1, max_notional=1_000_000,
                         fee_per_contract=10.0, slippage_ticks=1, tick_size=1.0, r=0.06):
    merged = prepare_option_greeks(und_df,opt_df,option_symbol,r=r)
    if merged.empty:
        return None
    merged = merged.set_index("interval_start")
    qty_opts = entry_qty
    lot_multiplier = 50  # Example lot size
    contracts = qty_opts
    cash = 0.0
    underlying_pos = 0.0
    pnl_hist=[]
    last_rebalance_time = None

    for t, row in merged.iterrows():
        mtm_options = contracts * row["option_price"] * lot_multiplier
        if np.isnan(row["delta"]):
            desired_hedge = 0.0
        else:
            desired_hedge = - contracts * row["delta"] * lot_multiplier

        if last_rebalance_time is None or (t - last_rebalance_time) >= pd.Timedelta(minutes=rebalance_minutes):
            trade_size = desired_hedge - underlying_pos
            if abs(trade_size) > 0:
                trade_price = row["underlying"] + slippage_ticks * tick_size * np.sign(trade_size)
                trade_cost = trade_price * trade_size
                fees = fee_per_contract * abs(trade_size)/lot_multiplier
                cash -= trade_cost + fees
                underlying_pos = desired_hedge
            last_rebalance_time = t

        mtm_underlying = underlying_pos * row["underlying"]
        total_mv = mtm_options + mtm_underlying + cash

        pnl_hist.append({
            "timestamp": t,
            "underlying": row["underlying"],
            "option_price": row["option_price"],
            "iv": row["iv"],
            "contracts": contracts,
            "delta": row["delta"],
            "gamma": row["gamma"],
            "theta": row["theta"],
            "vega": row["vega"],
            "rho": row["rho"],
            "underlying_pos": underlying_pos,
            "mv_options": mtm_options,
            "mv_underlying": mtm_underlying,
            "cash": cash,
            "portfolio_value": total_mv
        })

    df_pnl = pd.DataFrame(pnl_hist).set_index("timestamp")
    df_pnl["pnl_change"] = df_pnl["portfolio_value"].diff().fillna(0)
    df_pnl["cum_pnl"] = df_pnl["pnl_change"].cumsum()
    total_pnl = df_pnl["cum_pnl"].iloc[-1]
    return {"pnl_timeseries": df_pnl, "total_pnl": total_pnl}

# --------- Plotting Utility ---------
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
import numpy as np

def plot_colored_line(ax, x, y, cmap="viridis", label=None):
    """
    Plot a line with gradient colors according to y values.
    """
    # prepare segments for LineCollection
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # normalize y for colormap scaling
    norm = mcolors.Normalize(vmin=np.nanmin(y), vmax=np.nanmax(y))
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(y)
    lc.set_linewidth(2)

    # add collection to axis
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(np.nanmin(y), np.nanmax(y))
    if label: ax.set_title(label)
    ax.grid(True)

    # add colorbar for reference
    plt.colorbar(lc, ax=ax, orientation="vertical", label=label)


def plot_results(df):
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True)

    # convert index to numeric for LineCollection
    x = df.index.values.astype(float)  # numeric timestamps

    plot_colored_line(axes[0,0], x, df["underlying"].values, cmap="plasma", label="Underlying Price")
    plot_colored_line(axes[0,1], x, df["option_price"].values, cmap="cividis", label="Option Price")
    plot_colored_line(axes[1,0], x, df["iv"].values, cmap="viridis", label="Implied Volatility")
    plot_colored_line(axes[1,1], x, df["delta"].values, cmap="coolwarm", label="Delta")
    plot_colored_line(axes[2,0], x, df["gamma"].values, cmap="magma", label="Gamma")
    plot_colored_line(axes[2,1], x, df["theta"].values, cmap="PuBuGn", label="Theta")
    plot_colored_line(axes[3,0], x, df["vega"].values, cmap="YlGnBu", label="Vega")
    plot_colored_line(axes[3,1], x, df["cum_pnl"].values, cmap="RdYlGn", label="Cumulative PnL")

    plt.tight_layout()
    plt.show()


# --------- Example run with plots & CSV ---------
if __name__=="__main__":
    START = "2025-09-02T03:45:00.000Z"
    END   = "2025-09-02T23:59:59.000Z"
    und = fetch_ohlcv("NIFTY-I", START, END)
    opt = fetch_ohlcv("NIFTY25090924900PE", START, END)
    res = backtest_gamma_scalp(und,opt,"NIFTY25090924900PE", rebalance_minutes=1, entry_qty=1)

    df_pnl = res["pnl_timeseries"]
    print("Total P&L:", res["total_pnl"])
    print(df_pnl.head())

    # plot everything
    plot_results(df_pnl)

    # export CSV
    df_pnl.to_csv("gamma_scalp_results.csv")
    print("Results exported to gamma_scalp_results.csv")
