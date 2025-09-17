# Options Greeks Calculator & Gamma Scalping Backtester

A Python-based financial analytics tool for options trading that calculates Black-Scholes Greeks and performs gamma scalping backtesting using real market data from the FutPrint API.

## Features

- **Black-Scholes Options Pricing**: Calculate theoretical option prices using the Black-Scholes model
- **Greeks Calculation**: Compute Delta, Gamma, Theta, Vega, and Rho for options
- **Implied Volatility**: Extract implied volatility from market prices
- **Gamma Scalping Backtesting**: Simulate delta-hedged gamma scalping strategies
- **Visualization**: Generate colored plots showing P&L progression and Greeks evolution
- **Data Export**: Export results to CSV for further analysis

## Installation

1. Clone or download this repository
2. Install Python 3.8+ 
3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your environment variables:

```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your actual API credentials
```

## Configuration

### Environment Variables

Create a `.env` file based on `env.example` and configure the following:

- `API_OHLCV_URL`: FutPrint API endpoint for OHLCV data
- `API_TOKEN`: Your FutPrint API Bearer token
- `API_KEY`: Admin API key (usually 'FutPrintIN')
- `RISK_FREE_RATE`: Risk-free rate for options pricing (default: 0.06)
- `DEFAULT_LOT_SIZE`: Contract lot size (default: 50 for NIFTY)

### Trading Parameters

- `DEFAULT_FEE_PER_CONTRACT`: Trading fees per contract
- `DEFAULT_SLIPPAGE_TICKS`: Slippage in ticks
- `DEFAULT_TICK_SIZE`: Minimum price increment

## Usage

### Basic Example

```python
from main import fetch_ohlcv, backtest_gamma_scalp, plot_results

# Define time range
START = "2025-09-02T03:45:00.000Z"
END = "2025-09-02T23:59:59.000Z"

# Fetch data
underlying_data = fetch_ohlcv("NIFTY-I", START, END)
option_data = fetch_ohlcv("NIFTY25090924900PE", START, END)

# Run backtest
results = backtest_gamma_scalp(
    underlying_data, 
    option_data, 
    "NIFTY25090924900PE", 
    rebalance_minutes=1, 
    entry_qty=1
)

# Display results
print(f"Total P&L: {results['total_pnl']}")

# Plot results
plot_results(results['pnl_timeseries'])

# Export to CSV
results['pnl_timeseries'].to_csv("gamma_scalp_results.csv")
```

### Option Symbol Format

Option symbols follow the format: `[UNDERLYING][EXPIRY][STRIKE][TYPE]`

- **UNDERLYING**: 5 characters (e.g., "NIFTY")
- **EXPIRY**: YYMMDD format (e.g., "250909")
- **STRIKE**: Strike price (e.g., "24900")
- **TYPE**: "CE" for Call, "PE" for Put

Example: `NIFTY25090924900PE` = NIFTY Put expiring Sep 9, 2025, strike 24900

## Key Functions

### Core Functions

- `fetch_ohlcv(symbol, start_iso, end_iso)`: Fetch OHLCV data from FutPrint API
- `prepare_option_greeks(und_df, opt_df, option_symbol)`: Calculate Greeks for option time series
- `backtest_gamma_scalp(...)`: Run gamma scalping backtest simulation

### Utility Functions

- `bs_price_vec(opt_type, S, K, T, r, sigma)`: Black-Scholes option pricing
- `implied_vol_scalar(...)`: Calculate implied volatility
- `bs_greeks_scalar(...)`: Calculate option Greeks
- `plot_results(df)`: Generate visualization plots

## Strategy Details

### Gamma Scalping

The gamma scalping strategy implemented:

1. **Long Option Position**: Buy option contracts at market open
2. **Delta Hedging**: Maintain delta-neutral position by trading underlying
3. **Rebalancing**: Adjust hedge position at specified intervals
4. **P&L Tracking**: Mark-to-market option and hedge positions
5. **Cost Modeling**: Include transaction fees and slippage

### Risk Parameters

- Maximum notional exposure limits
- Fee structure per contract
- Slippage modeling in ticks
- Rebalancing frequency controls

## Output

The backtester generates:

- **Time Series Data**: Complete P&L progression with Greeks
- **Performance Metrics**: Total P&L, Sharpe ratio, maximum drawdown
- **Visualizations**: Multi-panel plots showing price action and Greeks
- **CSV Export**: Detailed results for external analysis

## Dependencies

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `scipy`: Scientific computing (optimization, statistics)
- `requests`: HTTP API calls
- `pytz`: Timezone handling
- `matplotlib`: Plotting and visualization

## API Requirements

This tool requires access to the FutPrint API for market data. You'll need:

- Valid API credentials (Bearer token)
- Access to historical OHLCV data
- Sufficient API rate limits for your analysis period

## License

This project is for educational and research purposes. Please ensure compliance with your broker's API terms of service and applicable financial regulations.

## Disclaimer

This software is for educational purposes only. Past performance does not guarantee future results. Options trading involves substantial risk and may not be suitable for all investors. Please consult with a qualified financial advisor before making investment decisions.
