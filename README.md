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
- `DEFAULT_LOT_SIZE`: Option contract lot size (default: 75 for NIFTY)
- `DEFAULT_UNDERLYING_LOT_SIZE`: Underlying futures lot size (default: 75 for NIFTY-I)

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

### Gamma Scalping with Realistic Lot-Based Hedging

The gamma scalping strategy implemented:

1. **Long Option Position**: Buy option contracts at market open
2. **Delta Hedging**: Maintain delta-neutral position by trading underlying
3. **Lot-Based Constraints**: Realistic hedging limited to whole contract lots
4. **Rebalancing**: Adjust hedge position at specified intervals
5. **P&L Tracking**: Mark-to-market option and hedge positions
6. **Cost Modeling**: Include transaction fees and slippage

### Realistic Hedging Implementation

Unlike theoretical backtests, this implementation accounts for real-world constraints:

#### **Lot Size Constraints**
- **NIFTY Options**: 75 units per contract
- **NIFTY-I Futures**: 75 units per contract
- **Hedging Challenge**: Perfect delta neutrality is impossible with discrete lot sizes

#### **Example Hedging Calculation**
```python
# Option Position: 1 contract (75 units) with delta = -0.45
ideal_hedge = 1 × 0.45 × 75 = 33.75 units  # What we want
available_lots = [0, 75, 150, ...]          # What we can trade
actual_hedge = 0 units (0 lots)             # Closest available

hedge_efficiency = |0 / 33.75| = 0%         # Hedging effectiveness
```

#### **Dynamic Hedge Adjustments**
The system continuously calculates:
- **Ideal Hedge**: Perfect delta-neutral position
- **Actual Hedge**: Rounded to nearest whole lots
- **Hedge Efficiency**: Ratio of actual vs ideal hedging
- **Trade Decision**: Only execute if lot change is required

### Risk Parameters

- Maximum notional exposure limits
- Fee structure per contract
- Slippage modeling in ticks
- Rebalancing frequency controls

## Output

The backtester generates:

- **Time Series Data**: Complete P&L progression with Greeks and hedging metrics
- **Performance Metrics**: Total P&L, hedge efficiency statistics
- **Hedging Analytics**: Ideal vs actual hedge positions, lot utilization
- **Visualizations**: Multi-panel plots showing price action, Greeks, and hedge efficiency
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
