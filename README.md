<p align="center">
  <img src="assets/logo.png" width="140" alt="TMO logo">
</p>

<h1 align="center">TooManyOptions</h1>
<p align="center"><i>Draw any payoff. Replicate it with options.</i></p>

TooManyOptions is a Python desktop application for drawing a target payoff and approximating it with the underlying asset and options. It helps students, researchers, and options practitioners explore static payoff replication without manually deriving and assembling each portfolio. Replications can use either a dense numerical strike grid or filtered option-chain data for supported index tickers.

## Features

- Draw a payoff curve in a Tkinter interface and compare it with its replicated payoff.
- Run a numerical replication on a dense strike grid.
- Load index prices, expirations, and option chains from Yahoo Finance using `yfinance`.
- Restrict market-data loading to `^SPX`, `^XSP`, `^NDX`, `^NQX`, `^RUT`, and `^DJX`.
- Filter listed options by minimum daily volume and inspect eligible strike boundaries.
- Review payoff error plots and the resulting position quantities.

## Architecture

The project is implemented in Python. The GUI uses Tkinter and embeds Matplotlib charts; numerical calculations use NumPy, SciPy, SymPy, and pandas. Yahoo Finance data is accessed through `yfinance`.

| Path                                 | Purpose                                                                                     |
| ------------------------------------ | ------------------------------------------------------------------------------------------- |
| `app.py`                           | Application entry point; opens the welcome screen and drawing interface.                    |
| `model/draw.py`                    | Tkinter GUI, payoff drawing and smoothing, market-data controls, and result visualizations. |
| `model/carrmadan.py`               | Symbolic, numerical, and market-option payoff replication calculations.                     |
| `model/loader.py`                  | Supported ticker validation and Yahoo Finance price, expiration, and option-chain loading.  |
| `tests/test_market_replication.py` | Unit tests for replication, strike selection, market-data loading, and chart behavior.      |
| `requirements.txt`                 | Python package dependencies.                                                                |

```text
.
├── app.py
├── model/
│   ├── carrmadan.py
│   ├── draw.py
│   └── loader.py
├── tests/
│   └── test_market_replication.py
└── requirements.txt
```

## Installation

Python 3.10 or newer is required by the type syntax used in the source. Tkinter must also be available in the Python installation. The project uses `pip` and `requirements.txt`; dependency versions are not pinned.

### Windows PowerShell

From the project root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, invoke the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No environment variables or `.env` file are required. Internet access is needed to retrieve Yahoo Finance market data; the numerical-grid replication does not fetch external data.

## Run

Start the desktop application from the project root:

```bash
python app.py
```

In the app:

1. Choose **Start drawing** and draw the target payoff from left to right.
2. Set the initial underlying price, $S_0$.
3. For an artificial numerical-grid replication, choose **Calculate static replication**.
4. For real listed options, select one of the supported index tickers, load its data, select an expiration, set the minimum daily volume in contracts, and choose **Replicate with real options**.

The market-data controls accept only `^SPX`, `^XSP`, `^NDX`, `^NQX`, `^RUT`, and `^DJX`. This is an intentional product-scope restriction: the listed index-option products represented by these symbols are European-style, so they are exercised only at expiration, matching the terminal-payoff assumption in the static replication formula. The implementation does not model early exercise; American-style equity and ETF options are therefore not supported in the listed-options workflow. The numerical-grid replication is independent of market tickers.

The volume threshold is the number of contracts traded that day, not dollar volume or open interest.

## Quickstart

The following standalone example calls the numerical replication pipeline for a quadratic target payoff. Run it from the project root after installing the requirements:

```python
import numpy as np

from model.carrmadan import run_numeric_replication_pipeline

underlying_prices = np.linspace(50.0, 150.0, 1_000)
target_payoff = (underlying_prices - 100.0) ** 2

result = run_numeric_replication_pipeline(
    strike_grid=underlying_prices,
    payoff_values=target_payoff,
    initial_price=100.0,
)

print(f"Underlying units per payoff unit: {result.asset_quantity:.6f}")
print(f"Fixed cash flow at expiration: {result.fixed_component:.6f}")
print(f"Maximum replication error: {np.max(result.absolute_error):.6f}")
```

The desktop workflow can also be launched directly with `python app.py`.

## Methodology & Theoretical Background

For a sufficiently smooth target payoff $f(S_T)$, the static replication decomposition is

$f(S_T) = \underbrace{f(S_0)-S_0 f'(S_0)}_{\text{fixed cash flow at expiration}} + f'(S_0)S_T + \int_0^{S_0} f''(K)(K-S_T)^+\,dK + \int_{S_0}^{\infty} f''(K)(S_T-K)^+\,dK$

where $S_0$ is the initial underlying price, $S_T$ is its value at expiration, $K$ is an option strike, and $x^+=\max(x,0)$. The underlying position is $f'(S_0)$; the put and call positions approximate the two integrals using the payoff curvature $f''(K)$.

This decomposition is a statement about terminal payoffs using European exercise: the options contribute their specified expiration payoffs at $S_T$. Early exercise, available with American-style contracts, is not represented by these payoff terms and is not modeled by this application. In the listed-options mode, the European-style index-option restriction is intended to align the selected contracts with this assumption.

The GUI smooths the hand-drawn points with a Gaussian filter and a Savitzky–Golay filter, then estimates the first and second derivatives on a 1,000-point price grid. The numerical-grid mode approximates option quantities by $w_i \approx f''(K_i)\Delta K_i$. In market mode, only finite-volume, out-of-the-money options meeting the minimum-volume threshold and lying within the drawn price range are selected. Curvature is interpolated at each selected strike and multiplied by an irregular-strike quadrature width: interior widths span the midpoints to neighboring strikes, while the first and last widths use one full adjacent-strike interval.

The displayed fixed cash value is a **cash flow in the expiration payoff**, not an amount invested or borrowed at $t=0$. Positive means receive cash at expiration; negative means pay cash at expiration. The application does not discount this value or calculate initial funding requirements.

## Data and limitations

- Market prices and option chains come from Yahoo Finance through `yfinance`. The latest quote is preferred; if unavailable, the loader falls back to the latest non-null daily close in the recent history.
- The app uses option-chain daily volume as a liquidity filter. It does not use option premiums or model bid-ask spreads, fees, taxes, funding or borrowing rates, early exercise, or changes before expiration.
- Results are static payoff approximations per payoff unit, not trade recommendations or execution instructions. Estimates depend on the hand-drawn curve, smoothing, selected strikes, and available market data.

## Tests

Run the unit tests from the project root:

```bash
python -m unittest discover -s tests -v
```

## References

Carr, Peter, and Dilip Madan. “Towards a Theory of Volatility Trading.” In *Volatility: New Developments in Options and Derivatives*, edited by Robert Jarrow, 417–427. Risk Books, 1998.

```bibtex
@incollection{carr1998towards,
  author    = {Carr, Peter and Madan, Dilip},
  title     = {Towards a Theory of Volatility Trading},
  booktitle = {Volatility: New Developments in Options and Derivatives},
  editor    = {Jarrow, Robert},
  publisher = {Risk Books},
  year      = {1998},
  pages     = {417--427}
}
```
