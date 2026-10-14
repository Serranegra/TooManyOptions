"""Functions for analyzing and replicating option payoffs."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
NumericFunction = Callable[..., Any]


@dataclass(frozen=True)
class ReplicationResult:
    """Numerical results and portfolio positions for a payoff replication.

    Array fields are ordered by increasing underlying price or option strike.
    Put and call weights are the corresponding option quantities. The fixed
    cash component is the constant cash flow at expiration: positive is a
    receipt and negative is a payment. It is not discounted to an initial
    funding amount. All quantities are expressed per payoff unit.
    """

    expiry_grid: FloatArray
    asset_quantity: float
    fixed_component: float
    synthetic_payoff: FloatArray
    theoretical_payoff: FloatArray
    absolute_error: FloatArray
    info: str
    put_strikes: FloatArray
    put_weights: FloatArray
    call_strikes: FloatArray
    call_weights: FloatArray


def convert_payoff_function(
    strike_symbol: sp.Symbol,
    symbolic_payoff: sp.Expr,
) -> tuple[NumericFunction, NumericFunction, NumericFunction]:
    """Compile a symbolic payoff and its first two derivatives for NumPy.

    Returns the payoff, first-derivative, and second-derivative functions,
    in that order.
    """
    first_derivative = sp.diff(symbolic_payoff, strike_symbol, 1)
    second_derivative = sp.diff(symbolic_payoff, strike_symbol, 2)

    payoff_function = sp.lambdify(strike_symbol, symbolic_payoff, "numpy")
    first_derivative_function = sp.lambdify(
        strike_symbol, first_derivative, "numpy"
    )
    second_derivative_function = sp.lambdify(
        strike_symbol, second_derivative, "numpy"
    )

    return payoff_function, first_derivative_function, second_derivative_function


def calculate_strike_spacing(price_grid: FloatArray) -> float:
    """Return the spacing between the first two points of a uniform grid."""
    return float(price_grid[1] - price_grid[0])


def calculate_option_weights(
    price_grid: FloatArray,
    second_derivative_function: NumericFunction,
    strike_spacing: float,
    initial_price: float,
) -> tuple[FloatArray, FloatArray]:
    """Approximate option weights from payoff curvature on a uniform grid.

    Returns put weights below the initial price and call weights above it.
    """
    derivative_values = second_derivative_function(price_grid)
    weights = np.ones_like(price_grid) * derivative_values * strike_spacing

    call_mask = price_grid > initial_price
    put_mask = price_grid < initial_price

    return weights[put_mask], weights[call_mask]


def calculate_asset_quantity(
    initial_price: float,
    first_derivative_function: NumericFunction,
) -> float:
    """Return the underlying position implied by the payoff slope at spot."""
    return float(first_derivative_function(initial_price))


def calculate_fixed_component(
    initial_price: float,
    payoff_function: NumericFunction,
    first_derivative_function: NumericFunction,
) -> float:
    """Return the fixed cash payoff at expiration matching value and slope at spot.

    A positive value is a cash receipt at expiration; a negative value is a
    cash payment at expiration. This amount is not discounted to t=0.
    """
    return float(
        payoff_function(initial_price)
        - first_derivative_function(initial_price) * initial_price
    )


def create_expiry_grid() -> FloatArray:
    """Create the default terminal-price grid for symbolic payoff plots."""
    return np.linspace(1, 100, 1000)


def calculate_payoffs(
    expiry_grid: FloatArray,
    initial_price: float,
    put_strikes: FloatArray,
    put_weights: FloatArray,
    call_strikes: FloatArray,
    call_weights: FloatArray,
    fixed_component: float,
    payoff_function: NumericFunction,
    first_derivative_function: NumericFunction,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate the portfolio payoff and target payoff on a terminal-price grid."""
    asset_payoff = first_derivative_function(initial_price) * expiry_grid

    put_matrix = np.maximum(
        put_strikes[None, :] - expiry_grid[:, None],
        0,
    )
    call_matrix = np.maximum(
        expiry_grid[:, None] - call_strikes[None, :],
        0,
    )

    put_payoff = put_matrix @ put_weights
    call_payoff = call_matrix @ call_weights

    synthetic_payoff = (
        fixed_component + asset_payoff + put_payoff + call_payoff
    )
    theoretical_payoff = payoff_function(expiry_grid)

    return synthetic_payoff, theoretical_payoff


def calculate_absolute_error(
    synthetic_payoff: FloatArray,
    theoretical_payoff: FloatArray,
) -> FloatArray:
    """Return the pointwise absolute difference between two payoff arrays."""
    return np.abs(synthetic_payoff - theoretical_payoff)


def format_portfolio_summary(
    put_count: int,
    call_count: int,
    asset_quantity: float,
    fixed_component: float,
    initial_price: float,
) -> str:
    """Format a summary of the portfolio data."""
    lines = [
        "=" * 50,
        f"Number of puts: {put_count}",
        f"Number of calls: {call_count}",
        f"Underlying units to hold: {asset_quantity:.6f}",
        (
            "Fixed cash flow at expiration "
            f"(positive = receive, negative = pay): {fixed_component:.6f}"
        ),
        f"Initial underlying price: {initial_price:.6f}",
        "=" * 50,
    ]
    return "\n".join(lines)


def run_replication_pipeline(
    strike_symbol: sp.Symbol,
    symbolic_payoff: sp.Expr,
    strike_grid: FloatArray,
    initial_price: float,
) -> ReplicationResult:
    """Replicate a symbolic payoff using a uniform grid of option strikes.

    The payoff slope and level at ``initial_price`` determine the underlying
    and fixed-cash positions. The second derivative determines option weights.
    """
    (
        payoff_function,
        first_derivative_function,
        second_derivative_function,
    ) = convert_payoff_function(strike_symbol, symbolic_payoff)

    strike_spacing = calculate_strike_spacing(strike_grid)
    put_weights, call_weights = calculate_option_weights(
        strike_grid,
        second_derivative_function,
        strike_spacing,
        initial_price,
    )

    put_mask = strike_grid < initial_price
    call_mask = strike_grid > initial_price
    put_strikes = strike_grid[put_mask]
    call_strikes = strike_grid[call_mask]

    call_count = len(call_strikes)
    put_count = len(put_strikes)

    asset_quantity = calculate_asset_quantity(
        initial_price,
        first_derivative_function,
    )
    fixed_component = calculate_fixed_component(
        initial_price,
        payoff_function,
        first_derivative_function,
    )

    expiry_grid = create_expiry_grid()
    synthetic_payoff, theoretical_payoff = calculate_payoffs(
        expiry_grid,
        initial_price,
        put_strikes,
        put_weights,
        call_strikes,
        call_weights,
        fixed_component,
        payoff_function,
        first_derivative_function,
    )
    absolute_error = calculate_absolute_error(synthetic_payoff, theoretical_payoff)

    info = format_portfolio_summary(
        put_count=put_count,
        call_count=call_count,
        asset_quantity=asset_quantity,
        fixed_component=fixed_component,
        initial_price=initial_price,
    )

    return ReplicationResult(
        expiry_grid=expiry_grid,
        asset_quantity=asset_quantity,
        fixed_component=fixed_component,
        synthetic_payoff=synthetic_payoff,
        theoretical_payoff=theoretical_payoff,
        absolute_error=absolute_error,
        info=info,
        put_strikes=put_strikes,
        put_weights=put_weights,
        call_strikes=call_strikes,
        call_weights=call_weights,
    )


def run_numeric_replication_pipeline(
    strike_grid: FloatArray,
    payoff_values: FloatArray,
    initial_price: float,
    first_derivative: FloatArray | None = None,
    second_derivative: FloatArray | None = None,
) -> ReplicationResult:
    """Replicate a sampled payoff using a uniform strike grid.

    Missing first and second derivatives are estimated with NumPy gradients.
    The initial price is inserted into the grid when it is not already present.
    """
    strike_grid = np.asarray(strike_grid, dtype=float)
    payoff_values = np.asarray(payoff_values, dtype=float)

    if strike_grid.ndim != 1 or payoff_values.ndim != 1:
        raise ValueError("The grid and payoff must be one-dimensional arrays.")
    if strike_grid.size < 3 or strike_grid.shape != payoff_values.shape:
        raise ValueError("The curve must contain at least three matching points.")
    if not np.all(np.isfinite(strike_grid)) or not np.all(
        np.isfinite(payoff_values)
    ):
        raise ValueError("The curve contains non-finite values.")
    if np.any(np.diff(strike_grid) <= 0):
        raise ValueError("Underlying prices must be in ascending order.")
    if not np.isfinite(initial_price) or not (
        strike_grid[0] <= initial_price <= strike_grid[-1]
    ):
        raise ValueError("The initial price must be within the drawn range.")

    if first_derivative is None:
        first_derivative = np.gradient(payoff_values, strike_grid, edge_order=2)
    else:
        first_derivative = np.asarray(first_derivative, dtype=float)
    if second_derivative is None:
        second_derivative = np.gradient(
            first_derivative, strike_grid, edge_order=2
        )
    else:
        second_derivative = np.asarray(second_derivative, dtype=float)

    if first_derivative.shape != strike_grid.shape or (
        second_derivative.shape != strike_grid.shape
    ):
        raise ValueError("The derivatives must have the same size as the grid.")
    if not np.all(np.isfinite(first_derivative)) or not np.all(
        np.isfinite(second_derivative)
    ):
        raise ValueError("The derivatives contain non-finite values.")

    if not np.any(strike_grid == initial_price):
        spot_index = int(np.searchsorted(strike_grid, initial_price))
        spot_payoff = np.interp(initial_price, strike_grid, payoff_values)
        spot_delta = np.interp(
            initial_price, strike_grid, first_derivative
        )
        spot_gamma = np.interp(
            initial_price, strike_grid, second_derivative
        )
        strike_grid = np.insert(strike_grid, spot_index, initial_price)
        payoff_values = np.insert(
            payoff_values, spot_index, spot_payoff
        )
        first_derivative = np.insert(
            first_derivative, spot_index, spot_delta
        )
        second_derivative = np.insert(
            second_derivative, spot_index, spot_gamma
        )

    option_strikes = (strike_grid[:-1] + strike_grid[1:]) / 2
    strike_spacing = np.diff(strike_grid)
    option_weights = np.interp(
        option_strikes, strike_grid, second_derivative
    ) * strike_spacing
    put_mask = option_strikes < initial_price
    call_mask = ~put_mask

    put_strikes = option_strikes[put_mask]
    put_weights = option_weights[put_mask]
    call_strikes = option_strikes[call_mask]
    call_weights = option_weights[call_mask]

    asset_quantity = float(
        np.interp(initial_price, strike_grid, first_derivative)
    )
    payoff_at_spot = float(np.interp(initial_price, strike_grid, payoff_values))
    fixed_component = payoff_at_spot - asset_quantity * initial_price
    expiry_grid = strike_grid.copy()

    synthetic_payoff, theoretical_payoff = calculate_payoffs(
        expiry_grid=expiry_grid,
        initial_price=initial_price,
        put_strikes=put_strikes,
        put_weights=put_weights,
        call_strikes=call_strikes,
        call_weights=call_weights,
        fixed_component=fixed_component,
        payoff_function=lambda values: np.interp(
            values, strike_grid, payoff_values
        ),
        first_derivative_function=lambda _: asset_quantity,
    )
    absolute_error = calculate_absolute_error(synthetic_payoff, theoretical_payoff)
    info = format_portfolio_summary(
        put_count=int(np.count_nonzero(put_weights)),
        call_count=int(np.count_nonzero(call_weights)),
        asset_quantity=asset_quantity,
        fixed_component=fixed_component,
        initial_price=initial_price,
    )

    return ReplicationResult(
        expiry_grid=expiry_grid,
        asset_quantity=asset_quantity,
        fixed_component=fixed_component,
        synthetic_payoff=synthetic_payoff,
        theoretical_payoff=theoretical_payoff,
        absolute_error=absolute_error,
        info=info,
        put_strikes=put_strikes,
        put_weights=put_weights,
        call_strikes=call_strikes,
        call_weights=call_weights,
    )


def calculate_strike_quadrature_widths(strikes: FloatArray) -> FloatArray:
    """Calculate integration widths around ordered, irregular option strikes.

    Each interior width spans the midpoints to its neighboring strikes.
    The first and last widths extend by one full adjacent strike interval.
    """
    strikes = np.asarray(strikes, dtype=float)
    if strikes.ndim != 1 or strikes.size < 2:
        raise ValueError("At least two option strikes are required.")
    if not np.all(np.isfinite(strikes)) or np.any(np.diff(strikes) <= 0):
        raise ValueError("Option strikes must be finite and strictly increasing.")

    widths = np.empty_like(strikes)
    widths[0] = strikes[1] - strikes[0]
    widths[-1] = strikes[-1] - strikes[-2]
    if strikes.size > 2:
        widths[1:-1] = (strikes[2:] - strikes[:-2]) / 2
    return widths


def select_eligible_option_strikes(
    option_strikes: FloatArray,
    option_volumes: FloatArray,
    minimum_volume: float,
    minimum_price: float,
    maximum_price: float,
    initial_price: float,
    is_put: bool,
) -> FloatArray:
    """Select liquid, in-range out-of-the-money strikes for one option type."""
    option_strikes = np.asarray(option_strikes, dtype=float)
    option_volumes = np.asarray(option_volumes, dtype=float)
    if option_strikes.ndim != 1 or option_volumes.ndim != 1:
        raise ValueError("Option strikes and volumes must be one-dimensional.")
    if option_strikes.shape != option_volumes.shape:
        raise ValueError("Each option strike must have a corresponding volume.")
    if not np.isfinite(minimum_volume) or minimum_volume < 0:
        raise ValueError(
            "Minimum option volume must be a finite non-negative number."
        )
    if not (
        np.isfinite(minimum_price)
        and np.isfinite(maximum_price)
        and minimum_price < maximum_price
    ):
        raise ValueError("The option price range must be finite and increasing.")

    out_of_the_money = (
        option_strikes < initial_price
        if is_put
        else option_strikes > initial_price
    )
    eligible = (
        np.isfinite(option_strikes)
        & np.isfinite(option_volumes)
        & (option_volumes >= minimum_volume)
        & (option_strikes >= minimum_price)
        & (option_strikes <= maximum_price)
        & out_of_the_money
    )
    return np.unique(option_strikes[eligible])


def run_market_replication_pipeline(
    strike_grid: FloatArray,
    payoff_values: FloatArray,
    initial_price: float,
    put_strikes: FloatArray,
    put_volumes: FloatArray,
    call_strikes: FloatArray,
    call_volumes: FloatArray,
    minimum_volume: float,
    first_derivative: FloatArray | None = None,
    second_derivative: FloatArray | None = None,
) -> ReplicationResult:
    """Replicate a sampled payoff using volume-qualified listed options.

    Options are restricted to out-of-the-money puts and calls within the payoff
    grid's range. A contract qualifies when its volume is at least
    ``minimum_volume``; missing or non-finite volumes are excluded. Curvature is
    interpolated at each selected strike and multiplied by its irregular-grid
    quadrature width.

    Raises:
        ValueError: If the curve, derivatives, option arrays, or volume threshold
            are invalid, or fewer than two qualifying strikes remain.
    """
    strike_grid = np.asarray(strike_grid, dtype=float)
    payoff_values = np.asarray(payoff_values, dtype=float)
    if strike_grid.ndim != 1 or payoff_values.ndim != 1:
        raise ValueError("The grid and payoff must be one-dimensional arrays.")
    if strike_grid.size < 3 or strike_grid.shape != payoff_values.shape:
        raise ValueError("The curve must contain at least three matching points.")
    if not np.all(np.isfinite(strike_grid)) or not np.all(
        np.isfinite(payoff_values)
    ):
        raise ValueError("The curve contains non-finite values.")
    if np.any(np.diff(strike_grid) <= 0):
        raise ValueError("Underlying prices must be in ascending order.")
    if not np.isfinite(initial_price) or not (
        strike_grid[0] <= initial_price <= strike_grid[-1]
    ):
        raise ValueError("The initial price must be within the drawn range.")
    if not np.isfinite(minimum_volume) or minimum_volume < 0:
        raise ValueError(
            "Minimum option volume must be a finite non-negative number."
        )

    if first_derivative is None:
        first_derivative = np.gradient(payoff_values, strike_grid, edge_order=2)
    else:
        first_derivative = np.asarray(first_derivative, dtype=float)
    if second_derivative is None:
        second_derivative = np.gradient(
            first_derivative, strike_grid, edge_order=2
        )
    else:
        second_derivative = np.asarray(second_derivative, dtype=float)
    if first_derivative.shape != strike_grid.shape or (
        second_derivative.shape != strike_grid.shape
    ):
        raise ValueError("The derivatives must have the same size as the grid.")
    if not np.all(np.isfinite(first_derivative)) or not np.all(
        np.isfinite(second_derivative)
    ):
        raise ValueError("The derivatives contain non-finite values.")

    if not np.any(strike_grid == initial_price):
        spot_index = int(np.searchsorted(strike_grid, initial_price))
        payoff_at_spot = np.interp(initial_price, strike_grid, payoff_values)
        first_derivative_at_spot = np.interp(
            initial_price, strike_grid, first_derivative
        )
        second_derivative_at_spot = np.interp(
            initial_price, strike_grid, second_derivative
        )
        strike_grid = np.insert(strike_grid, spot_index, initial_price)
        payoff_values = np.insert(
            payoff_values, spot_index, payoff_at_spot
        )
        first_derivative = np.insert(
            first_derivative, spot_index, first_derivative_at_spot
        )
        second_derivative = np.insert(
            second_derivative, spot_index, second_derivative_at_spot
        )

    selected_put_strikes = select_eligible_option_strikes(
        put_strikes,
        put_volumes,
        minimum_volume,
        float(strike_grid[0]),
        float(strike_grid[-1]),
        initial_price,
        is_put=True,
    )
    selected_call_strikes = select_eligible_option_strikes(
        call_strikes,
        call_volumes,
        minimum_volume,
        float(strike_grid[0]),
        float(strike_grid[-1]),
        initial_price,
        is_put=False,
    )
    all_option_strikes = np.sort(
        np.concatenate((selected_put_strikes, selected_call_strikes))
    )
    if all_option_strikes.size < 2:
        raise ValueError(
            "At least two in-range out-of-the-money options must meet the minimum volume."
        )

    option_widths = calculate_strike_quadrature_widths(all_option_strikes)
    # Integrate the payoff curvature over each strike's midpoint-bounded interval.
    option_gammas = np.interp(
        all_option_strikes, strike_grid, second_derivative
    )
    option_weights = option_gammas * option_widths
    put_mask = all_option_strikes < initial_price
    call_mask = all_option_strikes > initial_price
    selected_put_strikes = all_option_strikes[put_mask]
    selected_call_strikes = all_option_strikes[call_mask]
    selected_put_weights = option_weights[put_mask]
    selected_call_weights = option_weights[call_mask]

    asset_quantity = float(
        np.interp(initial_price, strike_grid, first_derivative)
    )
    payoff_at_spot = float(
        np.interp(initial_price, strike_grid, payoff_values)
    )
    fixed_component = payoff_at_spot - asset_quantity * initial_price
    expiry_grid = strike_grid.copy()
    payoff_function = lambda values: np.interp(
        values, strike_grid, payoff_values
    )
    derivative_function = lambda _: asset_quantity
    synthetic_payoff, theoretical_payoff = calculate_payoffs(
        expiry_grid=expiry_grid,
        initial_price=initial_price,
        put_strikes=selected_put_strikes,
        put_weights=selected_put_weights,
        call_strikes=selected_call_strikes,
        call_weights=selected_call_weights,
        fixed_component=fixed_component,
        payoff_function=payoff_function,
        first_derivative_function=derivative_function,
    )
    absolute_error = calculate_absolute_error(synthetic_payoff, theoretical_payoff)
    info = format_portfolio_summary(
        put_count=int(np.count_nonzero(selected_put_weights)),
        call_count=int(np.count_nonzero(selected_call_weights)),
        asset_quantity=asset_quantity,
        fixed_component=fixed_component,
        initial_price=initial_price,
    )

    return ReplicationResult(
        expiry_grid=expiry_grid,
        asset_quantity=asset_quantity,
        fixed_component=fixed_component,
        synthetic_payoff=synthetic_payoff,
        theoretical_payoff=theoretical_payoff,
        absolute_error=absolute_error,
        info=info,
        put_strikes=selected_put_strikes,
        put_weights=selected_put_weights,
        call_strikes=selected_call_strikes,
        call_weights=selected_call_weights,
    )


def plot_payoffs(
    expiry_grid: FloatArray,
    synthetic_payoff: FloatArray,
    theoretical_payoff: FloatArray,
) -> None:
    """Plot replicated and theoretical payoffs on the same axes."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        expiry_grid,
        synthetic_payoff,
        label="Replicated payoff",
        color="blue",
    )
    plt.plot(
        expiry_grid,
        theoretical_payoff,
        label="Theoretical payoff",
        color="orange",
        linestyle="--",
    )
    plt.title("Replicated vs. theoretical payoff (Carr-Madan)")
    plt.xlabel("Underlying price at expiration (S_T)")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    S = sp.symbols("S")
    strike_grid = np.linspace(1, 100, 500)
    initial_price = 50.0
    limit = 20
    k = 0.25

    # Smooth bull spread payoff
    smooth_bull_spread_payoff = limit / (1 + sp.exp(-k * (S - initial_price)))  # Example of a smooth function

    result = run_replication_pipeline(
        strike_symbol=S,
        symbolic_payoff=smooth_bull_spread_payoff,
        strike_grid=strike_grid,
        initial_price=initial_price,
    )

    plot_payoffs(
        result.expiry_grid,
        result.synthetic_payoff,
        result.theoretical_payoff,
    )

    # Print the summary stored in the info attribute
    print(result.info)
