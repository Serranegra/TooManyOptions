"""Interface for drawing a payoff and obtaining its static replication."""

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from .carrmadan import (
    ReplicationResult,
    run_market_replication_pipeline,
    run_numeric_replication_pipeline,
    select_eligible_option_strikes,
)
from .loader import (
    EUROPEAN_INDEX_TICKERS,
    create_option_dataframe,
    get_ticker_market_data,
)


class SmoothPayoffDrawer:
    """Draw a payoff curve and compare continuous-grid or listed-option replicas."""

    def __init__(
        self,
        width: int = 800,
        height: int = 500,
        *,
        root: tk.Tk | None = None,
    ) -> None:
        """Create the drawing canvas and payoff-replication controls."""
        self.width = width
        self.height = height
        self.x_left = 65
        self.x_right = width - 25
        self.y_top = 25
        self.y_bottom = height - 35
        self.y_zero = (self.y_top + self.y_bottom) / 2
        self.max_payoff = 100.0
        self.min_price = 0.0
        self.max_price = 200.0
        self.raw_points: list[tuple[int, int]] = []
        self.result_window: tk.Toplevel | None = None
        self._market_option_chain: tuple[
            str,
            str,
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ] | None = None

        self.root = root or tk.Tk()
        self.root.title("TooManyOptions — Static Payoff Replication")
        self.root.minsize(width + 40, height + 190)

        self.canvas = tk.Canvas(
            self.root,
            width=width,
            height=height,
            bg="#fafafa",
            highlightthickness=0,
        )
        self.canvas.pack(padx=12, pady=(12, 4))
        self.canvas.bind("<B1-Motion>", self.add_point)

        controls = ttk.Frame(self.root, padding=(12, 8))
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Initial underlying price (S₀):").pack(
            side=tk.LEFT
        )
        self.spot_entry = ttk.Entry(controls, width=10)
        self.spot_entry.insert(0, "100")
        self.spot_entry.pack(side=tk.LEFT, padx=(6, 14))
        self.spot_entry.bind("<KeyRelease>", self._on_spot_changed)
        self._draw_axes()
        self._draw_spot_marker()
        ttk.Button(controls, text="Clear", command=self.clear).pack(
            side=tk.LEFT
        )
        ttk.Button(
            controls,
            text="Calculate static replication",
            command=self.process_curve,
        ).pack(side=tk.RIGHT)

        market_controls = ttk.LabelFrame(
            self.root, text="Real options", padding=(12, 8)
        )
        market_controls.pack(fill=tk.X, padx=12, pady=(0, 6))
        ttk.Label(market_controls, text="Index ticker:").pack(side=tk.LEFT)
        self.ticker_entry = ttk.Combobox(
            market_controls,
            width=10,
            state="readonly",
            values=EUROPEAN_INDEX_TICKERS,
        )
        self.ticker_entry.pack(side=tk.LEFT, padx=(4, 8))
        self.ticker_entry.bind(
            "<<ComboboxSelected>>", lambda _: self._clear_ticker_data()
        )
        ttk.Button(
            market_controls,
            text="Load ticker data",
            command=self.load_expirations,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(market_controls, text="Expiration:").pack(side=tk.LEFT)
        self.expiration_choice = ttk.Combobox(
            market_controls, width=13, state="readonly"
        )
        self.expiration_choice.pack(side=tk.LEFT, padx=(4, 12))
        self.expiration_choice.bind(
            "<<ComboboxSelected>>",
            lambda _: self._load_selected_option_chain(),
        )
        ttk.Label(market_controls, text="Min. daily volume (contracts):").pack(
            side=tk.LEFT
        )
        self.minimum_volume_entry = ttk.Entry(market_controls, width=8)
        self.minimum_volume_entry.insert(0, "1")
        self.minimum_volume_entry.pack(side=tk.LEFT, padx=(4, 8))
        self.minimum_volume_entry.bind(
            "<KeyRelease>", lambda _: self._update_strike_range_preview()
        )
        ttk.Button(
            market_controls,
            text="Replicate with real options",
            command=self.process_real_options,
        ).pack(side=tk.RIGHT)
        ttk.Label(
            self.root,
            text="Volume means contracts traded today; it is not a dollar amount or open interest.",
            padding=(12, 0, 12, 6),
        ).pack(fill=tk.X)

        self.status = ttk.Label(
            self.root,
            text=self._drawing_instructions(),
            padding=(12, 0, 12, 10),
        )
        self.status.pack(fill=tk.X)

    def _draw_axes(self) -> None:
        """Draw chart axes, grid lines, labels, and the current spot marker."""
        self.canvas.delete("axes")
        self.canvas.create_line(
            self.x_left,
            self.y_top,
            self.x_left,
            self.y_bottom,
            fill="#9e9e9e",
            width=1,
            tags="axes",
        )
        self.canvas.create_line(
            self.x_left,
            self.y_zero,
            self.x_right,
            self.y_zero,
            fill="#9e9e9e",
            width=1,
            tags="axes",
        )

        for price in np.linspace(self.min_price, self.max_price, 5):
            x = self.x_left + (price - self.min_price) / (
                self.max_price - self.min_price
            ) * (
                self.x_right - self.x_left
            )
            self.canvas.create_line(
                x,
                self.y_top,
                x,
                self.y_bottom,
                fill="#e0e0e0",
                dash=(2, 4),
                tags="axes",
            )
            self.canvas.create_text(
                x,
                self.y_bottom + 15,
                text=f"{price:g}",
                fill="#616161",
                tags="axes",
            )

        for payoff in (-100, -50, 0, 50, 100):
            y = self.y_zero - payoff / self.max_payoff * (
                self.y_zero - self.y_top
            )
            self.canvas.create_line(
                self.x_left,
                y,
                self.x_right,
                y,
                fill="#e0e0e0" if payoff else "#9e9e9e",
                dash=(2, 4) if payoff else (),
                tags="axes",
            )
            self.canvas.create_text(
                self.x_left - 28,
                y,
                text=str(payoff),
                fill="#616161",
                tags="axes",
            )

        self.canvas.create_text(
            (self.x_left + self.x_right) / 2,
            self.height - 8,
            text="Underlying price at expiration (Sₜ)",
            fill="#424242",
            tags="axes",
        )
        self.canvas.create_text(
            14,
            (self.y_top + self.y_bottom) / 2,
            text="Payoff",
            angle=90,
            fill="#424242",
            tags="axes",
        )
        self._draw_spot_marker()

    def _draw_spot_marker(self) -> None:
        """Update the vertical marker showing the entered initial price."""
        self.canvas.delete("spot")
        try:
            spot = float(self.spot_entry.get())
        except ValueError:
            return
        if self.min_price <= spot <= self.max_price:
            x = self.x_left + (spot - self.min_price) / (
                self.max_price - self.min_price
            ) * (
                self.x_right - self.x_left
            )  # Map the spot price to its relative position on the chart.
            self.canvas.create_line(
                x,
                self.y_top,
                x,
                self.y_bottom,
                fill="#ef6c00",
                dash=(5, 4),
                tags="spot",
            )
            self.canvas.create_text(
                x + 4,
                self.y_top + 8,
                text="S₀",
                anchor=tk.W,
                fill="#ef6c00",
                tags="spot",
            )

    def _on_spot_changed(self, _: tk.Event) -> None:
        """Refresh the spot marker and option-strike preview after edits."""
        self._draw_spot_marker()
        self._update_strike_range_preview()

    def _draw_strike_range(self, minimum_strike: float, maximum_strike: float) -> None:
        """Mark the lowest and highest strikes used by market replication."""
        self.canvas.delete("strike_bounds")
        for strike, label, y_offset, anchor in (
            (minimum_strike, "Eligible min strike", 8, tk.NW),
            (maximum_strike, "Eligible max strike", 24, tk.NE),
        ):
            x = self.x_left + (strike - self.min_price) / (
                self.max_price - self.min_price
            ) * (self.x_right - self.x_left)
            self.canvas.create_line(
                x,
                self.y_top,
                x,
                self.y_bottom,
                fill="#7b1fa2",
                dash=(5, 3),
                width=2,
                tags="strike_bounds",
            )
            self.canvas.create_text(
                x,
                self.y_top + y_offset,
                text=f"{label}: {strike:g}",
                anchor=anchor,
                fill="#7b1fa2",
                tags="strike_bounds",
            )

    def add_point(self, event: tk.Event) -> None:
        """Add a mouse-drawn point while keeping the curve left-to-right."""
        x = min(max(event.x, self.x_left), self.x_right)
        y = min(max(event.y, self.y_top), self.y_bottom)
        if self.raw_points and x <= self.raw_points[-1][0]:
            return

        self.raw_points.append((x, y))
        if len(self.raw_points) > 1:
            x0, y0 = self.raw_points[-2]
            self.canvas.create_line(
                x0, y0, x, y, fill="#1565c0", width=2.5, tags="curve"
            )

    def clear(self) -> None:
        """Remove the drawn payoff curve and restore the chart instructions."""
        self.raw_points.clear()
        self.canvas.delete("curve")
        self.canvas.delete("strike_bounds")
        self._draw_axes()
        self.status.configure(text=self._drawing_instructions())

    def _drawing_instructions(self) -> str:
        """Describe the underlying-price and payoff ranges shown on the chart."""
        return (
            f"Draw from left to right. The chart represents S from "
            f"{self.min_price:g} to {self.max_price:g} and payoff from "
            "-100 to 100."
        )

    def _prepare_curve(
        self,
    ) -> tuple[
        float,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ] | None:
        """Convert the drawing to prices and payoffs and estimate its derivatives.

        Gaussian and Savitzky-Golay filters smooth the hand-drawn samples before
        the payoff curve and its first two derivatives are returned.
        """
        if len(self.raw_points) < 15:
            messagebox.showwarning(
                "Incomplete drawing",
                "Draw a curve with at least 15 points, from left to right.",
                parent=self.root,
            )
            return None

        try:
            initial_price = float(self.spot_entry.get())
        except ValueError:
            messagebox.showerror(
                "Invalid initial price",
                "Enter a valid number for the initial price S₀.",
                parent=self.root,
            )
            return None

        points = np.asarray(self.raw_points, dtype=float)
        x_raw, y_raw = points[:, 0], points[:, 1]
        raw_prices = self.min_price + (x_raw - self.x_left) / (
            self.x_right - self.x_left
        ) * (self.max_price - self.min_price)
        raw_payoffs = (self.y_zero - y_raw) / (self.y_zero - self.y_top) * (
            self.max_payoff
        )

        price_grid = np.linspace(raw_prices[0], raw_prices[-1], 1000)
        pre_smoothed_payoffs = gaussian_filter1d(raw_payoffs, sigma=2.5)
        interpolated_payoffs = np.interp(price_grid, raw_prices, pre_smoothed_payoffs)
        window_size = min(51, len(price_grid) if len(price_grid) % 2 else len(price_grid) - 1)
        price_step = float(price_grid[1] - price_grid[0])
        smoothed_payoffs = savgol_filter(
            interpolated_payoffs, window_length=window_size, polyorder=3
        )
        first_derivative = savgol_filter(
            interpolated_payoffs,
            window_length=window_size,
            polyorder=3,
            deriv=1,
            delta=price_step,
        )
        second_derivative = savgol_filter(
            interpolated_payoffs,
            window_length=window_size,
            polyorder=3,
            deriv=2,
            delta=price_step,
        )

        return (
            initial_price,
            raw_prices,
            raw_payoffs,
            price_grid,
            smoothed_payoffs,
            first_derivative,
            second_derivative,
        )

    def process_curve(self) -> None:
        """Replicate the drawing using the default uniform strike grid."""
        curve = self._prepare_curve()
        if curve is None:
            return
        (
            initial_price,
            raw_prices,
            raw_payoffs,
            price_grid,
            smoothed_payoffs,
            first_derivative,
            second_derivative,
        ) = curve

        try:
            result = run_numeric_replication_pipeline(
                strike_grid=price_grid,
                payoff_values=smoothed_payoffs,
                initial_price=initial_price,
                first_derivative=first_derivative,
                second_derivative=second_derivative,
            )
        except ValueError as error:
            messagebox.showerror(
                "Could not calculate replication",
                str(error),
                parent=self.root,
            )
            return

        self._draw_spot_marker()
        self.status.configure(
            text=(
                f"Replication calculated for S₀ = {initial_price:g}. "
                "Quantities are per payoff unit and exclude prices and costs."
            )
        )
        self._show_result(result, raw_prices, raw_payoffs, price_grid, smoothed_payoffs)

    def load_expirations(self) -> None:
        """Load ticker expirations and center the chart on its latest price."""
        ticker_symbol = self.ticker_entry.get().strip().upper()
        if not ticker_symbol:
            messagebox.showerror(
                "Ticker required",
                "Select a supported index ticker before loading expirations.",
                parent=self.root,
            )
            return

        try:
            expirations, current_price = get_ticker_market_data(ticker_symbol)
        except Exception as error:
            messagebox.showerror(
                "Could not load ticker data",
                str(error),
                parent=self.root,
            )
            return

        self.expiration_choice["values"] = expirations
        if expirations:
            self.expiration_choice.current(0)
        else:
            self.expiration_choice.set("")
            messagebox.showwarning(
                "No options available",
                f"No option expirations were found for {ticker_symbol}.",
                parent=self.root,
            )

        self._center_price_range(current_price)
        self.spot_entry.delete(0, tk.END)
        self.spot_entry.insert(0, f"{current_price:g}")
        self._draw_axes()
        self.status.configure(
            text=(
                f"Latest available {ticker_symbol} price: {current_price:g}. "
                "The price axis is centered on the latest available price. "
                "Redraw the payoff on the updated chart."
            )
        )
        if expirations:
            self._load_selected_option_chain()

    def _center_price_range(self, current_price: float) -> None:
        """Center the chart on a positive price with a 50% range on each side."""
        half_range = current_price * 0.5
        self.min_price = current_price - half_range
        self.max_price = current_price + half_range
        self.raw_points.clear()
        self.canvas.delete("curve")
        self.canvas.delete("strike_bounds")

    def _clear_ticker_data(self) -> None:
        """Clear expirations when the ticker symbol is edited."""
        self.expiration_choice["values"] = ()
        self.expiration_choice.set("")
        self._market_option_chain = None
        self.canvas.delete("strike_bounds")

    def _load_selected_option_chain(self) -> None:
        """Load the selected chain and preview its eligible strike boundaries."""
        ticker_symbol = self.ticker_entry.get().strip().upper()
        expiration = self.expiration_choice.get().strip()
        if not ticker_symbol or not expiration:
            self._market_option_chain = None
            self.canvas.delete("strike_bounds")
            return

        try:
            calls, puts = create_option_dataframe(ticker_symbol, expiration)
            self._market_option_chain = (
                ticker_symbol,
                expiration,
                puts["strike"].to_numpy(dtype=float),
                puts["volume"].to_numpy(dtype=float, na_value=np.nan),
                calls["strike"].to_numpy(dtype=float),
                calls["volume"].to_numpy(dtype=float, na_value=np.nan),
            )
        except Exception as error:
            self._market_option_chain = None
            self.canvas.delete("strike_bounds")
            messagebox.showerror(
                "Could not load option chain",
                str(error),
                parent=self.root,
            )
            return

        self._update_strike_range_preview()

    def _update_strike_range_preview(self) -> None:
        """Show the strike limits implied by the current volume and price range."""
        chain = self._market_option_chain
        if chain is None:
            self.canvas.delete("strike_bounds")
            return

        ticker_symbol, expiration, put_strikes, put_volumes, call_strikes, call_volumes = chain
        try:
            minimum_volume = float(self.minimum_volume_entry.get())
            initial_price = float(self.spot_entry.get())
        except ValueError:
            self.canvas.delete("strike_bounds")
            return
        if (
            not np.isfinite(minimum_volume)
            or minimum_volume < 0
            or not np.isfinite(initial_price)
        ):
            self.canvas.delete("strike_bounds")
            return

        try:
            selected_puts = select_eligible_option_strikes(
                put_strikes,
                put_volumes,
                minimum_volume,
                self.min_price,
                self.max_price,
                initial_price,
                is_put=True,
            )
            selected_calls = select_eligible_option_strikes(
                call_strikes,
                call_volumes,
                minimum_volume,
                self.min_price,
                self.max_price,
                initial_price,
                is_put=False,
            )
        except ValueError as error:
            self.canvas.delete("strike_bounds")
            self.status.configure(text=f"Cannot preview strike limits: {error}")
            return

        selected_strikes = np.concatenate((selected_puts, selected_calls))
        if selected_strikes.size < 2:
            self.canvas.delete("strike_bounds")
            self.status.configure(
                text=(
                    f"Fewer than two eligible strikes for {ticker_symbol}, "
                    f"expiration {expiration}, at the current minimum volume. "
                    "Lower the volume threshold or adjust the payoff range."
                )
            )
            return

        minimum_strike = float(np.min(selected_strikes))
        maximum_strike = float(np.max(selected_strikes))
        self._draw_strike_range(minimum_strike, maximum_strike)
        self.status.configure(
            text=(
                f"Eligible strike range for {ticker_symbol}, expiration "
                f"{expiration} across the chart's full price range: "
                f"{minimum_strike:g} to {maximum_strike:g} "
                f"(minimum volume {minimum_volume:g} contracts)."
            )
        )

    def process_real_options(self) -> None:
        """Replicate the drawing with listed options meeting the volume filter."""
        curve = self._prepare_curve()
        if curve is None:
            return
        (
            initial_price,
            raw_prices,
            raw_payoffs,
            price_grid,
            smoothed_payoffs,
            first_derivative,
            second_derivative,
        ) = curve

        ticker_symbol = self.ticker_entry.get().strip().upper()
        expiration = self.expiration_choice.get().strip()
        if not ticker_symbol or not expiration:
            messagebox.showerror(
                "Ticker and expiration required",
                "Enter a ticker and load/select an expiration first.",
                parent=self.root,
            )
            return
        try:
            minimum_volume = float(self.minimum_volume_entry.get())
        except ValueError:
            messagebox.showerror(
                "Invalid minimum volume",
                "Enter a non-negative number for the minimum option volume.",
                parent=self.root,
            )
            return
        if not np.isfinite(minimum_volume) or minimum_volume < 0:
            messagebox.showerror(
                "Invalid minimum volume",
                "Enter a finite, non-negative number for the minimum option volume.",
                parent=self.root,
            )
            return

        chain = self._market_option_chain
        if chain is None or chain[:2] != (ticker_symbol, expiration):
            self._load_selected_option_chain()
            chain = self._market_option_chain
        if chain is None or chain[:2] != (ticker_symbol, expiration):
            return
        _, _, put_strikes, put_volumes, call_strikes, call_volumes = chain

        try:
            result = run_market_replication_pipeline(
                strike_grid=price_grid,
                payoff_values=smoothed_payoffs,
                initial_price=initial_price,
                put_strikes=put_strikes,
                put_volumes=put_volumes,
                call_strikes=call_strikes,
                call_volumes=call_volumes,
                minimum_volume=minimum_volume,
                first_derivative=first_derivative,
                second_derivative=second_derivative,
            )
        except ValueError as error:
            messagebox.showerror(
                "Could not calculate market replication",
                str(error),
                parent=self.root,
            )
            return

        selected_strikes = np.concatenate(
            (result.put_strikes, result.call_strikes)
        )
        minimum_strike = float(np.min(selected_strikes))
        maximum_strike = float(np.max(selected_strikes))
        self.status.configure(
            text=(
                f"Replication calculated for {ticker_symbol}, expiration "
                f"{expiration}, with minimum daily volume "
                f"{minimum_volume:g} contracts. Strikes used: "
                f"{minimum_strike:g} to {maximum_strike:g}. "
                "The chart markers continue to show all eligible strikes "
                "within the visible price range."
            )
        )
        self._show_result(result, raw_prices, raw_payoffs, price_grid, smoothed_payoffs)

    def _show_result(
        self,
        result: ReplicationResult,
        raw_prices: NDArray[np.float64],
        raw_payoffs: NDArray[np.float64],
        price_grid: NDArray[np.float64],
        smoothed_payoffs: NDArray[np.float64],
    ) -> None:
        """Show payoff/error charts and a table of replication positions."""
        if self.result_window is not None and self.result_window.winfo_exists():
            self.result_window.destroy()

        window = tk.Toplevel(self.root)
        self.result_window = window
        window.title("Static replication results")
        window.geometry("1120x790")
        window.minsize(850, 620)

        maximum_error = float(np.max(result.absolute_error))
        rmse = float(np.sqrt(np.mean(result.absolute_error**2)))
        summary = (
            f"Underlying: {result.asset_quantity:.6f} units    |    "
            f"Fixed cash flow at expiration: {result.fixed_component:.6f}    |    "
            f"Maximum error: {maximum_error:.6f}    |    RMSE: {rmse:.6f}"
        )
        ttk.Label(window, text=summary, padding=10).pack(fill=tk.X)
        ttk.Label(
            window,
            text=(
                "All quantities are per payoff unit. Fixed cash is a payoff "
                "flow at expiration: positive means receive cash; negative "
                "means pay cash. It is not the amount invested or borrowed at "
                "t=0; the initial funding amount is not calculated."
            ),
            padding=(10, 0, 10, 6),
        ).pack(fill=tk.X)

        figure, (payoff_axis, error_axis) = plt.subplots(
            1, 2, figsize=(11, 4.4), constrained_layout=True
        )
        payoff_axis.plot(
            raw_prices, raw_payoffs, color="#9e9e9e", linestyle=":", label="Original drawing"
        )
        payoff_axis.plot(
            price_grid, smoothed_payoffs, color="#ef6c00", linestyle="--", label="Smoothed payoff"
        )
        payoff_axis.plot(
            result.expiry_grid,
            result.synthetic_payoff,
            color="#1565c0",
            label="Replicated payoff",
        )
        payoff_axis.axhline(0, color="#757575", linewidth=0.7)
        payoff_axis.set_title("Theoretical and replicated payoff")
        payoff_axis.set_xlabel("Underlying price at expiration (Sₜ)")
        payoff_axis.set_ylabel("Payoff")
        payoff_axis.grid(True, alpha=0.25)
        payoff_axis.legend()

        error_axis.plot(
            result.expiry_grid,
            result.absolute_error,
            color="#c62828",
        )
        error_axis.set_title("Absolute replication error")
        error_axis.set_xlabel("Underlying price at expiration (Sₜ)")
        error_axis.set_ylabel("|replicated − theoretical|")
        error_axis.grid(True, alpha=0.25)

        chart = FigureCanvasTkAgg(figure, master=window)
        chart.draw()
        chart.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8)
        plt.close(figure)

        table_frame = ttk.LabelFrame(
            window,
            text="Expiration payoff components (per payoff unit)",
            padding=8,
        )
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 10))
        ttk.Label(
            table_frame,
            text=(
                "Fixed cash flow at expiration: positive = receive; negative = "
                "pay. This is a terminal payoff amount, not a t=0 borrowing or "
                "investment amount."
            ),
        ).pack(anchor=tk.W, pady=(0, 6))
        columns = ("instrument_type", "strike", "quantity")
        table = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=9
        )
        table.heading("instrument_type", text="Instrument")
        table.heading("strike", text="Strike")
        table.heading("quantity", text="Quantity")
        table.column("instrument_type", width=150, anchor=tk.W)
        table.column("strike", width=130, anchor=tk.E)
        table.column("quantity", width=180, anchor=tk.E)
        scrollbar = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=table.yview
        )
        table.configure(yscrollcommand=scrollbar.set)
        table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        table.insert(
            "", tk.END, values=("Underlying", "—", f"{result.asset_quantity:.8f}")
        )
        table.insert(
            "",
            tk.END,
            values=(
                "Fixed cash flow at expiration",
                "—",
                f"{result.fixed_component:.8f}",
            ),
        )
        weights = np.concatenate((result.put_weights, result.call_weights))
        scale = float(np.max(np.abs(weights))) if weights.size else 0.0
        display_threshold = max(1e-10, scale * 1e-5)
        for strike, quantity in zip(
            result.put_strikes, result.put_weights
        ):
            if abs(quantity) > display_threshold:
                table.insert(
                    "",
                    tk.END,
                    values=("Put", f"{strike:.4f}", f"{quantity:.8f}"),
                )
        for strike, quantity in zip(
            result.call_strikes, result.call_weights
        ):
            if abs(quantity) > display_threshold:
                table.insert(
                    "",
                    tk.END,
                    values=("Call", f"{strike:.4f}", f"{quantity:.8f}"),
                )

    def run(self) -> None:
        """Start the Tk event loop."""
        self.root.mainloop()


def main() -> None:
    """Show the introduction screen and launch the drawing interface."""
    root = tk.Tk()
    root.title("TooManyOptions")
    root.geometry("900x760")
    root.minsize(760, 650)

    welcome = ttk.Frame(root, padding=(32, 24))
    welcome.pack(fill=tk.BOTH, expand=True)

    ttk.Label(
        welcome,
        text="TooManyOptions",
        font=("TkDefaultFont", 24, "bold"),
    ).pack(anchor=tk.W)
    ttk.Label(
        welcome,
        text="Explore static replication of option payoffs",
        font=("TkDefaultFont", 13),
        padding=(0, 4, 0, 14),
    ).pack(anchor=tk.W)
    ttk.Label(
        welcome,
        text=(
            "Draw a target payoff and see how it can be approximated with the "
            "underlying asset, a fixed cash position, and options."
        ),
        wraplength=820,
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X, pady=(0, 14))

    sections = (
        (
            "How to use it",
            "1. Select Start drawing and draw the payoff from left to right. "
            "The horizontal axis is the underlying price at expiration; the "
            "vertical axis is the payoff.\n"
            "2. Set the initial underlying price S₀. For market data, select a "
            "supported index ticker and choose Load ticker data first. This "
            "updates S₀ and centers the chart around the latest available "
            "price; redraw the payoff after the chart range changes.\n"
            "3. Use Calculate static replication for a dense theoretical strike "
            "grid, or select an expiration and a minimum daily volume before "
            "choosing Replicate with real options.",
        ),
        (
            "Supported underlyings",
            "Real-option data is available only for these index tickers: "
            "^SPX (S&P 500), ^XSP (Mini S&P 500), ^NDX (Nasdaq-100), "
            "^NQX (Mini Nasdaq-100), ^RUT (Russell 2000), and "
            "^DJX (Dow Jones Industrial Average). Other stock, ETF, and "
            "index tickers cannot be loaded.",
        ),
        (
            "Data source",
            "Ticker prices and option chains are retrieved from Yahoo Finance "
            "through yfinance. The app uses the latest available quote, falling "
            "back to a recent daily close when needed. Option-chain volume is "
            "the number of contracts traded that day, not dollar volume or open "
            "interest. The selected expiration is loaded when you run the "
            "real-options replication.",
        ),
        (
            "How the replication works",
            "The drawn curve is smoothed, then its first and second derivatives "
            "are estimated numerically. The slope at S₀ determines the "
            "underlying position; the curve value and slope determine the fixed "
            "cash flow at expiration; curvature determines option weights. "
            "The fixed cash value is part of the terminal payoff and is not "
            "the initial amount to invest or borrow at t=0. For listed options, "
            "curvature is evaluated at eligible out-of-the-money strikes and "
            "integrated using midpoint intervals, with full neighbor intervals "
            "at the two outer strikes.",
        ),
        (
            "Assumptions and limitations",
            "The result is a static payoff approximation per payoff unit, not a "
            "trade recommendation or execution. It does not model option "
            "premiums, bid-ask spreads, fees, taxes, funding or borrowing rates, "
            "or discount the expiration cash flow to t=0. Therefore, the initial "
            "capital needed to establish the replication is not provided. It "
            "also omits early exercise, liquidity beyond the volume filter, and "
            "changes before expiration. The drawn payoff is smoothed, so results "
            "depend on the drawing and the available strikes.",
        ),
    )

    for title, description in sections:
        section = ttk.LabelFrame(welcome, text=title, padding=(12, 8))
        section.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            section,
            text=description,
            wraplength=800,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X)

    def start_drawing() -> None:
        welcome.destroy()
        SmoothPayoffDrawer(root=root)

    actions = ttk.Frame(welcome)
    actions.pack(fill=tk.X, pady=(2, 0))
    ttk.Button(actions, text="Quit", command=root.destroy).pack(side=tk.RIGHT)
    ttk.Button(
        actions, text="Start drawing", command=start_drawing
    ).pack(side=tk.RIGHT, padx=(0, 8))

    root.mainloop()


if __name__ == "__main__":
    main()
