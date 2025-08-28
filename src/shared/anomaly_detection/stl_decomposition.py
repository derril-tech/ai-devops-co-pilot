"""
STL (Seasonal and Trend decomposition using Loess) decomposition for time series analysis
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class STLResult:
    """Result of STL decomposition"""
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    original: np.ndarray
    timestamps: List[datetime]

    @property
    def seasonal_adjusted(self) -> np.ndarray:
        """Get seasonally adjusted series (original - seasonal)"""
        return self.original - self.seasonal

    @property
    def trend_adjusted(self) -> np.ndarray:
        """Get trend adjusted series (original - trend)"""
        return self.original - self.trend


class STLDecomposer:
    """STL decomposition implementation"""

    def __init__(self, seasonal_period: int = 24, robust: bool = False):
        """
        Initialize STL decomposer

        Args:
            seasonal_period: Number of observations per season (e.g., 24 for hourly data)
            robust: Whether to use robust fitting for trend and seasonal components
        """
        self.seasonal_period = seasonal_period
        self.robust = robust

    def decompose(self, values: List[float], timestamps: List[datetime]) -> STLResult:
        """
        Perform STL decomposition on time series

        Args:
            values: Time series values
            timestamps: Corresponding timestamps

        Returns:
            STLResult with trend, seasonal, and residual components
        """
        if len(values) < 2 * self.seasonal_period:
            raise ValueError(f"Time series too short. Need at least {2 * self.seasonal_period} observations")

        values_array = np.array(values)

        # Step 1: Detrend the series using a low-pass filter
        trend = self._extract_trend(values_array)

        # Step 2: Extract seasonal component
        detrended = values_array - trend
        seasonal = self._extract_seasonal(detrended)

        # Step 3: Calculate residual
        residual = values_array - trend - seasonal

        return STLResult(
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            original=values_array,
            timestamps=timestamps
        )

    def _extract_trend(self, values: np.ndarray) -> np.ndarray:
        """Extract trend component using LOESS smoothing"""
        n = len(values)
        trend = np.zeros(n)

        # Use a simple moving average for trend extraction
        # In a full implementation, this would use LOESS regression
        window_size = max(3, n // 10)  # Adaptive window size

        for i in range(n):
            start = max(0, i - window_size // 2)
            end = min(n, i + window_size // 2 + 1)
            trend[i] = np.mean(values[start:end])

        return trend

    def _extract_seasonal(self, detrended: np.ndarray) -> np.ndarray:
        """Extract seasonal component"""
        n = len(detrended)
        seasonal = np.zeros(n)

        # Calculate seasonal averages
        seasonal_sums = np.zeros(self.seasonal_period)
        seasonal_counts = np.zeros(self.seasonal_period)

        for i in range(n):
            period_idx = i % self.seasonal_period
            seasonal_sums[period_idx] += detrended[i]
            seasonal_counts[period_idx] += 1

        # Calculate seasonal averages
        seasonal_averages = seasonal_sums / seasonal_counts

        # Handle missing data in seasonal periods
        overall_mean = np.mean(detrended)
        seasonal_averages = np.where(np.isnan(seasonal_averages), overall_mean, seasonal_averages)

        # Apply seasonal component to entire series
        for i in range(n):
            period_idx = i % self.seasonal_period
            seasonal[i] = seasonal_averages[period_idx]

        return seasonal

    def forecast_seasonal(self, values: List[float], timestamps: List[datetime],
                         forecast_steps: int) -> Tuple[np.ndarray, List[datetime]]:
        """
        Forecast seasonal component for future periods

        Args:
            values: Historical values
            timestamps: Historical timestamps
            forecast_steps: Number of steps to forecast

        Returns:
            Tuple of (seasonal_forecast, forecast_timestamps)
        """
        # Decompose historical data
        decomposition = self.decompose(values, timestamps)

        # Use the last seasonal period as forecast
        seasonal_pattern = decomposition.seasonal[-self.seasonal_period:]

        # Repeat pattern for forecast period
        seasonal_forecast = np.tile(seasonal_pattern, forecast_steps // self.seasonal_period + 1)
        seasonal_forecast = seasonal_forecast[:forecast_steps]

        # Generate forecast timestamps
        last_timestamp = timestamps[-1]
        forecast_timestamps = [
            last_timestamp + timedelta(seconds=i * self._infer_frequency_seconds(timestamps))
            for i in range(1, forecast_steps + 1)
        ]

        return seasonal_forecast, forecast_timestamps

    def _infer_frequency_seconds(self, timestamps: List[datetime]) -> int:
        """Infer the frequency of the time series in seconds"""
        if len(timestamps) < 2:
            return 60  # Default to 1 minute

        # Calculate average time difference
        diffs = []
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            diffs.append(diff)

        return int(np.median(diffs))


class SeasonalBaseline:
    """Seasonal baseline calculation for anomaly detection"""

    def __init__(self, seasonal_period: int = 24, baseline_window_days: int = 7):
        """
        Initialize seasonal baseline

        Args:
            seasonal_period: Number of observations per season
            baseline_window_days: Number of days to use for baseline calculation
        """
        self.seasonal_period = seasonal_period
        self.baseline_window_days = baseline_window_days
        self.decomposer = STLDecomposer(seasonal_period)

    def calculate_baseline(self, values: List[float], timestamps: List[datetime]) -> Dict[str, Any]:
        """
        Calculate seasonal baseline for anomaly detection

        Args:
            values: Time series values
            timestamps: Corresponding timestamps

        Returns:
            Dictionary containing baseline statistics
        """
        if len(values) < self.seasonal_period:
            # Fallback to simple statistics
            return self._calculate_simple_baseline(values)

        try:
            # Perform STL decomposition
            decomposition = self.decomposer.decompose(values, timestamps)

            # Calculate baseline statistics from residual component
            residual_mean = np.mean(decomposition.residual)
            residual_std = np.std(decomposition.residual)

            # Calculate seasonal component statistics
            seasonal_mean = np.mean(decomposition.seasonal)
            seasonal_std = np.std(decomposition.seasonal)

            return {
                "method": "stl",
                "seasonal_period": self.seasonal_period,
                "residual_mean": residual_mean,
                "residual_std": residual_std,
                "seasonal_mean": seasonal_mean,
                "seasonal_std": seasonal_std,
                "trend_slope": self._calculate_trend_slope(decomposition.trend, timestamps),
                "decomposition": {
                    "trend": decomposition.trend.tolist(),
                    "seasonal": decomposition.seasonal.tolist(),
                    "residual": decomposition.residual.tolist()
                }
            }

        except Exception as e:
            logger.warning(f"STL decomposition failed, falling back to simple baseline: {e}")
            return self._calculate_simple_baseline(values)

    def _calculate_simple_baseline(self, values: List[float]) -> Dict[str, Any]:
        """Calculate simple baseline statistics"""
        values_array = np.array(values)

        return {
            "method": "simple",
            "mean": np.mean(values_array),
            "std": np.std(values_array),
            "median": np.median(values_array),
            "q25": np.percentile(values_array, 25),
            "q75": np.percentile(values_array, 75),
            "min": np.min(values_array),
            "max": np.max(values_array)
        }

    def _calculate_trend_slope(self, trend: np.ndarray, timestamps: List[datetime]) -> float:
        """Calculate the slope of the trend component"""
        if len(trend) < 2 or len(timestamps) < 2:
            return 0.0

        # Convert timestamps to seconds since epoch
        times = np.array([(ts - timestamps[0]).total_seconds() for ts in timestamps])

        # Calculate linear regression slope
        n = len(times)
        sum_x = np.sum(times)
        sum_y = np.sum(trend)
        sum_xy = np.sum(times * trend)
        sum_xx = np.sum(times * times)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)

        return slope

    def detect_anomalies(self, current_value: float, baseline: Dict[str, Any],
                        timestamps: List[datetime]) -> Dict[str, Any]:
        """
        Detect if a value is anomalous compared to baseline

        Args:
            current_value: The value to check
            baseline: Baseline statistics
            timestamps: Recent timestamps for context

        Returns:
            Dictionary with anomaly detection results
        """
        if baseline["method"] == "stl":
            return self._detect_anomalies_stl(current_value, baseline, timestamps)
        else:
            return self._detect_anomalies_simple(current_value, baseline)

    def _detect_anomalies_stl(self, current_value: float, baseline: Dict[str, Any],
                             timestamps: List[datetime]) -> Dict[str, Any]:
        """Detect anomalies using STL baseline"""
        residual_mean = baseline["residual_mean"]
        residual_std = baseline["residual_std"]

        # Calculate residual for current value (simplified)
        # In practice, this would require the current seasonal component
        residual = current_value - residual_mean

        # Calculate z-score
        z_score = abs(residual - residual_mean) / residual_std if residual_std > 0 else 0

        # Determine if anomalous (using 3-sigma rule)
        is_anomalous = z_score > 3.0

        return {
            "is_anomalous": is_anomalous,
            "z_score": z_score,
            "residual": residual,
            "threshold_upper": residual_mean + 3 * residual_std,
            "threshold_lower": residual_mean - 3 * residual_std,
            "method": "stl"
        }

    def _detect_anomalies_simple(self, current_value: float, baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies using simple baseline"""
        mean = baseline["mean"]
        std = baseline["std"]

        # Calculate z-score
        z_score = abs(current_value - mean) / std if std > 0 else 0

        # Determine if anomalous (using 3-sigma rule)
        is_anomalous = z_score > 3.0

        return {
            "is_anomalous": is_anomalous,
            "z_score": z_score,
            "deviation": current_value - mean,
            "threshold_upper": mean + 3 * std,
            "threshold_lower": mean - 3 * std,
            "method": "simple"
        }
