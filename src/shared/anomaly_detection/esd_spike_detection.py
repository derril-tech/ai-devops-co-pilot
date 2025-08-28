"""
ESD (Extreme Studentized Deviate) spike detection algorithm
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpikeDetectionResult:
    """Result of spike detection"""
    is_spike: bool
    spike_score: float
    spike_magnitude: float
    confidence: float
    method: str
    details: Dict[str, Any]


class ESDSpikeDetector:
    """ESD (Extreme Studentized Deviate) spike detection"""

    def __init__(self, max_anomalies: int = 10, alpha: float = 0.05):
        """
        Initialize ESD spike detector

        Args:
            max_anomalies: Maximum number of anomalies to detect
            alpha: Significance level for statistical test
        """
        self.max_anomalies = max_anomalies
        self.alpha = alpha

    def detect_spikes(self, values: List[float], timestamps: Optional[List[datetime]] = None) -> List[SpikeDetectionResult]:
        """
        Detect spikes in time series using ESD test

        Args:
            values: Time series values
            timestamps: Optional timestamps for the values

        Returns:
            List of spike detection results
        """
        if len(values) < 10:
            logger.warning("Time series too short for reliable ESD test")
            return [self._create_no_spike_result() for _ in values]

        values_array = np.array(values)

        # Remove trend if present (simplified detrending)
        detrended_values = self._detrend_series(values_array)

        # Apply ESD test
        anomalies = self._esd_test(detrended_values)

        # Create results for all points
        results = []
        for i, value in enumerate(values):
            if i in anomalies:
                # This is an anomaly
                anomaly_info = anomalies[i]
                result = SpikeDetectionResult(
                    is_spike=True,
                    spike_score=anomaly_info['test_statistic'],
                    spike_magnitude=abs(value - np.mean(detrended_values)),
                    confidence=1 - anomaly_info['p_value'],
                    method="esd",
                    details={
                        "rank": anomaly_info['rank'],
                        "p_value": anomaly_info['p_value'],
                        "critical_value": anomaly_info['critical_value'],
                        "original_value": value,
                        "detrended_value": detrended_values[i]
                    }
                )
            else:
                # Not an anomaly
                result = SpikeDetectionResult(
                    is_spike=False,
                    spike_score=0.0,
                    spike_magnitude=0.0,
                    confidence=0.0,
                    method="esd",
                    details={}
                )
            results.append(result)

        return results

    def _detrend_series(self, values: np.ndarray) -> np.ndarray:
        """Remove trend from time series using simple moving average"""
        if len(values) < 5:
            return values

        # Use a simple moving average to estimate trend
        window_size = min(5, len(values) // 3)
        trend = np.convolve(values, np.ones(window_size)/window_size, mode='same')

        # Remove trend
        detrended = values - trend

        return detrended

    def _esd_test(self, values: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Perform ESD test for anomaly detection

        Returns:
            Dictionary mapping indices to anomaly information
        """
        anomalies = {}
        remaining_values = values.copy()
        original_indices = np.arange(len(values))

        for i in range(min(self.max_anomalies, len(values) // 2)):
            if len(remaining_values) < 3:
                break

            # Find the most extreme value
            mean = np.mean(remaining_values)
            std = np.std(remaining_values, ddof=1)

            if std == 0:
                break

            # Calculate absolute deviations
            deviations = np.abs(remaining_values - mean)

            # Find the index of the maximum deviation
            max_dev_idx = np.argmax(deviations)
            max_deviation = deviations[max_dev_idx]

            # Calculate test statistic (Extreme Studentized Deviate)
            test_statistic = max_deviation / std

            # Calculate critical value using Grubbs test
            n = len(remaining_values)
            t_value = stats.t.ppf(1 - self.alpha / (2 * n), n - 2)
            critical_value = ((n - 1) / np.sqrt(n)) * np.sqrt(t_value**2 / (n - 2 + t_value**2))

            # Calculate p-value (approximate)
            p_value = 1 - stats.t.cdf(test_statistic * np.sqrt((n - 2) / ((n - 1)**2 - test_statistic**2 * (n - 2))), n - 2)

            # Check if this is an anomaly
            if test_statistic > critical_value:
                # This is an anomaly
                original_idx = original_indices[max_dev_idx]

                anomalies[original_idx] = {
                    'test_statistic': test_statistic,
                    'p_value': p_value,
                    'critical_value': critical_value,
                    'rank': i + 1,
                    'deviation': max_deviation
                }

                # Remove this value and continue
                remaining_values = np.delete(remaining_values, max_dev_idx)
                original_indices = np.delete(original_indices, max_dev_idx)
            else:
                # No more anomalies
                break

        return anomalies

    def _create_no_spike_result(self) -> SpikeDetectionResult:
        """Create a result indicating no spike"""
        return SpikeDetectionResult(
            is_spike=False,
            spike_score=0.0,
            spike_magnitude=0.0,
            confidence=0.0,
            method="esd",
            details={}
        )

    def detect_single_value(self, current_value: float, baseline_values: List[float]) -> SpikeDetectionResult:
        """
        Detect if a single value is a spike compared to baseline

        Args:
            current_value: The value to test
            baseline_values: Baseline values for comparison

        Returns:
            Spike detection result
        """
        if len(baseline_values) < 5:
            return self._create_no_spike_result()

        # Combine baseline with current value
        all_values = baseline_values + [current_value]

        # Run ESD on combined series
        results = self.detect_spikes(all_values)

        # Return result for the last (current) value
        return results[-1]


class MultiResolutionSpikeDetector:
    """Multi-resolution spike detection using multiple window sizes"""

    def __init__(self, window_sizes: List[int] = None, max_anomalies: int = 5):
        """
        Initialize multi-resolution spike detector

        Args:
            window_sizes: List of window sizes to use for detection
            max_anomalies: Maximum anomalies per window
        """
        if window_sizes is None:
            window_sizes = [10, 20, 50, 100]

        self.window_sizes = window_sizes
        self.max_anomalies = max_anomalies
        self.detectors = [ESDSpikeDetector(max_anomalies) for _ in window_sizes]

    def detect_spikes_multiresolution(self, values: List[float]) -> List[Dict[str, Any]]:
        """
        Detect spikes using multiple resolution windows

        Args:
            values: Time series values

        Returns:
            List of detection results with consensus information
        """
        if len(values) < max(self.window_sizes):
            # Fallback to single resolution
            detector = ESDSpikeDetector(self.max_anomalies)
            results = detector.detect_spikes(values)
            return [
                {
                    "index": i,
                    "is_spike": result.is_spike,
                    "confidence": result.confidence,
                    "methods_used": ["esd"],
                    "consensus_score": 1.0 if result.is_spike else 0.0
                }
                for i, result in enumerate(results)
            ]

        # Run detection at multiple resolutions
        all_results = []
        for window_size, detector in zip(self.window_sizes, self.detectors):
            if len(values) >= window_size:
                # Use sliding window approach
                window_results = []
                for start_idx in range(len(values) - window_size + 1):
                    end_idx = start_idx + window_size
                    window_values = values[start_idx:end_idx]

                    window_detections = detector.detect_spikes(window_values)

                    # Map back to original indices
                    for i, detection in enumerate(window_detections):
                        original_idx = start_idx + i
                        if original_idx < len(all_results):
                            all_results[original_idx].append(detection.is_spike)
                        else:
                            all_results.append([detection.is_spike])

        # Aggregate results
        final_results = []
        for i, spike_flags in enumerate(all_results):
            if len(spike_flags) > 0:
                consensus_score = sum(spike_flags) / len(spike_flags)
                is_spike_consensus = consensus_score >= 0.5  # Majority vote

                final_results.append({
                    "index": i,
                    "is_spike": is_spike_consensus,
                    "confidence": consensus_score,
                    "methods_used": [f"esd_w{w}" for w in self.window_sizes],
                    "consensus_score": consensus_score
                })
            else:
                final_results.append({
                    "index": i,
                    "is_spike": False,
                    "confidence": 0.0,
                    "methods_used": [],
                    "consensus_score": 0.0
                })

        return final_results
