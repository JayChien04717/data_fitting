# -*- coding: utf-8 -*-
"""Utilities for analyzing avoided level crossings.

This module defines :class:`AvoidedCrossingAnalysis` which loads two-dimensional
spectroscopy data, extracts resonance branches and fits them using
``lmfit`` to the ``avoided_crossing_direct_coupling`` model.
"""

from __future__ import annotations

import numpy as np
import scipy.signal
from lmfit import Model, Parameters
import matplotlib.pyplot as plt
try:
    import h5py
except Exception:  # pragma: no cover - optional dependency
    h5py = None


def avoided_crossing_direct_coupling(flux: np.ndarray,
                                     f_q0: float,
                                     f_q_amp: float,
                                     period: float,
                                     f_r: float,
                                     g: float) -> tuple[np.ndarray, np.ndarray]:
    """Simple direct coupling model.

    Parameters
    ----------
    flux : array_like
        Flux values.
    f_q0 : float
        Qubit frequency at zero flux.
    f_q_amp : float
        Amplitude of the flux dependence of the qubit frequency.
    period : float
        Flux period (typically ``Phi0``).
    f_r : float
        Bare resonator frequency.
    g : float
        Coupling strength.

    Returns
    -------
    tuple of ndarray
        Upper and lower branch frequencies.
    """
    flux = np.asarray(flux)
    f_q = f_q0 + f_q_amp * np.cos(2 * np.pi * flux / period)
    delta = f_q - f_r
    split = np.sqrt(delta ** 2 + 4 * g ** 2)
    f_plus = (f_q + f_r + split) / 2
    f_minus = (f_q + f_r - split) / 2
    return f_plus, f_minus


class AvoidedCrossingAnalysis:
    """Analyze and fit avoided crossing data."""

    def __init__(self, flux: np.ndarray, frequency: np.ndarray, data: np.ndarray):
        self.flux = np.asarray(flux)
        self.frequency = np.asarray(frequency)
        self.data = np.asarray(data)

        self.upper: np.ndarray | None = None
        self.lower: np.ndarray | None = None
        self.result = None

    # ------------------------------------------------------------------
    @classmethod
    def from_hdf5(cls, filename: str,
                  flux_dataset: str,
                  freq_dataset: str,
                  data_dataset: str) -> "AvoidedCrossingAnalysis":
        """Load flux, frequency and data arrays from a HDF5 file."""
        if h5py is None:
            raise ImportError("h5py is required to load HDF5 files")
        with h5py.File(filename, "r") as f:
            flux = f[flux_dataset][...]
            freq = f[freq_dataset][...]
            data = f[data_dataset][...]
        return cls(flux, freq, data)

    # ------------------------------------------------------------------
    def find_raw_peaks(self, prominence: float = 0.1) -> None:
        """Locate peaks for each flux value using ``scipy.signal.find_peaks``."""
        peaks_upper = []
        peaks_lower = []
        for i, row in enumerate(self.data):
            # choose two most prominent peaks; fall back to minima if necessary
            y = row
            peaks, props = scipy.signal.find_peaks(y, prominence=prominence)
            if len(peaks) < 2:
                peaks, props = scipy.signal.find_peaks(-y, prominence=prominence)
            if len(peaks) >= 2:
                idx = np.argsort(props["prominences"])[-2:]
                freq_peaks = self.frequency[peaks[idx]]
            elif len(peaks) == 1:
                freq_peaks = [self.frequency[peaks[0]], np.nan]
            else:
                freq_peaks = [np.nan, np.nan]
            freq_peaks = sorted(freq_peaks, reverse=True)
            peaks_upper.append(freq_peaks[0])
            peaks_lower.append(freq_peaks[1])
        self.upper = np.array(peaks_upper)
        self.lower = np.array(peaks_lower)

    # ------------------------------------------------------------------
    def clean_peaks(self) -> None:
        """Remove invalid peak entries and ensure branch ordering."""
        if self.upper is None or self.lower is None:
            raise RuntimeError("run find_raw_peaks first")
        mask = ~np.isnan(self.upper) & ~np.isnan(self.lower)
        self.flux = self.flux[mask]
        u = self.upper[mask]
        l = self.lower[mask]
        swap = u < l
        u[swap], l[swap] = l[swap], u[swap]
        self.upper = u
        self.lower = l

    # ------------------------------------------------------------------
    def fit(self,
            f_q0: float,
            f_q_amp: float,
            period: float,
            f_r: float,
            g: float) -> None:
        """Fit the extracted peaks using ``lmfit``."""
        if self.upper is None or self.lower is None:
            raise RuntimeError("run find_raw_peaks first")
        x = np.concatenate([self.flux, self.flux])
        y = np.concatenate([self.upper, self.lower])

        def model(flux_all, f_q0, f_q_amp, period, f_r, g):
            n = len(flux_all) // 2
            flux = flux_all[:n]
            up, low = avoided_crossing_direct_coupling(flux, f_q0, f_q_amp, period, f_r, g)
            return np.concatenate([up, low])

        mod = Model(model)
        params = Parameters()
        params.add("f_q0", value=f_q0)
        params.add("f_q_amp", value=f_q_amp)
        params.add("period", value=period)
        params.add("f_r", value=f_r)
        params.add("g", value=g, min=0)
        self.result = mod.fit(y, params, flux_all=x)

    # ------------------------------------------------------------------
    def plot(self, show_data: bool = True) -> None:
        """Plot extracted peaks and fit result."""
        if show_data:
            plt.pcolormesh(self.flux, self.frequency, self.data.T, shading="auto")
            plt.xlabel("Flux")
            plt.ylabel("Frequency")
            plt.colorbar(label="Amplitude")
        if self.upper is not None and self.lower is not None:
            plt.scatter(self.flux, self.upper, color="C1", s=10, label="upper")
            plt.scatter(self.flux, self.lower, color="C2", s=10, label="lower")
        if self.result is not None:
            up, low = avoided_crossing_direct_coupling(self.flux,
                                                       self.result.params["f_q0"].value,
                                                       self.result.params["f_q_amp"].value,
                                                       self.result.params["period"].value,
                                                       self.result.params["f_r"].value,
                                                       self.result.params["g"].value)
            plt.plot(self.flux, up, "C1--", label="fit upper")
            plt.plot(self.flux, low, "C2--", label="fit lower")
        plt.legend()
        plt.tight_layout()
        plt.show()
