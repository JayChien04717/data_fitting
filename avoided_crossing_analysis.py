# -*- coding: utf-8 -*-
"""Utilities for analyzing avoided level crossings.

This module defines :class:`AvoidedCrossingAnalysis` which loads two-dimensional
spectroscopy data, extracts resonance branches and fits them using ``lmfit`` to
the :func:`avoided_crossing_direct_coupling` model with linear flux
dependence.
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
                                     f_center1: float,
                                     f_center2: float,
                                     c1: float,
                                     c2: float,
                                     g: float,
                                     flux_state: int | np.ndarray | None = None,
                                     ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Direct coupling model with linear flux dependence.

    Parameters
    ----------
    flux : array_like
        Flux values.
    f_center1 : float
        Intercept of the first branch.
    f_center2 : float
        Intercept of the second branch.
    c1 : float
        Slope of the first branch.
    c2 : float
        Slope of the second branch.
    g : float
        Coupling strength.
    flux_state : int or array_like, optional
        If given, select which eigenvalue to return for each flux value.

    Returns
    -------
    tuple of ndarray or ndarray
        Upper and lower branch frequencies, or the selected branch if
        ``flux_state`` is provided.
    """
    flux = np.asarray(flux)
    if flux_state is not None and isinstance(flux_state, int):
        flux_state = [flux_state] * len(flux)

    frequencies = np.zeros((len(flux), 2))
    for kk, dac in enumerate(flux):
        f_1 = dac * c1 + f_center1
        f_2 = dac * c2 + f_center2
        matrix = [[f_1, g], [g, f_2]]
        frequencies[kk, :] = np.linalg.eigvalsh(matrix)

    f_minus = frequencies[:, 0]
    f_plus = frequencies[:, 1]

    if flux_state is None:
        return f_plus, f_minus
    flux_state = np.asarray(flux_state)
    return np.where(flux_state, f_minus, f_plus)


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
            f_center1: float,
            f_center2: float,
            c1: float,
            c2: float,
            g: float) -> None:
        """Fit the extracted peaks using ``lmfit``."""
        if self.upper is None or self.lower is None:
            raise RuntimeError("run find_raw_peaks first")
        x = np.concatenate([self.flux, self.flux])
        y = np.concatenate([self.upper, self.lower])

        def model(flux_all, f_center1, f_center2, c1, c2, g):
            n = len(flux_all) // 2
            flux = flux_all[:n]
            up, low = avoided_crossing_direct_coupling(flux, f_center1, f_center2, c1, c2, g)
            return np.concatenate([up, low])

        mod = Model(model)
        params = Parameters()
        params.add("f_center1", value=f_center1)
        params.add("f_center2", value=f_center2)
        params.add("c1", value=c1)
        params.add("c2", value=c2)
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
            up, low = avoided_crossing_direct_coupling(
                self.flux,
                self.result.params["f_center1"].value,
                self.result.params["f_center2"].value,
                self.result.params["c1"].value,
                self.result.params["c2"].value,
                self.result.params["g"].value,
            )
            plt.plot(self.flux, up, "C1--", label="fit upper")
            plt.plot(self.flux, low, "C2--", label="fit lower")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    def estimate_initial_params(self) -> dict[str, float]:
        """Estimate initial fit parameters from extracted peaks."""
        if self.upper is None or self.lower is None:
            raise RuntimeError("run find_raw_peaks first")

        idx = int(np.nanargmin(np.abs(self.upper - self.lower)))
        g = 0.5 * np.abs(self.upper[idx] - self.lower[idx])

        mask_up = ~np.isnan(self.upper)
        mask_low = ~np.isnan(self.lower)

        c1, f_center1 = np.polyfit(self.flux[mask_up], self.upper[mask_up], 1)
        c2, f_center2 = np.polyfit(self.flux[mask_low], self.lower[mask_low], 1)

        return {
            "f_center1": float(f_center1),
            "f_center2": float(f_center2),
            "c1": float(c1),
            "c2": float(c2),
            "g": float(g),
        }

    # ------------------------------------------------------------------
    def autofit(self) -> None:
        """Automatically estimate parameters and run :meth:`fit`."""
        params = self.estimate_initial_params()
        self.fit(**params)
