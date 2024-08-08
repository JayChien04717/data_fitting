import cmath
from typing import Dict, Optional

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import minimize_scalar

from fitting import *

figsize = (8, 6)


def NormalizeData(zval: float) -> float:
    """Normalize function to normalize data

    Parameters
    ----------
    zval : float
        Input data

    Returns
    -------
    float
        Normalize data
    """
    return (zval - np.min(zval)) / (np.max(zval) - np.min(zval))


def post_rotate(ydata):
    def rotate_complex(iq, angle):
        return (iq) * np.exp(1j * np.pi * angle / 180)

    def std_q(y, rot_agl_):
        iq = rotate_complex(y, rot_agl_)
        return np.std(iq.imag)

    res = minimize_scalar(lambda agl: std_q(ydata, agl), bounds=[0, 360])
    rotation_angle = res.x
    ydata = rotate_complex(ydata, rotation_angle)
    return ydata


def rsquare(y: float, ybar: float) -> float:
    """Calculate the coefficient of determination

    Parameters
    ----------
    y : float
        measure data
    ybar : float
        fitting data

    Returns
    -------
    float
        return the r square coefficient
    """
    ss_res = np.sum((y - ybar) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_square = 1 - ss_res / ss_tot
    return r_square


def resonator_analyze(x: float, y: float) -> Optional[Dict]:
    """Hanger geometry resonator circult fit
    This fitting tool is from https://github.com/sebastianprobst/resonator_tools
    Only for notch type hanger resonator

    Parameters
    ----------
    x : float
        frequency list/array
    y : float
        S21 data

    Returns
    -------
    Optional[Dict]
        fitting result, data contain Qc, Qi, Ql .ect.
    """
    from resonator_tools import circuit

    port1 = circuit.notch_port()
    port1.add_data(x, y)
    port1.autofit()
    port1.plotall()
    return port1.fitresults


def resonator_analyze2(x, y, fit: bool = True):
    """Fitting function using asymmetric lorentzian function

    Parameters
    ----------
    x : _type_
        frequncy list/array
    y : _type_
        S21 data
    fit : bool, optional
        If fitting is true, it will plot the fitting result. Or only plot input data

    Returns
    -------
    float :
        return resonatnce frequency(GHz)
    """
    y = np.abs(y)
    pOpt, pCov = fit_asym_lor(x, y)
    res = pOpt[2] / 1e9

    plt.figure(figsize=figsize)
    plt.title(f"resonator spectrum.", fontsize=15)
    plt.plot(x, y, label="mag", marker="o", markersize=3)
    if fit == True:
        plt.plot(x, asym_lorfunc(x, *pOpt),
                 label=f"fit, $\kappa$={pOpt[3]/1e6:.3f}MHz")
    plt.axvline(res, color="r", ls="--", label=f"$f_res$ = {res/1e9:.2f} GHz")
    plt.legend()
    plt.show()
    return round(res, 2)


def spectrum_analyze(x: float, y: float, fit: bool = True) -> float:
    """Analyze the spectrum
    This function is using lorentzian funtion

    Parameters
    ----------
    x : float
        frequency list/array
    y : float
        S21 data
    fit : bool, optional
        if fit is true, plot the fitting result, by default True

    Returns
    -------
    float
        return resonance frequency(GHz)
    """
    y = np.abs(y)
    pOpt, pCov = fitlor(x, y)
    res = pOpt[2] / 1e9

    plt.figure(figsize=figsize)
    plt.title(f"mag.", fontsize=15)
    plt.plot(x, y, label="mag", marker="o", markersize=3)
    if fit == True:
        plt.plot(x, lorfunc(x, *pOpt), label="fit")
        plt.axvspan(
            res - pOpt[3], res + pOpt[3], alpha=0.5, label=f"lw = {2*pOpt[3]/1e6}MHz"
        )
        plt.legend()
    plt.axvline(res, color="r", ls="--", label=f"$f_res$ = {res/1e6:.2f}")
    plt.legend()
    plt.show()
    return round(res, 3)


def dispersive_analyze(x: float, y1: float, y2: float, fit: bool = True):
    """Plot the dispersive shift and maximum shift frequency

    Parameters
    ----------
    x : float
        frequency
    y1 : float
        ground/excited state spectrum
    y2 : float
        ground/excited state spectrum
    fit : bool, optional
        If fit is true, plot the fitting data, by default True
    """
    y1 = np.abs(y1)
    y2 = np.abs(y2)
    pOpt1, pCov1 = fit_asym_lor(x, y1)
    res1 = pOpt1[2]
    pOpt2, pCov2 = fit_asym_lor(x, y2)
    res2 = pOpt2[2]

    plt.figure(figsize=figsize)
    plt.title(f"$\chi=${(res2/1e6-res1/1e6):.3f}, unit = MHz", fontsize=15)
    plt.plot(x, y1, color="r", label="e", marker="s", markersize=5)
    plt.plot(x, y2, color="g", label="g", marker="s", markersize=5)
    if fit == True:
        plt.plot(
            x, asym_lorfunc(x, *pOpt1), label=f"fite, $\kappa$ = {pOpt1[3]/1e6:.2f}MHz"
        )
        plt.plot(
            x, asym_lorfunc(x, *pOpt2), label=f"fitg, $\kappa$ = {pOpt2[3]/1e6:.2f}MHz"
        )
    plt.axvline(res1, color="r", ls="--", label=f"$f_res$ = {res1:.2f}")
    plt.axvline(res2, color="g", ls="--", label=f"$f_res$ = {res2:.2f}")
    plt.legend()
    plt.figure(figsize=figsize)
    plt.plot(x, y1 - y2)
    plt.axvline(
        x[np.argmax(y1 - y2)],
        color="r",
        ls="--",
        label=f"max SNR1 = {x[np.argmax(y1-y2)]:.2f}",
    )
    plt.axvline(
        x[np.argmin(y1 - y2)],
        color="g",
        ls="--",
        label=f"max SNR2 = {x[np.argmin(y1-y2)]:.2f}",
    )
    plt.legend()
    plt.show()


def amprabi_analyze(x: int, y: float, fit: bool = True, normalize: bool = False):
    """Analyze and fit the amplitude Rabi

    Parameters
    ----------
    x : int
        gain/power
    y : float
        rabi data
    fit : bool, optional
        If fit is true, plot the fitting data, by default True
    normalize : bool, optional
        If normalize is true, normalize the data, by default False

    Returns
    -------
    list
        return the pi pulse gain, pi/2 pulse gain and max value minus min value
    """
    y = np.abs(y)
    pOpt, pCov = fitdecaysin(x, y, [None, None, None, 1e5, None])
    sim = decaysin(x, *pOpt)

    pi = round(x[np.argmax(sim)], 1)
    pi2 = round(x[round((np.argmin(sim) + np.argmax(sim)) / 2)], 1)

    if pOpt[2] > 180:
        pOpt[2] = pOpt[2] - 360
    elif pOpt[2] < -180:
        pOpt[2] = pOpt[2] + 360
    if pOpt[2] < 0:
        pi_gain = (1 / 2 - pOpt[2] / 180) / 2 / pOpt[1]
        pi2_gain = (0 - pOpt[2] / 180) / 2 / pOpt[1]
    else:
        pi_gain = (3 / 2 - pOpt[2] / 180) / 2 / pOpt[1]
        pi2_gain = (1 - pOpt[2] / 180) / 2 / pOpt[1]

    plt.figure(figsize=figsize)
    plt.plot(x, y, label="meas", ls="-", marker="s", markersize=5)
    if fit == True:
        plt.plot(x, sim, label="fit")
    plt.title(f"Amplitude Rabi", fontsize=15)
    plt.xlabel("$gain$", fontsize=15)
    if normalize == True:
        plt.ylabel("Population", fontsize=15)
        plt.axvline(pi, ls="--", c="red", label=f"$\pi$ gain={pi}")
        plt.axvline(pi2, ls="--", c="red", label=f"$\pi/2$ gain={pi2}")
        plt.legend(loc=4)
        plt.tight_layout()
        plt.show()
        return round(pi, 1), round(pi2, 1), max(y) - min(y)
    else:
        plt.axvline(pi_gain, ls="--", c="red",
                    label=f"$\pi$ gain={pi_gain:.1f}")
        plt.axvline(pi2_gain, ls="--", c="red",
                    label=f"$\pi$ gain={(pi2_gain):.1f}")
        plt.legend(loc=4)
        plt.tight_layout()
        plt.show()
        return round(pi_gain, 1), round(pi2_gain, 1), max(y) - min(y)


def lengthrabi_analyze(x: float, y: float, fit: bool = True, normalize: bool = False):
    """Analyze and fit the length Rabi data

    Parameters
    ----------
    x : float
        Rabi pulse length
    y : float
        Rabi data
    fit : bool, optional
        If fit is true, plot the fitting data, by default True
    normalize : bool, optional
        If normalize is true, normalize the data, by default False

    Returns
    -------
    List
        return the pi pulse legnth, pi/2 pulse length and max value minus min value
    """
    y = np.abs(y)
    pOpt, pCov = fitdecaysin(x, y)
    sim = decaysin(x, *pOpt)

    pi = round(x[np.argmax(sim)], 1)
    pi2 = round(x[round((np.argmin(sim) + np.argmax(sim)) / 2)], 1)
    if pOpt[2] > 180:
        pOpt[2] = pOpt[2] - 360
    elif pOpt[2] < -180:
        p[2] = pOpt[2] + 360
    if pOpt[2] < 0:
        pi_length = (1 / 2 - pOpt[2] / 180) / 2 / pOpt[1]
        pi2_length = (0 - pOpt[2] / 180) / 2 / pOpt[1]
    else:
        pi_length = (3 / 2 - pOpt[2] / 180) / 2 / pOpt[1]
        pi2_length = (1 - pOpt[2] / 180) / 2 / pOpt[1]

    plt.figure(figsize=figsize)
    plt.plot(x, y, label="meas", ls="-", marker="s", markersize=5)
    if fit == True:
        plt.plot(x, sim, label="fit")
    plt.title(f"Length Rabi, Rabi frequency={pOpt[1]/1e6:.2f}MHz", fontsize=15)
    plt.xlabel("$t\ (us)$", fontsize=15)
    if normalize == True:
        plt.ylabel("Population", fontsize=15)
        plt.axvline(pi, ls="--", c="red", label=f"$\pi$ len={pi}")
        plt.axvline(pi2, ls="--", c="red", label=f"$\pi/2$ len={pi2}")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return pi, pi2
    else:
        plt.axvline(
            pi_length, ls="--", c="red", label=f"$\pi$ length={pi_length:.3f}$\mu$s"
        )
        plt.axvline(
            pi2_length, ls="--", c="red", label=f"$\pi$ length={pi2_length:.3f}$\mu$s"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()
        return pi_length, pi2_length


def T1_analyze(x: float, y: float, fit: bool = True, normalize: bool = False):
    """T1 relaxation analyze

    Parameters
    ----------
    x : float
        T1 program relax time
    y : float
        T1 data
    fit : bool, optional
        If fit is true, plot the fitting data, by default True
    normalize : bool, optional
        If normalize is true, normalize the data, by default False
    """
    y = np.abs(y)
    pOpt, pCov = fitexp(x, y)
    sim = expfunc(x, *pOpt)
    r_square = rsquare(y, sim)

    plt.figure(figsize=figsize)
    plt.plot(x, y, label="meas", ls="-", marker="s", markersize=5)
    if fit == True:
        plt.plot(x, sim, label=f"fit, R square = {r_square:.2f}")
    plt.title(f"T1 = {pOpt[3]/1e6:.2f}$\mu s$", fontsize=15)
    plt.xlabel("$t\ (\mu s)$", fontsize=15)
    if normalize == True:
        plt.ylabel("Population", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()


def T2r_analyze(x: float, y: float, fit: bool = True, normalize: bool = False):
    """T2 ramsey analyze

    Parameters
    ----------
    x : float
        T2 ramsey program time
    y : float
        T2 data
    fit : bool, optional
        If fit is true, plot the fitting data, by default True
    normalize : bool, optional
        If normalize is true, normalize the data, by default False

    Returns
    -------
    float
        Detuning frequency
    """
    y = np.abs(y)
    pOpt, pCov = fitdecaysinmod(x, y)
    sim = decaysinmod(x, *pOpt)
    error = np.sqrt(np.diag(pCov))
    r_square = rsquare(y, sim)

    plt.figure(figsize=figsize)
    plt.plot(x, y, label="meas", ls="-", marker="s", markersize=5)
    if fit == True:
        plt.plot(x, sim, label=f"fit, R square = {r_square:.2f}")
    plt.title(
        f"T2r = {pOpt[3]*1e6:.2f}$\mu s, detune = {pOpt[1]/1e6:.2f}MHz \pm {(error[1])/1e3:.2f}kHz$",
        fontsize=15,
    )
    plt.xlabel("$t\ (\mu s)$", fontsize=15)
    if normalize == True:
        plt.ylabel("Population", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return pOpt[1]


def T2e_analyze(x: float, y: float, fit: bool = True, normalize: bool = False):
    """_summary_

    Parameters
    ----------
    x : float
        T2 echo program time
    y : float
        T2 echo data
    fit : bool, optional
        If fit is true, plot the fitting data, by default True
    normalize : bool, optional
        If normalize is true, normalize the data, by default False

    """
    y = np.abs(y)
    pOpt, pCov = fitexp(x, y)
    sim = expfunc(x, *pOpt)
    r_square = rsquare(y, sim)

    plt.figure(figsize=figsize)
    plt.plot(x, y, label="meas", ls="-", marker="s", markersize=5)
    if fit == True:
        plt.plot(x, sim, label=f"fit, R square = {r_square:.2f}")
    plt.title(f"T2e = {pOpt[3]:.2f}$\mu s$", fontsize=15)
    plt.xlabel("$t\ (\mu s)$", fontsize=15)
    if normalize == True:
        plt.ylabel("Population", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()


# load data
def load_data(file_path: str):
    with h5.File(file_path, "r") as f:
        data: np.ndarray = f["Data"]["Data"][:]  # type: ignore

    X0 = data[..., 2, 0]
    Y0 = data[..., 3, 0]
    X1 = data[..., 2, 1]
    Y1 = data[..., 3, 1]

    return X0, Y0, X1, Y1


# calculate splitting threshold
BIN_NUM = 50


def calculate_histogram(X0, X1):
    bins = np.linspace(min(min(X0), min(X1)), max(max(X0), max(X1)), BIN_NUM)
    dbin = bins[1] - bins[0]
    hist0, _ = np.histogram(X0, bins=bins, density=True)
    hist1, _ = np.histogram(X1, bins=bins, density=True)

    return bins, hist0 * dbin, hist1 * dbin


def calculate_threshold(bins, hist0, hist1):
    contrast = np.abs(
        (np.cumsum(hist0) - np.cumsum(hist1)) /
        (0.5 * hist0.sum() + 0.5 * hist1.sum())
    )

    tind = np.argmax(contrast)
    threshold = (bins[tind] + bins[tind + 1]) / 2
    fid = contrast[tind]

    return fid, threshold


def plot_histogram2(bins, hist0, hist1, param0, param1, threshold):
    _, ax = plt.subplots()

    # plot histogram and threshold
    ax.bar(bins[:-1], hist0, width=bins[1] -
           bins[0], alpha=0.5, label="State 0")
    ax.bar(bins[:-1], hist1, width=bins[1] -
           bins[0], alpha=0.5, label="State 1")
    ax.axvline(threshold, color="black", linestyle="--", label="Threshold")

    # plot fitting curve
    def Gaussian1D(x, xo, sigma):
        amplitude = 1 / (sigma * np.sqrt(2 * np.pi))
        g = amplitude * np.exp(-((x - xo) ** 2) / (2 * sigma**2))

        return g

    def DualGaussian1D(x, xo0, sigma0, xo1, sigma1, a_ratio):
        A, B = a_ratio, 1 - a_ratio
        return A * Gaussian1D(x, xo0, sigma0) + B * Gaussian1D(x, xo1, sigma1)

    x = np.linspace(min(bins), max(bins), BIN_NUM)

    fit0 = DualGaussian1D(x, param0[0], param0[2], param0[3], param0[5], param0[6]) * (
        x[1] - x[0]
    )
    fit1 = DualGaussian1D(x, param1[0], param1[2], param1[3], param1[5], param1[6]) * (
        x[1] - x[0]
    )

    ax.plot(x, fit0, color="blue", label="Fit 0")
    ax.plot(x, fit1, color="red", label="Fit 1")

    ax.set_xlabel("X")
    ax.set_ylabel("Density")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    # import sys
    # import os

    # os.chdir(r"C:\Users\QEL\Desktop\Jay PhD\Code\data_fitting\data")
    # sys.path.append(r"C:\Program Files\Keysight\Labber\Script")
    # import Labber

    # np.bool = bool
    # np.float = np.float64
    # res = r"Normalized_coupler_chen_054[7]_@7.819GHz_power_dp_002.hdf5"
    # pdr = r"r1_pdr.hdf5"
    # lenghrabi = r"q1_rabi.hdf5"
    # t1 = r"q1_T1_2.hdf5"
    # t2 = r"q2_T2Ramsey.hdf5"
    # spec = r"q1_twotone_4.hdf5"

    # spec = Labber.LogFile(spec)
    # pdr = Labber.LogFile(pdr)
    # fr = Labber.LogFile(res)
    # f1 = Labber.LogFile(lenghrabi)
    # f2 = Labber.LogFile(t1)
    # f3 = Labber.LogFile(t2)
    # (x, y) = fr.getTraceXY(entry=3)
    # (sx, sy) = spec.getTraceXY()
    # (rx, ry) = f1.getTraceXY()
    # (t1x, t1y) = f2.getTraceXY()
    # (t2x, t2y) = f3.getTraceXY()
    # # spectrum_analyze(sx, sy, plot=True)
    # # lengthraig_analyze(rx, ry, plot=True, normalize=False)
    # # amprabi_analyze(rx, ry, fit=True, normalize=True)
    # # T1_analyze(t1x, t1y, fit=True,normalize=False)
    # T2r_analyze(t2x, t2y, fit=True, normalize=False)
    # # T2r_analyze(t1x, t1y, fit=True, normalize=False)
    # # resonator_analyze(x,y)

    X0, Y0, X1, Y1 = load_data("./data/q1 single shot 30000 gain 0.8us.hdf5")
    # X0, Y0, X1, Y1 = load_data("./data/q1 single shot 30000 gain 1.5us.hdf5")
    # X0, Y0, X1, Y1 = load_data("./data/q1 single shot 30000 gain2.hdf5")

    D0 = calculate_density(X0, Y0)
    D1 = calculate_density(X1, Y1)

    param0 = fit_data(X0, Y0, D0)
    param1 = fit_data(X1, Y1, D1)
    print(f"Fit 0: {param0}")
    print(f"Fit 1: {param1}")

    angle, rX0, rY0, rparam0, rX1, rY1, rparam1 = rotate_data(
        X0, Y0, param0, X1, Y1, param1
    )
    bins, hist0, hist1 = calculate_histogram(rX0, rX1)
    fidelity, threshold = calculate_threshold(bins, hist0, hist1)
    plot_histogram2(bins, hist0, hist1, rparam0, rparam1, threshold)
    print(f"Angle: {angle: .2f} (rad)")
    print(f"Threshold: {threshold: .2f}")
    print(f"Fidelity: {fidelity: .3f}")
