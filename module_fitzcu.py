import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cmath
from fitting import *
from typing import Optional, Dict
try:
    from resonator_tools import circuit
except:
    print("No circle fit package")
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
    from scipy.optimize import minimize_scalar

    def rotate_complex(iq, angle):
        return (iq) * np.exp(1j * np.pi * angle/180)

    def std_q(y, rot_agl_):
        iq = rotate_complex(y, rot_agl_)
        return np.std(iq.imag)
    res = minimize_scalar(lambda agl: std_q(ydata, agl), bounds=[0, 360])
    rotation_angle = res.x
    ydata = (rotate_complex(ydata, rotation_angle))
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
    ss_res = np.sum((y - ybar)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_square = 1 - ss_res/ss_tot
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
        return resonatnce frequency(MHz)
    """
    y = np.abs(y)
    pOpt, pCov = fit_asym_lor(x, y)
    res = pOpt[2]

    plt.figure(figsize=figsize)
    plt.title(f'mag.', fontsize=15)
    plt.plot(x, y, label='mag', marker='o', markersize=3)
    if fit == True:
        plt.plot(x, asym_lorfunc(x, *pOpt), label=f'fit, $\kappa$={pOpt[3]}')
    plt.axvline(res, color='r', ls='--', label=f'$f_res$ = {res:.2f}')
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
        return resonance frequency(MHz)
    """
    y = np.abs(y)
    pOpt, pCov = fitlor(x, y)
    res = pOpt[2]

    plt.figure(figsize=figsize)
    plt.title(f'mag.', fontsize=15)
    plt.plot(x, y, label='mag', marker='o', markersize=3)
    if fit == True:
        plt.plot(x, lorfunc(x, *pOpt), label='fit')
        plt.axvline(res, color='r', ls='--', label=f'$f_res$ = {res:.2f}')
    plt.legend()
    plt.show()
    return round(res, 2)


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
    plt.title(f'$\chi=${(res2-res1):.3f}, unit = MHz', fontsize=15)
    plt.plot(x, y1, label='e', marker='o', markersize=3)
    plt.plot(x, y2, label='g', marker='o', markersize=3)
    if fit == True:
        plt.plot(x, asym_lorfunc(x, *pOpt1),
                 label=f'fite, $\kappa$ = {pOpt1[3]:.2f}MHz')
        plt.plot(x, asym_lorfunc(x, *pOpt2),
                 label=f'fitg, $\kappa$ = {pOpt2[3]:.2f}MHz')
    plt.axvline(res1, color='r', ls='--', label=f'$f_res$ = {res1:.2f}')
    plt.axvline(res2, color='g', ls='--', label=f'$f_res$ = {res2:.2f}')
    plt.legend()
    plt.figure(figsize=figsize)
    plt.plot(x, y1-y2)
    plt.axvline(x[np.argmax(y1-y2)], color='r', ls='--',
                label=f'max SNR1 = {x[np.argmax(y1-y2)]:.2f}')
    plt.axvline(x[np.argmin(y1-y2)], color='g', ls='--',
                label=f'max SNR2 = {x[np.argmin(y1-y2)]:.2f}')
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
    pOpt, pCov = fitdecaysin(x, y)
    sim = decaysin(x, *pOpt)

    pi = round(x[np.argmax(sim)], 1)
    pi2 = round(x[round((np.argmin(sim) + np.argmax(sim))/2)], 1)

    if pOpt[2] > 180:
        pOpt[2] = pOpt[2] - 360
    elif pOpt[2] < -180:
        pOpt[2] = pOpt[2] + 360
    if pOpt[2] < 0:
        pi_gain = (1/2 - pOpt[2]/180)/2/pOpt[1]
        pi2_gain = (0 - pOpt[2]/180)/2/pOpt[1]
    else:
        pi_gain = (3/2 - pOpt[2]/180)/2/pOpt[1]
        pi2_gain = (1 - pOpt[2]/180)/2/pOpt[1]

    plt.figure(figsize=figsize)
    plt.plot(x, y, label='meas', ls='-', marker='o', markersize=3)
    if fit == True:
        plt.plot(x, sim, label='fit')
    plt.title(f'Amplitude Rabi', fontsize=15)
    plt.xlabel('$gain$', fontsize=15)
    if normalize == True:
        plt.ylabel('Population', fontsize=15)
        plt.axvline(pi, ls='--', c='red', label=f'$\pi$ gain={pi}')
        plt.axvline(pi2, ls='--', c='red', label=f'$\pi/2$ gain={pi2}')
        plt.legend(loc=4)
        plt.tight_layout()
        plt.show()
        return round(pi, 1), round(pi2, 1), max(y)-min(y)
    else:
        plt.axvline(pi_gain, ls='--', c='red',
                    label=f'$\pi$ gain={pi_gain:.1f}')
        plt.axvline(pi2_gain, ls='--', c='red',
                    label=f'$\pi$ gain={(pi2_gain):.1f}')
        plt.legend(loc=4)
        plt.tight_layout()
        plt.show()
        return round(pi_gain, 1), round(pi2_gain, 1), max(y)-min(y)


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
    pi2 = round(x[round((np.argmin(sim) + np.argmax(sim))/2)], 1)
    if pOpt[2] > 180:
        pOpt[2] = pOpt[2] - 360
    elif pOpt[2] < -180:
        p[2] = pOpt[2] + 360
    if pOpt[2] < 0:
        pi_length = (1/2 - pOpt[2]/180)/2/pOpt[1]
        pi2_length = (0 - pOpt[2]/180)/2/pOpt[1]
    else:
        pi_length = (3/2 - pOpt[2]/180)/2/pOpt[1]
        pi2_length = (1 - pOpt[2]/180)/2/pOpt[1]

    plt.figure(figsize=figsize)
    plt.plot(x, y, label='meas', ls='-', marker='o', markersize=3)
    if fit == True:
        plt.plot(x, sim, label='fit')
    plt.title(f'Length Rabi', fontsize=15)
    plt.xlabel('$t\ (us)$', fontsize=15)
    if normalize == True:
        plt.ylabel('Population', fontsize=15)
        plt.axvline(pi, ls='--', c='red', label=f'$\pi$ len={pi}')
        plt.axvline(pi2, ls='--', c='red', label=f'$\pi/2$ len={pi2}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        return pi, pi2
    else:
        plt.axvline(pi_length, ls='--', c='red',
                    label=f'$\pi$ length={pi_length:.3f}$\mu$s')
        plt.axvline(pi2_length, ls='--', c='red',
                    label=f'$\pi$ length={pi2_length:.3f}$\mu$s')
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

    plt.figure(figsize=figsize)
    plt.plot(x, y, label='meas', ls='-', marker='o', markersize=3)
    if fit == True:
        plt.plot(x, sim, label='fit')
    plt.title(f'T1 = {pOpt[3]:.2f}$\mu s$', fontsize=15)
    plt.xlabel('$t\ (\mu s)$', fontsize=15)
    if normalize == True:
        plt.ylabel('Population', fontsize=15)
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
    pOpt, pCov = fitdecaysin(x, y)
    sim = decaysin(x, *pOpt)
    error = np.sqrt(np.diag(pCov))

    plt.figure(figsize=figsize)
    plt.plot(x, y, label='meas', ls='-', marker='o', markersize=3)
    if fit == True:
        plt.plot(x, sim, label='fit')
    plt.title(
        f'T2r = {pOpt[3]:.2f}$\mu s, detune = {pOpt[1]:.2f}MHz \pm {(error[1])*1e3:.2f}kHz$', fontsize=15)
    plt.xlabel('$t\ (\mu s)$', fontsize=15)
    if normalize == True:
        plt.ylabel('Population', fontsize=15)
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

    plt.figure(figsize=figsize)
    plt.plot(x, y, label='meas', ls='-', marker='o', markersize=3)
    if fit == True:
        plt.plot(x, sim, label='fit')
    plt.title(f'T2e = {pOpt[3]:.2f}$\mu s$', fontsize=15)
    plt.xlabel('$t\ (\mu s)$', fontsize=15)
    if normalize == True:
        plt.ylabel('Population', fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    import os
    os.chdir(r'C:\Users\QEL\Desktop\Jay PhD\Code\data_fitting\data')
    sys.path.append(r'C:\Program Files\Keysight\Labber\Script')
    import Labber

    np.bool = bool
    np.float = np.float64
    res = r'Normalized_coupler_chen_054[7]_@7.819GHz_power_dp_002.hdf5'
    pdr = r'r1_pdr.hdf5'
    lenghrabi = r'q1_rabi.hdf5'
    t1 = r'q1_T1_2.hdf5'
    t2 = r'q1_T2Ramsey_3.hdf5'
    spec = r'q1_twotone_4.hdf5'

    spec = Labber.LogFile(spec)
    pdr = Labber.LogFile(pdr)
    fr = Labber.LogFile(res)
    f1 = Labber.LogFile(lenghrabi)
    f2 = Labber.LogFile(t1)
    f3 = Labber.LogFile(t2)
    (x, y) = fr.getTraceXY(entry=3)
    (sx, sy) = spec.getTraceXY()
    (rx, ry) = f1.getTraceXY()
    (t1x, t1y) = f2.getTraceXY()
    (t2x, t2y) = f3.getTraceXY()
    # spectrum_analyze(sx, sy, plot=True)
    # lengthraig_analyze(rx, ry, plot=True, normalize=False)
    # amprabi_analyze(rx, ry, fit=True, normalize=True)
    # T1_analyze(t1x, t1y, fit=True,normalize=True)
    T2r_analyze(t2x, t2y, fit=True, normalize=False)
    # resonator_analyze(x,y)
