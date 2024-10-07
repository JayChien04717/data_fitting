import numpy as np
import scipy as sp
import cmath

# ====================================================== #
"""
Compare the fit between the check_measures (amps, avgi, and avgq by default) in data, using the compare_param_i-th parameter to do the comparison. Pick the best method of measurement out of the check_measures, and return the fit, fit_err, and any other get_best_data_params corresponding to that measurement.

If fitfunc is specified, uses R^2 to determine best fit.
"""


def get_best_fit(
    data,
    fitfunc=None,
    prefixes=["fit"],
    check_measures=("amps", "avgi", "avgq"),
    get_best_data_params=(),
    override=None,
):
    fit_errs = [
        data[f"{prefix}_err_{check}"] for check in check_measures for prefix in prefixes
    ]
    all_check_measures = [
        f"{prefix}_err_{check}" for check in check_measures for prefix in prefixes
    ]

    # fix the error matrix so "0" error is adjusted to inf
    for fit_err_check in fit_errs:
        for i, fit_err in enumerate(np.diag(fit_err_check)):
            if fit_err == 0:
                fit_err_check[i][i] = np.inf

    fits = [
        data[f"{prefix}_{check}"] for check in check_measures for prefix in prefixes
    ]

    if override is not None and override in all_check_measures:
        i_best = np.argwhere(np.array(all_check_measures) == override)[0][0]
    else:
        if fitfunc is not None:
            ydata = [
                data[check] for check in all_check_measures
            ]  # need to figure out how to make this support multiple qubits readout
            xdata = data["xpts"]

            # residual sum of squares
            ss_res_checks = np.array(
                [
                    np.sum((fitfunc(xdata, *fit_check) - ydata_check) ** 2)
                    for fit_check, ydata_check in zip(fits, ydata)
                ]
            )
            # total sum of squares
            ss_tot_checks = np.array(
                [
                    np.sum((np.mean(ydata_check) - ydata_check) ** 2)
                    for ydata_check in ydata
                ]
            )
            # R^2 value
            r2 = 1 - ss_res_checks / ss_tot_checks

            # override r2 value if fit is bad
            for icheck, fit_err_check in enumerate(fit_errs):
                for i, fit_err in enumerate(np.diag(fit_err_check)):
                    if fit_err == np.inf:
                        r2[icheck] = np.inf
            i_best = np.argmin(r2)

        else:
            # i_best = np.argmin([np.sqrt(np.abs(fit_err[compare_param_i][compare_param_i])) for fit, fit_err in zip(fits, fit_errs)])
            # i_best = np.argmin([np.sqrt(np.abs(fit_err[compare_param_i][compare_param_i] / fit[compare_param_i])) for fit, fit_err in zip(fits, fit_errs)])
            errs = [
                np.average(np.sqrt(np.abs(np.diag(fit_err))) / np.abs(fit))
                for fit, fit_err in zip(fits, fit_errs)
            ]
            # print([np.sqrt(np.abs(np.diag(fit_err))) / np.abs(fit) for fit, fit_err in zip(fits, fit_errs)])
            for i_err, err in enumerate(errs):
                if err == np.nan:
                    errs[i_err] = np.inf
            # print(errs)
            i_best = np.argmin(errs)
            # print(i_best)
    print(i_best)

    best_data = [fits[i_best], fit_errs[i_best]]

    for param in get_best_data_params:
        assert (
            len(fit_errs) == len(check_measures)
        ), "this is a pathological use of this function anyway, so just restrict to these cases"
        # best_meas = all_check_measures[i_best]
        best_meas = check_measures[i_best]
        best_data.append(data[f"{param}_{best_meas}"])
    return best_data


# ====================================================== #


def expfunc(x, *p):
    y0, yscale, x0, decay = p
    return y0 + yscale * np.exp(-(x - x0) / decay)


def fitexp(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 4
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = ydata[-1]
    if fitparams[1] is None:
        fitparams[1] = ydata[0] - ydata[-1]
    if fitparams[2] is None:
        fitparams[2] = xdata[0]
    if fitparams[3] is None:
        fitparams[3] = (xdata[-1] - xdata[0]) / 5
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(expfunc, xdata, ydata, p0=fitparams)
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


def logexpfunc(x, *p):
    decay = p
    return np.log(np.exp(-x / decay))


def fitlogexp(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 1
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = (xdata[-1] - xdata[0]) / 5
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(logexpfunc, xdata, ydata, p0=fitparams)
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


# ====================================================== #
# see https://www.science.org/doi/epdf/10.1126/science.aah5844
# assumes function has been scaled to log scale and goes from 0 to 1
def qp_expfunc(x, *p):
    nqp, t1qp, t1r = p
    return nqp * (np.exp(-x / t1qp) - 1) - x / t1r


def fitqpexp(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 3
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = 1.0
    if fitparams[1] is None:
        fitparams[1] = (xdata[-1] - xdata[0]) / 2
    if fitparams[2] is None:
        fitparams[2] = (xdata[-1] - xdata[0]) / 2
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    bounds = (
        [0.01, 0.01 * fitparams[1], 0.01 * fitparams[2]],
        [5, 2 * fitparams[1], 10 * fitparams[2]],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            qp_expfunc, xdata, ydata, p0=fitparams, bounds=bounds
        )
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


# ====================================================== #
"""
single lorentzian function 
"""


def lorfunc(x, *p):
    y0, yscale, x0, xscale = p
    return y0 + yscale / (1 + (x - x0) ** 2 / xscale**2)


def fitlor(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 4
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = (ydata[0] + ydata[-1]) / 2
    if fitparams[1] is None:
        fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None:
        fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[3] is None:
        fitparams[3] = (max(xdata) - min(xdata)) / 10
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(lorfunc, xdata, ydata, p0=fitparams)
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


# ====================================================== #
""" 
Two lorentzian function
"""


def twolorfunc(x, *p):
    y0, yscale1, x0_1, xscale1, yscale2, x0_2, xscale2 = p
    return (
        y0
        + yscale1 / (1 + (x - x0_1) ** 2 / xscale1**2)
        + yscale2 / (1 + (x - x0_2) ** 2 / xscale2**2)
    )


def fittwolor(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 7
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = (ydata[0] + ydata[-1]) / 2
    if fitparams[1] is None:
        fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None:
        fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[3] is None:
        fitparams[3] = (max(xdata) - min(xdata)) / 3
    if fitparams[4] is None:
        fitparams[4] = max(ydata) - min(ydata)
    if fitparams[5] is None:
        fitparams[5] = xdata[np.argmax(abs(ydata - fitparams[0])) - 10]
    if fitparams[6] is None:
        fitparams[6] = (max(xdata) - min(xdata)) / 3

    fitparams = [0 if param is None else param for param in fitparams]
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(twolorfunc, xdata, ydata, p0=fitparams)
    except RuntimeError:
        print("Warning: fit failed!")
    return pOpt, pCov


# ====================================================== #
""" 
Three lorentzian function
"""


def threelorfunc(x, *p):
    y0, yscale1, x0_1, xscale1, yscale2, x0_2, xscale2, yscale3, x0_3, xscale3 = p
    return (
        y0
        + yscale1 / (1 + (x - x0_1) ** 2 / xscale1**2)
        + yscale2 / (1 + (x - x0_2) ** 2 / xscale2**2)
        + yscale3 / (1 + (x - x0_3) ** 2 / xscale3**2)
    )


def fitthreelor(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 10
    else:
        fitparams = np.copy(fitparams)

    if fitparams[0] is None:
        fitparams[0] = (ydata[0] + ydata[-1]) / 2
    if fitparams[1] is None:
        fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None:
        fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[3] is None:
        fitparams[3] = (max(xdata) - min(xdata)) / 3
    if fitparams[4] is None:
        fitparams[4] = max(ydata) - min(ydata)
    if fitparams[5] is None:
        fitparams[5] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[6] is None:
        fitparams[6] = (max(xdata) - min(xdata)) / 5
    if fitparams[7] is None:
        fitparams[7] = max(ydata) - min(ydata)
    if fitparams[8] is None:
        fitparams[8] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[9] is None:
        fitparams[9] = (max(xdata) - min(xdata)) / 5

    fitparams = [0 if param is None else param for param in fitparams]
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(threelorfunc, xdata, ydata, p0=fitparams)
    except RuntimeError:
        print("Warning: fit failed!")
    return pOpt, pCov


# ====================================================== #


def sinfunc(x, *p):
    yscale, freq, phase_deg, y0 = p
    return yscale * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180) + y0


def fitsin(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 4
    else:
        fitparams = np.copy(fitparams)
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    fft_phases = np.angle(fourier)
    max_ind = np.argmax(np.abs(fourier[1:])) + 1
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None:
        fitparams[0] = max(ydata) - min(ydata)
    if fitparams[1] is None:
        fitparams[1] = max_freq
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi
    if fitparams[3] is None:
        fitparams[3] = np.mean(ydata)
    bounds = (
        [0.5 * fitparams[0], 0.2 / (max(xdata) - min(xdata)), -360, np.min(ydata)],
        [2 * fitparams[0], 5 / (max(xdata) - min(xdata)), 360, np.max(ydata)],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            sinfunc, xdata, ydata, p0=fitparams, bounds=bounds
        )
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


# ====================================================== #
""" 
asymmtric lorentzian function
"""


def asym_lorfunc(x, *p):
    y0, A, x0, gamma, alpha = p
    return y0 + A / (1 + ((x - x0) / (gamma * (1 + alpha * (x - x0)))) ** 2)


def fit_asym_lor(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 5
    else:
        fitparams = np.copy(fitparams)

    if fitparams[0] is None:
        fitparams[0] = (ydata[0] + ydata[-1]) / 2
    if fitparams[1] is None:
        fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None:
        fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[3] is None:
        fitparams[3] = (max(xdata) - min(xdata)) / 10
    if fitparams[4] is None:
        fitparams[4] = 0

    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(asym_lorfunc, xdata, ydata, p0=fitparams)
    except RuntimeError:
        print("Warning: fit failed!")
    return pOpt, pCov


# ====================================================== #


def decaysin(x, *p):
    yscale, freq, phase_deg, decay, y0 = p
    return (
        yscale
        * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180)
        * np.exp(-x / decay)
        + y0
    )


def fitdecaysin(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 5
    else:
        fitparams = np.copy(fitparams)
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None:
        fitparams[0] = (max(ydata) - min(ydata)) / 2
    if fitparams[1] is None:
        fitparams[1] = max_freq
    # if fitparams[2] is None: fitparams[2]=0
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi
    if fitparams[3] is None:
        fitparams[3] = max(xdata) - min(xdata)
    if fitparams[4] is None:
        fitparams[4] = np.mean(ydata)
    bounds = (
        [
            0.75 * fitparams[0],
            0.1 / (max(xdata) - min(xdata)),
            -360,
            0.3 * (max(xdata) - min(xdata)),
            np.min(ydata),
        ],
        [
            1.25 * fitparams[0],
            30 / (max(xdata) - min(xdata)),
            360,
            np.inf,
            np.max(ydata),
        ],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            decaysin, xdata, ydata, p0=fitparams, bounds=bounds
        )
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


def fitampdecaysin(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*5
    else: fitparams = np.copy(fitparams)
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1]-xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None: fitparams[0]=(max(ydata)-min(ydata))/2
    if fitparams[1] is None: fitparams[1]=max_freq
    # if fitparams[2] is None: fitparams[2]=0
    if fitparams[2] is None: fitparams[2]=max_phase*180/np.pi
    if fitparams[3] is None: fitparams[3]=max(xdata) - min(xdata)
    if fitparams[4] is None: fitparams[4]=np.mean(ydata)
    bounds = (
        [0.75*fitparams[0], 0.1/(max(xdata)-min(xdata)), -360, 1e5, np.min(ydata)],
        [1.25*fitparams[0], 30/(max(xdata)-min(xdata)), 360, 1e10, np.max(ydata)]
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(decaysin, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov
# ====================================================== #
# T2 ramsey modify with 1/f noise type, check [A Quantum Engineer’s Guide to Superconducting Qubits]
def decaysinmod(x, *p):
    yscale, freq, phase_deg, t1, tphi, y0 = p
    return (
        yscale
        * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180)
        * np.exp(-x / (2 * t1))
        * np.exp(-(x**2) / tphi**2)
    )


def fitdecaysinmod(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 6
    else:
        fitparams = np.copy(fitparams)
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None:
        fitparams[0] = (max(ydata) - min(ydata)) / 2
    if fitparams[1] is None:
        fitparams[1] = max_freq
    # if fitparams[2] is None: fitparams[2]=0
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi
    if fitparams[3] is None:
        fitparams[3] = max(xdata) - min(xdata)
    if fitparams[4] is None:
        fitparams[4] = 0.5 * (max(xdata) - min(xdata))
    if fitparams[5] is None:
        fitparams[5] = np.mean(ydata)
    bounds = (
        [
            0.75 * fitparams[0],
            0.1 / (max(xdata) - min(xdata)),
            -360,
            0.05 * (max(xdata) - min(xdata)),
            0.05 * (max(xdata) - min(xdata)),
            np.min(ydata),
        ],
        [
            1.25 * fitparams[0],
            30 / (max(xdata) - min(xdata)),
            360,
            np.inf,
            np.inf,
            np.max(ydata),
        ],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            decaysinmod, xdata, ydata, p0=fitparams, bounds=bounds
        )
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


# ====================================================== #


def twofreq_decaysin(x, *p):
    yscale0, freq0, phase_deg0, decay0, yscale1, freq1, phase_deg1, y0 = p
    return y0 + np.exp(-x / decay0) * yscale0 * (
        (1 - yscale1) * np.sin(2 * np.pi * freq0 * x + phase_deg0 * np.pi / 180)
        + yscale1 * np.sin(2 * np.pi * freq1 * x + phase_deg1 * np.pi / 180)
    )


def fittwofreq_decaysin(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 10
    else:
        fitparams = np.copy(fitparams)
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None:
        fitparams[0] = max(ydata) - min(ydata)  # yscale0
    if fitparams[1] is None:
        fitparams[1] = max_freq  # freq0
    # if fitparams[2] is None: fitparams[2]=0
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi  # phase_deg0
    if fitparams[3] is None:
        fitparams[3] = max(xdata) - min(xdata)  # exp decay
    if fitparams[4] is None:
        fitparams[4] = 0.1  # yscale1
    if fitparams[5] is None:
        fitparams[5] = 0.5  # MHz
    if fitparams[6] is None:
        fitparams[6] = 0  # phase_deg1
    if fitparams[7] is None:
        fitparams[7] = np.mean(ydata)  # y0
    bounds = (
        [
            0.75 * fitparams[0],
            0.1 / (max(xdata) - min(xdata)),
            -360,
            0.1 * (max(xdata) - min(xdata)),
            0.001,
            0.01,
            -360,
            np.min(ydata),
        ],
        [
            1.25 * fitparams[0],
            30 / (max(xdata) - min(xdata)),
            360,
            np.inf,
            0.5,
            10,
            360,
            np.max(ydata),
        ],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            twofreq_decaysin, xdata, ydata, p0=fitparams, bounds=bounds
        )
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


def threefreq_decaysin(x, *p):
    (
        yscale0,
        freq0,
        phase_deg0,
        decay0,
        y00,
        x00,
        yscale1,
        freq1,
        phase_deg1,
        y01,
        yscale2,
        freq2,
        phase_deg2,
        y02,
    ) = p
    p0 = [yscale0, freq0, phase_deg0, decay0, 0]
    p1 = [yscale1, freq1, phase_deg1, y01]
    p2 = [yscale2, freq2, phase_deg2, y02]
    return y00 + decaysin(x, *p0) * sinfunc(x, *p1) * sinfunc(x, *p2)


def fitthreefreq_decaysin(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 14
    else:
        fitparams = np.copy(fitparams)
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None:
        fitparams[0] = max(ydata) - min(ydata)
    if fitparams[1] is None:
        fitparams[1] = max_freq
    # if fitparams[2] is None: fitparams[2]=0
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi
    if fitparams[3] is None:
        fitparams[3] = max(xdata) - min(xdata)
    if fitparams[4] is None:
        fitparams[4] = np.mean(ydata)  #
    if fitparams[5] is None:
        fitparams[5] = xdata[0]  # x0 (exp decay)
    if fitparams[6] is None:
        fitparams[6] = 1  # y scale
    if fitparams[7] is None:
        fitparams[7] = 1 / 10  # MHz
    if fitparams[8] is None:
        fitparams[8] = 0  # phase degrees
    if fitparams[9] is None:
        fitparams[9] = 0  # y0
    if fitparams[10] is None:
        fitparams[10] = 1  # y scale
    if fitparams[11] is None:
        fitparams[11] = 1 / 10  # MHz
    if fitparams[12] is None:
        fitparams[12] = 0  # phase degrees
    if fitparams[13] is None:
        fitparams[13] = 0  # y0
    bounds = (
        [
            0.75 * fitparams[0],
            0.1 / (max(xdata) - min(xdata)),
            -360,
            0.3 * (max(xdata) - min(xdata)),
            np.min(ydata),
            xdata[0] - (xdata[-1] - xdata[0]),
            0.5,
            0.01,
            -360,
            -0.1,
            0.5,
            0.01,
            -360,
            -0.1,
        ],
        [
            1.25 * fitparams[0],
            15 / (max(xdata) - min(xdata)),
            360,
            np.inf,
            np.max(ydata),
            xdata[-1] + (xdata[-1] - xdata[0]),
            1.5,
            10,
            360,
            0.1,
            1.5,
            10,
            360,
            0.1,
        ],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            threefreq_decaysin, xdata, ydata, p0=fitparams, bounds=bounds
        )
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


# ====================================================== #


def hangerfunc(x, *p):
    f0, Qi, Qe, phi, scale, a0 = p
    Q0 = 1 / (1 / Qi + np.real(1 / Qe))
    return scale * (1 - Q0 / Qe * np.exp(1j * phi) / (1 + 2j * Q0 * (x - f0) / f0))


def hangerS21func(x, *p):
    f0, Qi, Qe, phi, scale, a0 = p
    Q0 = 1 / (1 / Qi + np.real(1 / Qe))
    return a0 + np.abs(hangerfunc(x, *p)) - scale * (1 - Q0 / Qe)


def hangerS21func_sloped(x, *p):
    f0, Qi, Qe, phi, scale, a0, slope = p
    return hangerS21func(x, f0, Qi, Qe, phi, scale, a0) + slope * (x - f0)


def hangerphasefunc(x, *p):
    return np.angle(hangerfunc(x, *p))


def fithanger(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 7
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = np.average(xdata)
    if fitparams[1] is None:
        fitparams[1] = 5000
    if fitparams[2] is None:
        fitparams[2] = 1000
    if fitparams[3] is None:
        fitparams[3] = 0
    if fitparams[4] is None:
        fitparams[4] = max(ydata) - min(ydata)
    if fitparams[5] is None:
        fitparams[5] = np.average(ydata)
    if fitparams[6] is None:
        fitparams[6] = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])

    print(fitparams)

    # bounds = (
    #     [np.min(xdata), -1e9, -1e9, -2*np.pi, (max(np.abs(ydata))-min(np.abs(ydata)))/10, -np.max(np.abs(ydata))],
    #     [np.max(xdata), 1e9, 1e9, 2*np.pi, (max(np.abs(ydata))-min(np.abs(ydata)))*10, np.max(np.abs(ydata))]
    #     )
    bounds = (
        [np.min(xdata), 0, 0, -np.inf, 0, min(ydata), -np.inf],
        [np.max(xdata), np.inf, np.inf, np.inf, np.inf, max(ydata), np.inf],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )

    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            hangerS21func_sloped, xdata, ydata, p0=fitparams, bounds=bounds
        )
        print(pOpt)
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


# ====================================================== #


# calculate density of points by kde
def calculate_density(X, Y, bandwidth: float = 0.1):
    return sp.stats.gaussian_kde([X, Y], bw_method=bandwidth)([X, Y])


# fit data to a 2D Gaussian
def Gaussian2D(xy, xo, yo, sigma):
    x, y = xy
    amplitude = 1 / (2 * np.pi * sigma**2)
    g = amplitude * np.exp(-((x - xo) ** 2 + (y - yo) ** 2) / (2 * sigma**2))

    return g.ravel()


def DualGaussian2D(xy, xo0, yo0, sigma0, xo1, yo1, sigma1, a_ratio):
    A, B = a_ratio, 1 - a_ratio
    return A * Gaussian2D(xy, xo0, yo0, sigma0) + B * Gaussian2D(xy, xo1, yo1, sigma1)


X_LEN, Y_LEN = 100, 100


def fit_data(X: np.ndarray, Y: np.ndarray, D: np.ndarray):
    # initial guess
    max_id = np.argmax(D)
    x_max, y_max = X[max_id], Y[max_id]
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    sigma = (np.std(X) + np.std(Y)) / 2

    x0 = x_max
    y0 = y_max
    x1 = 2 * x_mean - x_max
    y1 = 2 * y_mean - y_max

    p0 = (x0, y0, 0.5 * sigma, x1, y1, 0.5 * sigma, 0.8)

    # fit
    param, _ = sp.optimize.curve_fit(
        DualGaussian2D,
        (X, Y),
        D,
        p0=p0,
        bounds=(
            [
                x0 - 2.5 * sigma,
                y0 - 2.5 * sigma,
                0,
                x1 - 5 * sigma,
                x1 - 5 * sigma,
                0,
                0.6,
            ],
            [
                x0 + 2.5 * sigma,
                y0 + 2.5 * sigma,
                sigma,
                x1 + 5 * sigma,
                y1 + 5 * sigma,
                sigma,
                1.0,
            ],
        ),
    )

    return param


# change coordinate to make two fit center align with x-axis
def rotate(x, y, angle):
    x_rot = x * np.cos(angle) + y * np.sin(angle)
    y_rot = -x * np.sin(angle) + y * np.cos(angle)

    return x_rot, y_rot


def rotate_data(X0, Y0, param0, X1, Y1, param1):
    center00 = np.array([param0[0], param0[1]])
    center01 = np.array([param0[3], param0[4]])
    center10 = np.array([param1[0], param1[1]])
    center11 = np.array([param1[3], param1[4]])

    angle = np.arctan2(center10[1] - center00[1], center10[0] - center00[0])

    X0_rot, Y0_rot = rotate(X0, Y0, angle)
    X1_rot, Y1_rot = rotate(X1, Y1, angle)

    param0_rot = list(param0)
    param0_rot[0], param0_rot[1] = rotate(center00[0], center00[1], angle)
    param0_rot[3], param0_rot[4] = rotate(center01[0], center01[1], angle)

    param1_rot = list(param1)
    param1_rot[0], param1_rot[1] = rotate(center10[0], center10[1], angle)
    param1_rot[3], param1_rot[4] = rotate(center11[0], center11[1], angle)

    return angle, X0_rot, Y0_rot, param0_rot, X1_rot, Y1_rot, param1_rot


# ====================================================== #


def rb_func(depth, p, a, b):
    return a * p**depth + b


# Gives the average error rate over all gates in sequence
def rb_error(p, d):  # d = dim of system = 2^(number of qubits)
    return 1 - (p + (1 - p) / d)


# return covariance of rb error
def error_fit_err(cov_p, d):
    return cov_p * (1 / d - 1) ** 2


# Run both regular RB and interleaved RB to calculate this
def rb_gate_fidelity(p_rb, p_irb, d):
    return 1 - (d - 1) * (1 - p_irb / p_rb) / d


def fitrb(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 3
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = 0.9
    if fitparams[1] is None:
        fitparams[1] = np.max(ydata) - np.min(ydata)
    if fitparams[2] is None:
        fitparams[2] = np.min(ydata)
    bounds = ([0, 0, 0], [1, 10 * np.max(ydata) - np.min(ydata), np.max(ydata)])
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            rb_func, xdata, ydata, p0=fitparams, bounds=bounds
        )
        print(pOpt)
        print(pCov[0][0], pCov[1][1], pCov[2][2])
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


# ====================================================== #
# Adiabatic pi pulse functions
# beta ~ slope of the frequency sweep (also adjusts width)
# mu ~ width of frequency sweep (also adjusts slope)
# period: delta frequency sweeps through zero at period/2
# amp_max


def adiabatic_amp(t, amp_max, beta, period):
    return amp_max / np.cosh(beta * (2 * t / period - 1))


def adiabatic_phase(t, mu, beta, period):
    return mu * np.log(adiabatic_amp(t, amp_max=1, beta=beta, period=period))


def adiabatic_iqamp(t, amp_max, mu, beta, period):
    amp = np.abs(adiabatic_amp(t, amp_max=amp_max, beta=beta, period=period))
    phase = adiabatic_phase(t, mu=mu, beta=beta, period=period)
    iamp = amp * (np.cos(phase) + 1j * np.sin(phase))
    qamp = amp * (-np.sin(phase) + 1j * np.cos(phase))
    return np.real(iamp), np.real(qamp)


# ====================================================== #
# Correcting for over/under rotation
# delta: angle error in degrees


def probg_Xhalf(n, *p):
    a, delta = p
    delta = delta * np.pi / 180
    return a + (0.5 * (-1) ** n * np.cos(np.pi / 2 + 2 * n * delta))


def probg_X(n, *p):
    a, delta = p
    delta = delta * np.pi / 180
    return a + (0.5 * np.cos(np.pi / 2 + 2 * n * delta))


def fit_probg_Xhalf(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 2
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = np.average(ydata)
    if fitparams[1] is None:
        fitparams[1] = 0.0
    bounds = (
        [min(ydata), -20.0],
        [max(ydata), 20.0],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            probg_Xhalf, xdata, ydata, p0=fitparams, bounds=bounds
        )
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


def fit_probg_X(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 2
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = np.average(ydata)
    if fitparams[1] is None:
        fitparams[1] = 0.0
    bounds = (
        [min(ydata), -20.0],
        [max(ydata), 20.0],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            probg_X, xdata, ydata, p0=fitparams, bounds=bounds
        )
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


# ====================================================== #
# ref: https://github.com/steelelab-delft/stlab/blob/master/misc/S11fit.py
# Thesis: Coupling Harmonic Oscillators to Superconducting Quantum Interference Cavities. Inês C. Rodrigues


"""Module for fitting resonance line shapes to different circuit models

This module contains the functions necessary to fit some general Lorentzian to
simulations or measurements.  The main function is "fit" and is imported with
stlab as "stlab.S11fit".  All other functions in this module are there to
supplement this fitting function or are not generally used.

"""

pi = np.pi


def newparams(f0=1, Qint=100, Qext=200, theta=0, a=1, b=0, c=0, ap=0, bp=0):
    """Makes new Parameters object compatible with fitting routine

    A new lmfit.Parameters object is created using the given values.  Default
    values are filled in for ommited parameters.  The fit model is:

    .. math:: \Gamma'(\omega)=(a+b\omega+c\omega^2)\exp(j(a'+b'\omega))\cdot
        \Gamma(\omega,Q_\\textrm{int},Q_\\textrm{ext},\\theta),

    Parameters
    ----------
    f0 : float, optional
        Resonance frequency
    Qint : float, optional
        Internal quality factor
    Qext : float, optional
        External quality factor
    theta : float, optional
        Rotation angle to compensate additive background effects.  Should be
        close to 0 for good fits.
    a : float, optional
        Background magnitude offset
    b : float, optional
        Background magnitude linear slope
    c : float, optional
        Background magnitude quadratic term
    ap : float, optional
        Background phase offset
    bp : float, optional
        Background phase slope

    Returns
    -------
    params : lmfit.Parameters
        lmfit fit object containing the provided parameters

    """

    params = lmfit.Parameters()

    params.add('a', value=a, vary=True)
    params.add('b', value=b, vary=True)
    params.add('c', value=c, vary=True)
    params.add('ap', value=ap, vary=True)
    params.add('bp', value=bp, vary=True)
    params.add('cp', value=0, vary=False)

    params.add('Qint', value=Qint, vary=True, min=0)
    params.add('Qext', value=Qext, vary=True, min=0)
    params.add('f0', value=f0, vary=True)
    params.add('theta', value=theta, vary=True)

    return params


def realimag(array):
    """Makes alternating real and imaginary part array from complex array

    Takes an array-like object of complex number elements and generates a
    1-D array aleternating the real and imaginary part of each element of the
    original array.  If array = (z1,z2,...,zn), then this function returns
    (x1,y1,x2,y2,...,xn,yn) where xi = np.real(zi) and yi=np.imag(zi).

    Parameters
    ----------
    array : array_like of complex
        1-D array of complex numbers

    Returns
    -------
    numpy.ndarray
        New array of alternating real and imag parts of each element of
        original array

    """
    return np.array([(x.real, x.imag) for x in array]).flatten()


def un_realimag(array):
    """Makes complex array from alternating real and imaginary part array

    Performs the reverse operation to realimag

    Parameters
    ----------
    array : array_like of float
        1-D array of real numbers.  Should have an even number of elements.

    Returns
    -------
    numpy.ndarray of numpy.complex128
        1-D array of complex numbers built by taking every two elements of
        original array as the real and imaginary parts

    """
    z = []
    for x, y in zip(array[::2], array[1::2]):
        z.append(x+1j*y)
    return np.array(z)


def phaseunwrap(array):
    """Removes a global phase slope from a complex array

    Unwraps the phase of a sequence of complex numbers and subtracts the average
    slope of the phase (desloped phase).

    Parameters
    ----------
    array : array_like of complex
        1-D array of complex numbers

    Returns
    -------
    numpy.ndarray of numpy.complex128
        Same array as original but with phase slope removed (0 average phase
        slope)

    """
    data = np.asarray(array)
    phase = np.unwrap(np.angle(data))
    avg = np.average(np.diff(phase))
    data = [x*np.exp(-1j*avg*i) for i, x in enumerate(data)]
#    print(np.average(np.diff(np.angle(data))))
    return np.asarray(data)


def getwidth_phase(i0, vec, margin):
    """Finds indices for peak width around given maximum position

    Auxiliary function for fit.  Given the complex array "vec" assuming "i0"
    is the resonance index, this function finds resonance peak width from the
    phase derivative of the signal.

    Parameters
    ----------
    i0 : int
        Array index of resonance
    vec : array_like of complex
        Complex array with resonance data
    margin : int
        Smoothing margin used on data.  Needed to remove spureous increases in
        the phase array that occur at the head and tail of vec after smoothing.
        Should be an odd number

    Returns
    -------
    (int, int)
        Indices of the lower and upper estimated edges of the resonance peak

    """
    maxvec = vec[i0]
    if margin == 0:
        avgvec = np.average(vec[0:-1])
    else:
        avgvec = np.average(vec[margin:-margin])
#    print maxvec, avgvec
    i2 = len(vec)-1
    i1 = 0
    for i in range(i0, len(vec)):
        #        print (maxvec-vec[i]), (maxvec-avgvec)
        if (maxvec-vec[i]) > (maxvec-avgvec)/1.:
            i2 = i
            break
    for i in range(i0, -1, -1):
        if (maxvec-vec[i]) > (maxvec-avgvec)/1.:
            i1 = i
            break
    return (i1, i2)


def trim(x, y, imin, imax):
    """Removes range from imin to imax from vectors x,y

    Given two (possibly complex) arrays and indices corresponding to a lower and upper edge,
    this function removes the index range between these edges from both input
    arrays

    Parameters
    ----------
    x, y : array_like of complex
        Arrays to be trimmed
    imin, imax : int
        Lower and upper edge of range to be removed (trimmed) from x,y

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Trimmed arrays

    """
    imin = int(imin)
    imax = int(imax)
    print(len(x), len(y))

    def corr(xx):
        xpart1 = xx[0:imin]
        xpart2 = xx[imax:]
        if len(xpart1) == 0:
            xpart1 = xx[0:1]
        if len(xpart2) == 0:
            xpart2 = xx[-1:]
        return xpart1, xpart2
    xnew = np.concatenate(corr(x))
    ynew = np.concatenate(corr(y))

    if len(xnew) < 4 or len(ynew) < 4:
        xnew = np.concatenate([x[0:2], x[-2:]])
        ynew = np.concatenate([y[0:2], y[-2:]])

    # print(xnew,ynew)

    return (xnew, ynew)


def backmodel(x, params):
    """Function for background model.

    Returns the background model values for a given set of parameters and
    frequency values.  Uses parameter object from lmfit.  The background model
    is given by

    .. math:: f_b(\omega)=(a+b\omega+c\omega^2)
        \exp(j(a'+b'\omega)),

    where :math:`a,b,c,a',b'` are real parameters.

    Parameters
    ----------
    x : float or array_like of float
        Frequency values to evaluate the model at
    params : lmfit.Parameters
        Parameters set with which to generate the background

    Returns
    -------
    numpy.complex128 or numpy.ndarray of numpy.complex128
        Background values at frequencies x with model parameters params

    """
    a = params['a'].value
    b = params['b'].value
    c = params['c'].value
    ap = params['ap'].value
    bp = params['bp'].value
    cp = params['cp'].value
    return (a+b*x+c*np.power(x, 2.))*np.exp(1j*(ap+bp*x+cp*np.power(x, 2.)))


def background2min(params, x, data):
    """Background residual

    Computes the residual vector for the background fitting.  Operates on
    complex vectors but returns a real vector with alternating real and imag
    parts

    Parameters
    ----------
    params : lmfit.Parameters
        Model parameters for background generation
    x : float or array_like of float
        Frequency values for background calculation.  backmodel function is
        used.
    data : complex or array_like of complex
        Background values of signal to be compared to generated background at
        frequency values x.

    Returns
    -------
    numpy.ndarray of numpy.float
        Residual values (model value - data value) at values given in x.
        Alternates real and imaginary values

    """
    model = backmodel(x, params)
    res = model - data
    return realimag(res)


def S11theo(frec, params, ftype='A'):  # Equation
    """Theoretical response functions of cavities with no background

    Returns the theory values of cavity response functions for a given set of
    parameters for different cavity models.

    Parameters
    ----------
    frec : float or array_like of float
        Frequency values to calculate the response function at
    params : lmfit.Parameters
        Parameter values for calculation
    ftype : {'A','-A','B','-B','X'}, optional
        Model for response function selection.  These are described in Daniel's
        response function document.  The desired model should be found there and
        selected with this parameter.

        The possible models are (CHECK DANIEL'S RESPONSE FUNCTION DOCUMENT):

        - 'A': Reflection cavity with short circuit boundary condition at input port (default selection)
        - '-A': Reflection cavity with open circuit boundary condition at input port (= model A with a minus sign)
        - 'B': Transmission through a side coupled geometry
        - '-B': Same as model B but with a minux sign
        - 'X': Non-standard model (used in a magnetic field sweep)

    Returns
    -------
    numpy.complex128 or numpy.ndarray of numpy.complex128
        Values of theoretical response function at frequencies given in frec

    """
    Qint = params['Qint'].value
    Qext = params['Qext'].value
    f0 = params['f0'].value
    w0 = f0*2*pi
    w = 2*pi*frec
    kint = w0/Qint
    kext = w0/Qext
    dw = w-w0
#    return (kint+2j*dw) / (kext+kint+2j*dw)
    theta = params['theta']
    if ftype == "A":
        return -1.+(2.*kext*np.exp(1j*theta)) / (kext+kint+2j*dw)
    elif ftype == '-A':
        return 1.-(2.*kext*np.exp(1j*theta)) / (kext+kint+2j*dw)
    elif ftype == 'B':
        return -1.+(kext*np.exp(1j*theta)) / (kext+kint+2j*dw)
    elif ftype == '-B':
        return 1.-(kext*np.exp(1j*theta)) / (kext+kint+2j*dw)
    elif ftype == 'X':
        return 1.-(kext*np.exp(1j*theta)) / (kext+kint-2j*dw)


def S11residual(params, frec, data, ftype='A'):
    """Residual for full fit including background

    Calculates the residual for the full signal and background with respect to
    the given background and theory model.

    Parameters
    ----------
    params : lmfit.Parameters
        Parameter values for calculation
    frec : float or array_like of float
        Frequency values to calculate the combined signal and background
        response function at using params
    data : complex or array_like of complex
        Original signal data values at frequencies frec
    ftype : {'A','-A','B','-B','X'}, optional
        Model for theory response function selection.  See S11theo for
        explanation

    Returns
    -------
        Residual values (model value - data value) at values given in x.
        Alternates real and imaginary values

    """
    model = S11full(frec, params, ftype)
    residual = model - data
    return realimag(residual)


def S11full(frec, params, ftype='A'):
    """
    Function for total response from background model and resonator response
    """
    if ftype == 'A' or ftype == 'B':
        model = -S11theo(frec, params, ftype)*backmodel(frec, params)
    elif ftype == '-A' or ftype == '-B' or ftype == 'X':
        model = S11theo(frec, params, ftype)*backmodel(frec, params)
    return model


def fit(frec, S11, ftype='A', fitbackground=True, trimwidth=5., doplots=False, margin=51, oldpars=None, refitback=True, reusefitpars=False, fitwidth=None):
    """**MAIN FIT ROUTINE**

    Fits complex data S11 vs frecuency to one of 4 models adjusting for a multiplicative complex background
    It fits the data in three steps.  Firstly it fits the background signal removing a certain window around the detected peak position.
    Then it fits the model times the background to the full data set keeping the background parameters fixed at the fitted values.  Finally it refits all background and
    model parameters once more starting from the previously fitted values.  The fit model is:

    .. math:: \Gamma'(\omega)=(a+b\omega+c\omega^2)\exp(j(a'+b'\omega))\cdot
        \Gamma(\omega,Q_\\textrm{int},Q_\\textrm{ext},\\theta),


    Parameters
    ----------
    frec : array_like
        Array of X values (typically frequency)
    S11 : array_like
        Complex array of Z values (typically S11 data)
    ftype : {'A','B','-A','-B', 'X'}, optional
        Fit model function (A,B,-A,-B, see S11theo for formulas)
    fitbackground : bool, optional
        If "True" will attempt to fit and remove background.  If "False", will use a constant background equal to 1 and fit only model function to data.
    trimwidth : float, optional
        Number of linewidths around resonance (estimated pre-fit) to remove for background only fit.
    doplots : bool, optional
        If "True", shows debugging and intermediate plots
    margin : float, optional
        Smoothing window to apply to signal for initial guess procedures (the fit uses unsmoothed data)
    oldpars : lmfit.Parameters, optional
        Parameter data from previous fit (expects lmfit Parameter object). Used when "refitback" is "False" or "reusefitpars" is "True".
    refitback : bool, optional
        If set to False, does not fit the background but uses parameters provided in "oldpars".  If set to "True", fits background normally
    reusefitpars : bool, optional
        If set to True, uses parameters provided in "oldpars" as initial guess for fit parameters in main model fit (ignored by background fit)
    fitwidth : float, optional
        If set to a numerical value, will trim the signal to a certain number of widths around the resonance for all the fit

    Returns
    -------
    params : lmfit.Parameters
        Fitted parameter values
    freq : numpy.ndarray
        Array of frequency values within the fitted range
    S11: numpy.ndarray
        Array of complex signal values within the fitted range
    finalresult : lmfit.MinimizerResult
        The full minimizer result object (see lmfit documentation for details)

    """
    # Convert frequency and S11 into arrays
    frec = np.array(frec)
    S11 = np.array(S11)

    # Smooth data for initial guesses
    if margin == None or margin == 0:  # If no smooting desired, pass None as margin
        margin = 0
        sReS11 = S11.real
        sImS11 = S11.imag
        sS11 = np.array([x+1j*y for x, y in zip(sReS11, sImS11)])
    elif type(margin) != int or margin % 2 == 0 or margin <= 3:
        raise ValueError(
            'margin has to be either None, 0, or an odd integer larger than 3')
    else:
        sReS11 = np.array(smooth(S11.real, margin, 3))
        sImS11 = np.array(smooth(S11.imag, margin, 3))
        sS11 = np.array([x+1j*y for x, y in zip(sReS11, sImS11)])
    # Make smoothed phase vector removing 2pi jumps
    sArgS11 = np.angle(sS11)
    sArgS11 = np.unwrap(sArgS11)
    sdiffang = np.diff(sArgS11)

    # sdiffang = [ x if np.abs(x)<pi else -pi for x in np.diff(np.angle(sS11)) ]
    # Get resonance index from maximum of the derivative of the imaginary part
    # ires = np.argmax(np.abs(np.diff(sImS11)))
    # f0i=frec[ires]

    # Get resonance index from maximum of the derivative of the phase
    avgph = np.average(sdiffang)
    errvec = [np.power(x-avgph, 2.) for x in sdiffang]
    # ires = np.argmax(np.abs(sdiffang[margin:-margin]))+margin
    if margin == 0:
        ires = np.argmax(errvec[0:-1])+0
    else:
        ires = np.argmax(errvec[margin:-margin])+margin
    f0i = frec[ires]
    print("Max index: ", ires, " Max frequency: ", f0i)

    if doplots:
        plt.clf()
        plt.title('Original signal (Re,Im)')
        plt.plot(frec, S11.real)
        plt.plot(frec, S11.imag)
        plt.axis('auto')
        plt.show()

        plt.plot(np.angle(sS11))
        plt.title('Smoothed Phase')
        plt.axis('auto')
        plt.show()
        if margin == 0:
            plt.plot(sdiffang[0:-1])
        else:
            plt.plot(sdiffang[margin:-margin])
        plt.plot(sdiffang)
        plt.title('Diff of Smoothed Phase')
        plt.show()

    # Get peak width by finding width of spike in diffphase plot
    (imin, imax) = getwidth_phase(ires, errvec, margin)
    di = imax-imin
    print("Peak limits: ", imin, imax)
    print("Lower edge: ", frec[imin], " Center: ", frec[ires],
          " Upper edge: ", frec[imax], " Points in width: ", di)

    if doplots:
        plt.title('Smoothed (ph-phavg)\^2')
        plt.plot(errvec)
        plt.plot([imin], [errvec[imin]], 'ro')
        plt.plot([imax], [errvec[imax]], 'ro')
        plt.show()

    if not fitwidth == None:
        i1 = max(int(ires-di*fitwidth), 0)
        i2 = min(int(ires+di*fitwidth), len(frec))
        frec = frec[i1:i2]
        S11 = S11[i1:i2]
        ires = ires - i1
        imin = imin - i1
        imax = imax - i1

    # Trim peak from data (trimwidth times the width)
    (backfrec, backsig) = trim(frec, S11, ires-trimwidth*di, ires+trimwidth*di)

    if doplots:
        plt.title('Trimmed signal for background fit (Re,Im)')
        plt.plot(backfrec, backsig.real, backfrec, backsig.imag)
        plt.plot(frec, S11.real, frec, S11.imag)
        plt.show()

        plt.title('Trimmed signal for background fit (Abs)')
        plt.plot(backfrec, np.abs(backsig))
        plt.plot(frec, np.abs(S11))
        plt.show()

        plt.title('Trimmed signal for background fit (Phase)')
        plt.plot(backfrec, np.angle(backsig))
        plt.plot(frec, np.angle(S11))
        plt.show()

    if fitbackground:
        # Make initial background guesses
        b0 = (np.abs(sS11)[-1] - np.abs(sS11)[0])/(frec[-1]-frec[0])
    #    a0 = np.abs(sS11)[0] - b0*frec[0]
        a0 = np.average(np.abs(sS11)) - b0*backfrec[0]
    #    a0 = np.abs(sS11)[0] - b0*backfrec[0]
        c0 = 0.
    #    bp0 = ( np.angle(sS11[di])-np.angle(sS11[0]) )/(frec[di]-frec[0])
        xx = []
        for i in range(0, len(backfrec)-1):
            df = backfrec[i+1]-backfrec[i]
            dtheta = np.angle(backsig[i+1])-np.angle(backsig[i])
            if (np.abs(dtheta) > pi):
                continue
            xx.append(dtheta/df)
        # Remove infinite values in xx (from repeated frequency points for example)
        xx = np.array(xx)
        idx = np.isfinite(xx)
        xx = xx[idx]

    #    bp0 = np.average([ x if np.abs(x)<pi else 0 for x in np.diff(np.angle(backsig))] )/(frec[1]-frec[0])
        bp0 = np.average(xx)
    #   ap0 = np.angle(sS11[0]) - bp0*frec[0]
    #   ap0 = np.average(np.unwrap(np.angle(backsig))) - bp0*backfrec[0]
        ap0 = np.unwrap(np.angle(backsig))[0] - bp0*backfrec[0]
        cp0 = 0.
        print(a0, b0, ap0, bp0)
    else:
        a0 = 0
        b0 = 0
        c0 = 0
        ap0 = 0
        bp0 = 0
        cp0 = 0

    params = Parameters()
    myvary = True
    params.add('a', value=a0, vary=myvary)
    params.add('b', value=b0, vary=myvary)
    params.add('c', value=c0, vary=myvary)
    params.add('ap', value=ap0, vary=myvary)
    params.add('bp', value=bp0, vary=myvary)
    params.add('cp', value=cp0, vary=myvary)

    if not fitbackground:
        if ftype == 'A' or ftype == 'B':
            params['a'].set(value=-1, vary=False)
        elif ftype == '-A' or ftype == '-B' or ftype == 'X':
            params['a'].set(value=1, vary=False)
        params['b'].set(value=0, vary=False)
        params['c'].set(value=0, vary=False)
        params['ap'].set(value=0, vary=False)
        params['bp'].set(value=0, vary=False)
        params['cp'].set(value=0, vary=False)
    elif not refitback and oldpars != None:
        params['a'].set(value=oldpars['a'].value, vary=False)
        params['b'].set(value=oldpars['b'].value, vary=False)
        params['c'].set(value=oldpars['c'].value, vary=False)
        params['ap'].set(value=oldpars['ap'].value, vary=False)
        params['bp'].set(value=oldpars['bp'].value, vary=False)
        params['cp'].set(value=oldpars['cp'].value, vary=False)

# do background fit

    params['cp'].set(value=0., vary=False)
    result = minimize(background2min, params, args=(backfrec, backsig))
    '''
    params = result.params
    params['a'].set(vary=False)
    params['b'].set(vary=False)
    params['c'].set(vary=False)
    params['ap'].set(vary=False)
    params['bp'].set(vary=False)
    params['cp'].set(vary=True)
    result = minimize(background2min, params, args=(backfrec, backsig))

    params = result.params
    params['a'].set(vary=True)
    params['b'].set(vary=True)
    params['c'].set(vary=True)
    params['ap'].set(vary=True)
    params['bp'].set(vary=True)
    params['cp'].set(vary=True)
    result = minimize(background2min, params, args=(backfrec, backsig))
    '''

# write error report
    report_fit(result.params)

# calculate final background and remove background from original data
    complexresidual = un_realimag(result.residual)
    # backgroundfit = backsig + complexresidual
    fullbackground = np.array([backmodel(xx, result.params) for xx in frec])
    S11corr = -S11 / fullbackground
    if ftype == '-A' or ftype == '-B':
        S11corr = -S11corr

    if doplots:
        plt.title('Signal and fitted background (Re,Im)')
        plt.plot(frec, S11.real)
        plt.plot(frec, S11.imag)
        plt.plot(frec, fullbackground.real)
        plt.plot(frec, fullbackground.imag)
        plt.show()

        plt.title('Signal and fitted background (Phase)')
        plt.plot(frec, np.angle(S11))
        plt.plot(frec, np.angle(fullbackground))
        plt.show()

        plt.title('Signal and fitted background (Polar)')
        plt.plot(S11.real, S11.imag)
        plt.plot(fullbackground.real, fullbackground.imag)
        plt.show()

        plt.title('Signal with background removed (Re,Im)')
        plt.plot(frec, S11corr.real)
        plt.plot(frec, S11corr.imag)
        plt.show()

        plt.title('Signal with background removed (Phase)')
        ph = np.unwrap(np.angle(S11corr))
        plt.plot(frec, ph)
        plt.show()

        plt.title('Signal with background removed (Polar)')
        plt.plot(S11corr.real, S11corr.imag)
        plt.show()

    ktot = np.abs(frec[imax]-frec[imin])
    if ftype == 'A':
        Tres = np.abs(S11corr[ires]+1)
        kext0 = ktot*Tres/2.
    elif ftype == '-A':
        Tres = np.abs(1-S11corr[ires])
        kext0 = ktot*Tres/2.
    elif ftype == '-B':
        Tres = np.abs(S11corr[ires])
        kext0 = (1-Tres)*ktot
    elif ftype == 'B':
        Tres = np.abs(S11corr[ires])
        kext0 = (1+Tres)*ktot
    elif ftype == 'X':
        Tres = np.abs(S11corr[ires])
        kext0 = (1-Tres)*ktot
    kint0 = ktot-kext0
    if kint0 <= 0.:
        kint0 = kext0
    Qint0 = f0i/kint0
    Qext0 = f0i/kext0

# Make new parameter object (includes previous fitted background values)
    params = result.params
    if reusefitpars and oldpars != None:
        params.add('Qint', value=oldpars['Qint'].value, vary=True, min=0)
        params.add('Qext', value=oldpars['Qext'].value, vary=True, min=0)
        params.add('f0', value=oldpars['f0'].value, vary=True)
        params.add('theta', value=oldpars['theta'].value, vary=True)
    else:
        params.add('Qint', value=Qint0, vary=True, min=0)
        params.add('Qext', value=Qext0, vary=True, min=0)
        params.add('f0', value=f0i, vary=True)
        params.add('theta', value=0, vary=True)
    params['a'].set(vary=False)
    params['b'].set(vary=False)
    params['c'].set(vary=False)
    params['ap'].set(vary=False)
    params['bp'].set(vary=False)
    params['cp'].set(vary=False)

# Do final fit
    finalresult = minimize(S11residual, params, args=(frec, S11, ftype))
# write error report
    report_fit(finalresult.params)
    params = finalresult.params
    try:
        print('QLoaded = ', 1 /
              (1./params['Qint'].value+1./params['Qext'].value))
    except ZeroDivisionError:
        print('QLoaded = ', 0.)

    if doplots:

        plt.title('Pre-Final signal and fit (Re,Im)')
        plt.plot(frec, S11corr.real)
        plt.plot(frec, S11corr.imag)
        plt.plot(frec, S11theo(frec, params, ftype).real)
        plt.plot(frec, S11theo(frec, params, ftype).imag)
        plt.show()

        plt.title('Pre-Final signal and fit (Polar)')
        plt.plot(S11.real, S11.imag)
        plt.plot(S11full(frec, params, ftype).real,
                 S11full(frec, params, ftype).imag)
        plt.gca().set_aspect('equal', 'datalim')
        plt.show()

# REDO final fit varying all parameters
    if refitback and fitbackground:
        params['a'].set(vary=True)
        params['b'].set(vary=True)
        params['c'].set(vary=True)
        params['ap'].set(vary=True)
        params['bp'].set(vary=True)
        params['cp'].set(vary=False)
        finalresult = minimize(S11residual, params, args=(frec, S11, ftype))

# write error report
        report_fit(finalresult.params)
        params = finalresult.params
        try:
            print('QLoaded = ', 1 /
                  (1./params['Qint'].value+1./params['Qext'].value))
        except ZeroDivisionError:
            print('QLoaded = ', 0.)


# calculate final result and background
    complexresidual = un_realimag(finalresult.residual)
    finalfit = S11 + complexresidual
    # newbackground = np.array([backmodel(xx,finalresult.params) for xx in frec])

    if doplots:
        plt.title('Final signal and fit (Re,Im)')
        plt.plot(frec, S11.real)
        plt.plot(frec, S11.imag)
        plt.plot(frec, finalfit.real)
        plt.plot(frec, finalfit.imag)
        plt.show()

        plt.title('Final signal and fit (Polar)')
        plt.plot(S11.real, S11.imag)
        plt.plot(finalfit.real, finalfit.imag)
        plt.gca().set_aspect('equal', 'datalim')
        plt.show()

        plt.title('Final signal and fit (Abs)')
        plt.plot(frec, np.abs(S11))
        plt.plot(frec, np.abs(finalfit))
        plt.show()

    # chi2 = finalresult.chisqr

    return params, frec, S11, finalresult
