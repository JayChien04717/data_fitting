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
