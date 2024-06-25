import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cmath
from fitting import *
# from resonator_tools import circuit
from addict import Dict

figsize = (8,6)

def NormalizeData(zval):
    return (zval - np.min(zval)) / (np.max(zval) - np.min(zval))

def post_rotate(ydata):
    from scipy.optimize import minimize_scalar
    def rotate_complex(iq, angle):
        return (iq) * np.exp(1j * np.pi * angle/180)

    def std_q(y, rot_agl_):
        iq = rotate_complex(y, rot_agl_)
        return np.std(iq.imag)
    res = minimize_scalar(lambda agl:std_q(ydata, agl), bounds=[0, 360])
    rotation_angle = res.x
    ydata = (rotate_complex(ydata,rotation_angle))
    return ydata


def resonator_analyze(x, y):    
    from resonator_tools import circuit

    port1 = circuit.notch_port()
    port1.add_data(x,y)
    port1.autofit()
    print("Fit results:", port1.fitresults)
    port1.plotall()
    pass

def resonator_analyze2(x, y, plot=False):    
    y = np.abs(y)
    pOpt, pCov = fit_asym_lor(x, y)

    if plot==True:
        plt.figure(figsize=figsize)
        plt.title(f'mag.',fontsize=15)
        plt.plot(x, y, label = 'mag', marker='o', markersize=3)
        plt.plot(x, asym_lorfunc(x, *pOpt), label = 'fit')
        res = pOpt[2]
        plt.axvline(res, color='r', ls='--', label=f'$f_res$ = {res/1e9:.2f}')
        plt.legend()
        plt.show()
        return res


def spectrum_analyze(x, y, plot=False):
    y = np.abs(y)
    pOpt, pCov = fitlor(x, y)

    if plot==True:
        plt.figure(figsize=figsize)
        plt.title(f'mag.',fontsize=15)
        plt.plot(x, y, label = 'mag', marker='o', markersize=3)
        plt.plot(x, lorfunc(x, *pOpt), label = 'fit')
        res = pOpt[2]
        plt.axvline(res, color='r', ls='--', label=f'$f_res$ = {res/1e9:.2f}')
        plt.legend()
        plt.show()
        return res

def amprabi_analyze(x, y, plot=False, normalize = False):
    if normalize==True:
        y = -np.abs(y)
        y = NormalizeData(y)
    elif normalize==False:
        y = np.abs(y)

    pOpt, pCov = fitdecaysin(x, y)
    sim = decaysin(x, *pOpt)
    pi = round(x[np.argmax(sim)], 1)
    pi2 = round(x[round((np.argmin(sim) + np.argmax(sim))/2)], 1)

    if pOpt[2] > 180: pOpt[2] = pOpt[2] - 360
    elif pOpt[2] < -180: p[2] = pOpt[2] + 360
    if pOpt[2] < 0: 
        pi_gain = (1/2 - pOpt[2]/180)/2/pOpt[1]
        pi2_gain = (0 - pOpt[2]/180)/2/pOpt[1]
    else: 
        pi_gain= (3/2 - pOpt[2]/180)/2/pOpt[1]
        pi2_gain = (1 - pOpt[2]/180)/2/pOpt[1]


    if plot==True:
        plt.figure(figsize=figsize)
        plt.plot(x, y, label = 'meas', ls='None', marker='o', markersize=3)
        plt.plot(x, sim, label = 'fit')
        plt.title(f'Amplitude Rabi',fontsize=15)
        plt.xlabel('$t\ (us)$',fontsize=15)
        if normalize==True:
            plt.ylabel('Population',fontsize=15)
            plt.axvline(pi, ls='--', c='red', label=f'$\pi$ gain={pi}')
            plt.axvline(pi2, ls='--', c='red', label=f'$\pi/2$ gain={pi2}')
        else:
            plt.axvline(pi_gain, ls='--', c='red', label=f'$\pi$ gain={pi_gain:.1f}')
            plt.axvline(pi2_gain//2, ls='--', c='red', label=f'$\pi$ gain={(pi_gain//2):.1f}')
        plt.legend()
        plt.tight_layout()
        plt.show()
    return pi_gain, pi2_gain, max(y)-min(y)
    
def lengthraig_analyze(x, y, plot=False, normalize = False):
    if normalize==True:
        y = -np.abs(y)
        y = NormalizeData(y)
    elif normalize==False:
        y = np.abs(y)

    pOpt, pCov = fitdecaysin(x, y)
    sim = decaysin(x, *pOpt)
    pi = round(x[np.argmax(sim)], 1)
    pi2 = round(x[round((np.argmin(sim) + np.argmax(sim))/2)], 1)
    if pOpt[2] > 180: pOpt[2] = pOpt[2] - 360
    elif pOpt[2] < -180: p[2] = pOpt[2] + 360
    if pOpt[2] < 0: 
        pi_length = (1/2 - pOpt[2]/180)/2/pOpt[1]
        pi2_length = (0 - pOpt[2]/180)/2/pOpt[1]
    else: 
        pi_length= (3/2 - pOpt[2]/180)/2/pOpt[1]
        pi2_length = (1 - pOpt[2]/180)/2/pOpt[1]

    if plot==True:
        plt.figure(figsize=figsize)
        plt.plot(x, y, label = 'meas', ls='None', marker='o', markersize=3)
        plt.plot(x, sim, label = 'fit')
        plt.title(f'Length Rabi',fontsize=15)
        plt.xlabel('$t\ (us)$',fontsize=15)
        if normalize==True:
            plt.ylabel('Population',fontsize=15)
            plt.axvline(pi, ls='--', c='red', label=f'$\pi$ len={pi}')
            plt.axvline(pi2, ls='--', c='red', label=f'$\pi/2$ len={pi2}')   
        else:
            plt.axvline(pi_length, ls='--', c='red', label=f'$\pi$ gain={pi_length:.1f}')
            plt.axvline(pi2_length, ls='--', c='red', label=f'$\pi$ gain={pi2_length:.1f}')
        plt.legend()
        plt.tight_layout()
        plt.show()

def T1_analyze(x, y, plot=False, normalize = False):
    if normalize==True:
        y = -np.abs(y)
        y = NormalizeData(y)
    elif normalize==False:
        y = np.abs(y)

    pOpt, pCov = fitexp(x, y)
    sim = expfunc(x, *pOpt)
    
    if plot==True:
        plt.figure(figsize=figsize)
        plt.plot(x, y, label = 'meas', ls='None', marker='o', markersize=3)
        plt.title(f'T1 = {pOpt[3]:.2f}$\mu s$',fontsize=15)
        plt.xlabel('$t\ (\mu s)$',fontsize=15)
        if normalize==True:
            plt.ylabel('Population',fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.show()

def T2r_analyze(x, y, plot=False, normalize = False):
    if normalize==True:
        y = -np.abs(y)
        y = NormalizeData(y)
    elif normalize==False:
        y = np.abs(y)

    pOpt, pCov = fitdecaysin(x, y)
    sim = decaysin(x, *pOpt)
    error = np.sqrt(np.diag(pCov))

    if plot==True:
        plt.figure(figsize=figsize)
        plt.plot(x, y, label = 'meas', ls='None', marker='o', markersize=3)
        plt.plot(x, sim, label = 'fit')
        plt.title(f'T2r = {pOpt[3]:.2f}$\mu s, detune = {pOpt[1]:.2f}MHz \pm {(error[1])*1e3:.2f}kHz$',fontsize=15)
        plt.xlabel('$t\ (\mu s)$',fontsize=15)
        if normalize==True:
            plt.ylabel('Population',fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.show()
    return pOpt[1]

def T2e_analyze(x, y, plot=False, normalize = False):
    if normalize==True:
        y = -np.abs(y)
        y = NormalizeData(y)
    elif normalize==False:
        y = np.abs(y)

    pOpt, pCov = fitexp(x, y)
    sim = expfunc(x, *pOpt)

    if plot==True:
        plt.figure(figsize=figsize)
        plt.plot(x, y, label = 'meas', ls='None', marker='o', markersize=3)
        plt.plot(x, sim, label = 'fit')
        plt.title(f'T2e = {pOpt[3]:.2f}$\mu s$',fontsize=15)
        plt.xlabel('$t\ (\mu s)$',fontsize=15)
        if normalize==True:
            plt.ylabel('Population',fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__=="__main__":
    import sys, os
    os.chdir(r'C:\Users\QEL\Desktop\fit_pack\data')
    sys.path.append(r'C:\Program Files\Keysight\Labber\Script')
    import Labber

    np.bool= bool
    np.float = np.float64
    res = r'Normalized_coupler_chen_054[7]_@7.819GHz_power_dp_002.hdf5'
    pdr = r'r1_pdr.hdf5'
    lenghrabi = r'q1_rabi.hdf5'
    t1= r'q1_T1_2.hdf5'
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
    (rx,ry) = f1.getTraceXY() 
    (t1x,t1y) = f2.getTraceXY() 
    (t2x,t2y) = f3.getTraceXY() 
    # spectrum_analyze(sx, sy, plot=True)
    lengthraig_analyze(rx, ry, plot=True, normalize=False)
    amprabi_analyze(rx, ry, plot=True, normalize=False)
    # T1_analyze(t1x, t1y, plot=True,normalize=True)
    # T2r_analyze(t2x, t2y, plot=True)
    # resonator_analyze(x,y)



    phase_deg = 10
    x = np.linspace(0, 3*np.pi, 1001)
    y = np.sin(2*np.pi*0.6*x + phase_deg*np.pi/180) * np.exp(-x)
    p, _ = fitdecaysin(x, y)
    print(p[2])
    if p[2] > 180: p[2] = p[2] - 360
    elif p[2] < -180: p[2] = p[2] + 360
    if p[2] < 0: 
        pi_length = (1/2 - p[2]/180)/2/p[1]
        pi2_length = (0 - p[2]/180)/2/p[1]
    else: 
        pi_length= (3/2 - p[2]/180)/2/p[1]
        pi2_length = (1 - p[2]/180)/2/p[1]
    sim = decaysin(x, *p)
    # plt.plot(x, y)
    # plt.plot(x, sim)
    # plt.axvline(pi_length, label = 'pi', c ='red')
    # plt.axvline(pi_length//2, label = 'pi//2', c = 'orange')
    # plt.axvline(pi2_length, label = 'pi2', c = 'green')
    # plt.legend()
    # plt.show()