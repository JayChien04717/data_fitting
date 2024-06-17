import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cmath
from fitting import *
from resonator_tools import circuit
from addict import Dict

def reformdata(zval):
    data = Dict()
    data.idata = zval.real
    data.qdata = zval.imag
    data.mag = np.abs(zval)
    return data

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
    ydata = np.real(rotate_complex(ydata,rotation_angle))
    return ydata

def NormalizeData2(data):
    norm_data = Dict()
    norm_data.idata = (data.idata - np.min(data.idata)) / (np.max(data.idata) - np.min(data.idata))
    norm_data.qdata = (data.qdata - np.min(data.qdata)) / (np.max(data.qdata) - np.min(data.qdata))
    norm_data.mag = (data.mag - np.min(data.mag)) / (np.max(data.mag) - np.min(data.mag))
    return norm_data



def resonator_analyze(x, y):    
    fit,_ = fithanger(x, np.abs(y))
    
    pass

def resonator_analyze(x, y):    
    
    
    pass

def spectrum_analyze(x, y, plot=False):
    y = post_rotate(y)
    pOpt, pCov = fitlor(x, y)

    if plot==True:
        plt.title(f'mag.',fontsize=15)
        plt.plot(x, y, label = 'mag', marker='o', markersize=3)
        plt.plot(x, lorfunc(x, *pOpt), label = 'fit')
        res = pOpt[2]
        plt.axvline(res, color='r', ls='--', label=f'$f_res$ = {res/1e9:.2f}')
        plt.legend()
        plt.show()


def amprabi_analyze(x, y, plot=False, normalize = False):
    if normalize==True:
        y = NormalizeData(np.abs(y))
    elif normalize==False:
        y = np.abs(y)
    pOpt, pCov = fitdecaysin(x, y)
    sim = decaysin(x, *pOpt)

    if plot==True:
        plt.plot(x, y, label = 'meas', marker='o', markersize=3)
        plt.plot(x, sim, label = 'fit')
        plt.title(f'Amp Rabi',fontsize=15)
        plt.xlabel('$t\ (us)$',fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.show()

    
def lengthraig_analyze(x, y, plot=False, normalize = False):
    if normalize==True:
        y = NormalizeData(np.abs(y))
    elif normalize==False:
        y = np.abs(y)
    pOpt, pCov = fitdecaysin(x, y)
    sim = decaysin(x, *pOpt)
    if plot==True:
        plt.plot(x, y, label = 'meas', marker='o', markersize=3)
        plt.plot(x, sim, label = 'fit')
        plt.title(f'Length Rabi',fontsize=15)
        plt.xlabel('$t\ (us)$',fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.show()

def T1_analyze(x, y, plot=False, normalize = False):
    if normalize==True:
        y = NormalizeData(np.abs(y))
    elif normalize==False:
        y = np.abs(y)
    pOpt, pCov = fitexp(x, y)
    sim = expfunc(x, *pOpt)
    if plot==True:
        plt.plot(x, y, label = 'meas', marker='o', markersize=3)
        plt.plot(x, sim, label = 'fit')
        plt.title(f'T1 = {pOpt[3]:.2f}$\mu s$',fontsize=15)
        plt.xlabel('$t\ (\mu s)$',fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.show()

def T2r_analyze(x, y, plot=False, normalize = False):
    if normalize==True:
        y = NormalizeData(np.abs(y))
    elif normalize==False:
        y = np.abs(y)
    pOpt, pCov = fitdecaysin(x, y)
    sim = decaysin(x, *pOpt)

    if plot==True:
        plt.plot(x, y, label = 'meas', marker='o', markersize=3)
        plt.plot(x, sim, label = 'fit')
        plt.title(f'T2r = {pOpt[3]:.2f}$\mu s$',fontsize=15)
        plt.xlabel('$t\ (\mu s)$',fontsize=15)
        # plt.ylabel('Population',fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.show()


def T2e_analyze(x, y, plot=False, normalize = False):
    if normalize==True:
        y = NormalizeData(np.abs(y))
    elif normalize==False:
        y = np.abs(y)
    pOpt, pCov = fitexp(x, y)
    sim = expfunc(x, *pOpt)
    if plot==True:
        plt.plot(x, y, label = 'meas', marker='o', markersize=3)
        plt.plot(x, sim, label = 'fit')
        plt.title(f'T2e = {pOpt[3]:.2f}$\mu s$',fontsize=15)
        plt.xlabel('$t\ (\mu s)$',fontsize=15)
        # plt.ylabel('Population',fontsize=15)
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
    res = r'./r1_cal.hdf5'
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
    (x, y) = fr.getTraceXY()
    (sx, sy) = spec.getTraceXY()
    (rx,ry) = f1.getTraceXY() 
    (t1x,t1y) = f2.getTraceXY() 
    (t2x,t2y) = f3.getTraceXY() 
    # spectrum_analyze(sx, sy, plot=True)
