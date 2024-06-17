import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cmath
from fitting import *
from resonator_tools import circuit
from addict import Dict
from plotfunc import plotmeas, plotfit, plotall

def reformdata(zval):
    data = Dict()
    data.idata = zval.real
    data.qdata = zval.imag
    data.mag = np.abs(zval)
    return data

def NormalizeData(data):
    data.idata = (data.idata - np.min(data.idata)) / (np.max(data.idata) - np.min(data.idata))
    data.qdata = (data.qdata - np.min(data.qdata)) / (np.max(data.qdata) - np.min(data.qdata))
    data.mag = (data.mag - np.min(data.mag)) / (np.max(data.mag) - np.min(data.mag))
    return data



def resonator_analyze(x, y):    
    fit,_ = fithanger(x, np.abs(y))
    pass

def resonator_analyze(x, y):    
    pass

def spectrum_analyze(x, y, plot=False):
    fitdata = Dict()
    data = reformdata(y)
    data.x = x
    fitdata.x = x

    try:
        mpOpt, mpCov = fitlor(x, data.mag)
        fitdata.mag = lorfunc(x, *mpOpt)
    except:
        fitdata.mag = np.zeros(len(fitdata.x))
    try:
        ipOpt, ipCov = fitlor(x, data.idata)
        fitdata.idata = lorfunc(x, *ipOpt)
    except:
        fitdata.idata = np.zeros(len(fitdata.x))
    try:
        qpOpt, qpCov = fitlor(x, data.qdata)
        fitdata.qdata = lorfunc(x, *qpOpt)
    except:
        fitdata.qdata = np.zeros(len(fitdata.x))

    plotall(data, fitdata)


def amprabi_analyze(x, y, plot=False, normalize = False):
    fitdata = Dict()
    data = reformdata(y)
    data.x = x
    fitdata.x = x
    if normalize==True:
        data = NormalizeData(data)
    elif normalize==False:
        pass
    try:
        mpOpt, mpCov = fitdecaysin(x, data.mag)
        fitdata.mag = decaysin(x, *mpOpt)
    except:
        fitdata.mag = np.zeros(len(fitdata.x))
    try:
        ipOpt, ipCov = fitdecaysin(x, data.idata)
        fitdata.idata = decaysin(x, *ipOpt)
    except:
        fitdata.idata = np.zeros(len(fitdata.x))
    try:
        qpOpt, qpCov = fitdecaysin(x, data.qdata)
        fitdata.qdata = decaysin(x, *qpOpt)
    except:
        fitdata.qdata = np.zeros(len(fitdata.x))

    plt.figure(figsize=(8,6))
    plt.title(f'Amplitude Rabi',fontsize=15)
    plotall(data, fitdata)

    

    
def lengthraig_analyze(x, y, plot=False, normalize = False):
    fitdata = Dict()
    data = reformdata(y)
    data.x = x
    fitdata.x = x
    if normalize==True:
        data = NormalizeData(data)
    elif normalize==False:
        pass
    try:
        mpOpt, mpCov = fitdecaysin(x, data.mag)
        fitdata.mag = decaysin(x, *mpOpt)
    except:
        fitdata.mag = np.zeros(len(fitdata.x))
    try:
        ipOpt, ipCov = fitdecaysin(x, data.idata)
        fitdata.idata = decaysin(x, *ipOpt)
    except:
        fitdata.idata = np.zeros(len(fitdata.x))
    try:
        qpOpt, qpCov = fitdecaysin(x, data.qdata)
        fitdata.qdata = decaysin(x, *qpOpt)
    except:
        fitdata.qdata = np.zeros(len(fitdata.x))

    plt.figure(figsize=(8,6))
    plt.title(f'Length Rabi',fontsize=15)
    plotall(data, fitdata)

def T1_analyze(x, y, plot=False, normalize = False):
    fitdata = Dict()
    data = reformdata(y)
    data.x = x
    fitdata.x = x
    if normalize==True:
        data = NormalizeData(data)
    elif normalize==False:
        pass
    try:
        mpOpt, mpCov = fitexp(x, data.mag)
        fitdata.mag = expfunc(x, *mpOpt)
    except:
        fitdata.mag = np.zeros(len(fitdata.x))
    try:
        ipOpt, ipCov = fitexp(x, data.idata)
        fitdata.idata = expfunc(x, *ipOpt)
    except:
        fitdata.idata = np.zeros(len(fitdata.x))
    try:
        qpOpt, qpCov = fitexp(x, data.qdata)
        fitdata.qdata = expfunc(x, *qpOpt)
    except:
        fitdata.qdata = np.zeros(len(fitdata.x))

    plt.figure(figsize=(8,6))
    plt.title(f'T1',fontsize=15)
    plotall(data, fitdata)

def T2r_analyze(x, y, plot=False, normalize = False):
    fitdata = Dict()
    data = reformdata(y)
    data.x = x
    fitdata.x = x
    if normalize==True:
        data = NormalizeData(data)
    elif normalize==False:
        pass
    try:
        mpOpt, mpCov = fitdecaysin(x, data.mag)
        fitdata.mag = decaysin(x, *mpOpt)
    except:
        fitdata.mag = np.zeros(len(fitdata.x))
    try:
        ipOpt, ipCov = fitdecaysin(x, data.idata)
        fitdata.idata = decaysin(x, *ipOpt)
    except:
        fitdata.idata = np.zeros(len(fitdata.x))
    try:
        qpOpt, qpCov = fitdecaysin(x, data.qdata)
        fitdata.qdata = decaysin(x, *qpOpt)
    except:
        fitdata.qdata = np.zeros(len(fitdata.x))

    plt.figure(figsize=(8,6))
    plt.title(f'T2 Ramsey',fontsize=15)
    plotall(data, fitdata)


def T2e_analyze(x, y, plot=False, normalize = False):
    fitdata = Dict()
    data = reformdata(y)
    data.x = x
    fitdata.x = x
    if normalize==True:
        data = NormalizeData(data)
    elif normalize==False:
        pass
    try:
        mpOpt, mpCov = fitexp(x, data.mag)
        fitdata.mag = expfunc(x, *mpOpt)
    except:
        fitdata.mag = np.zeros(len(fitdata.x))
    try:
        ipOpt, ipCov = fitexp(x, data.idata)
        fitdata.idata = expfunc(x, *ipOpt)
    except:
        fitdata.idata = np.zeros(len(fitdata.x))
    try:
        qpOpt, qpCov = fitexp(x, data.qdata)
        fitdata.qdata = expfunc(x, *qpOpt)
    except:
        fitdata.qdata = np.zeros(len(fitdata.x))


    plotall(data, fitdata)


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
    fr = Labber.LogFile(res)
    f1 = Labber.LogFile(lenghrabi) 
    f2 = Labber.LogFile(t1) 
    f3 = Labber.LogFile(t2) 
    (x, y) = fr.getTraceXY()
    (sx, sy) = spec.getTraceXY()
    (rx,ry) = f1.getTraceXY() 
    (t1x,t1y) = f2.getTraceXY() 
    (t2x,t2y) = f3.getTraceXY() 
    amprabi_analyze(rx, ry, plot=True)


