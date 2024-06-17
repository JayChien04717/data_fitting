import matplotlib.pyplot as plt
import numpy as np
import sys, os
os.chdir(r'C:\Users\QEL\Desktop\fit_pack\data')
sys.path.append(r'C:\Program Files\Keysight\Labber\Script')
import Labber
from lmfit import create_params, minimize
np.bool= bool
np.float = np.float64
res = r'./r1_cal.hdf5'
fr = Labber.LogFile(res)
(x, y) = fr.getTraceXY()

fit_params = create_params(amp=dict(value=13, max=20, min=0),
                           period=dict(value=2, max=10),
                           shift=dict(value=0, max=np.pi/2., min=-np.pi/2.),
                           decay=dict(value=0.02, max=0.1, min=0))
print(fit_params['decay'])