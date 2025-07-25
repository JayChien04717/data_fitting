import numpy as np


def avoided_crossing_direct_coupling(flux, f_center1, f_center2, c1, c2, g, flux_state=0):
    """Compute eigenfrequencies for a two-mode avoided crossing.

    Parameters
    ----------
    flux : array_like or float
        Flux bias value(s).
    f_center1 : float
        Center frequency of the first mode at ``flux_state``.
    f_center2 : float
        Center frequency of the second mode at ``flux_state``.
    c1 : float
        Linear coefficient for the first mode frequency shift ``f1 = f_center1 + c1*(flux - flux_state)``.
    c2 : float
        Linear coefficient for the second mode frequency shift ``f2 = f_center2 + c2*(flux - flux_state)``.
    g : float
        Direct coupling rate between the two modes.
    flux_state : float, optional
        Reference flux value. Default is 0.

    Returns
    -------
    numpy.ndarray
        Eigenfrequencies of the coupled system for each ``flux`` value.
    """

    flux = np.asarray(flux, dtype=float)
    if flux.ndim == 0:
        flux = flux[None]

    if not np.isscalar(f_center1) or not np.isscalar(f_center2):
        raise TypeError("f_center1 and f_center2 must be scalars")
    if not np.isscalar(c1) or not np.isscalar(c2) or not np.isscalar(g):
        raise TypeError("c1, c2 and g must be scalars")

    f1 = f_center1 + c1 * (flux - flux_state)
    f2 = f_center2 + c2 * (flux - flux_state)

    # Build an array of 2x2 matrices for each flux value
    mats = np.zeros((flux.size, 2, 2), dtype=float)
    mats[:, 0, 0] = f1
    mats[:, 1, 1] = f2
    mats[:, 0, 1] = mats[:, 1, 0] = g

    return np.linalg.eigvalsh(mats)
