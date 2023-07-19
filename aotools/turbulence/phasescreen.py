"""
Finite Phase Screens
--------------------

Creation of phase screens of a defined size with Von Karmen Statistics.

"""

import numpy
from numpy import fft
import time
import random


def ft_sh_phase_screen(r0, N, delta, L0, l0, FFT=None, seed=None):

    """
    Creates a random phase screen with Von Karmen statistics with added
    sub-harmonics to augment tip-tilt modes.
    (Schmidt 2010)

    .. note::
        The phase screen is returned as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)

    Args:
        r0 (float): r0 parameter of scrn in metres
        N (int): Size of phase scrn in pxls
        delta (float): size in Metres of each pxl
        L0 (float): Size of outer-scale in metres
        l0 (float): inner scale in metres
        seed (int, optional): seed for random number generator. If provided, 
            allows for deterministic screens  

    Returns:
        ndarray: numpy array representing phase screen in radians
    """
    R = numpy.random.default_rng(seed)

    D = N * delta
    # high-frequency screen from FFT method
    phs_hi = ft_phase_screen(r0, N, delta, L0, l0, FFT, seed=seed)

    # spatial grid [m]
    coords = numpy.arange(-N/2,N/2)*delta
    x, y = numpy.meshgrid(coords,coords)

    # initialize low-freq screen
    phs_lo = numpy.zeros(phs_hi.shape)

    # loop over frequency grids with spacing 1/(3^p*L)
    for p in range(1,4):
        # setup the PSD
        del_f = 1 / (3**p*D) #frequency grid spacing [1/m]
        fx = numpy.arange(-1,2) * del_f

        # frequency grid [1/m]
        fx, fy = numpy.meshgrid(fx,fx)
        f = numpy.sqrt(fx**2 +  fy**2) # polar grid

        fm = 5.92/l0/(2*numpy.pi) # inner scale frequency [1/m]
        f0 = 1./L0

        # outer scale frequency [1/m]
        # modified von Karman atmospheric phase PSD
        PSD_phi = (0.023*r0**(-5./3)
                    * numpy.exp(-1*(f/fm)**2) / ((f**2 + f0**2)**(11./6)) )
        PSD_phi[1,1] = 0

        # random draws of Fourier coefficients
        cn = ( (R.normal(size=(3,3))
            + 1j*R.normal(size=(3,3)) )
                        * numpy.sqrt(PSD_phi)*del_f )
        SH = numpy.zeros((N,N),dtype="complex")
        # loop over frequencies on this grid
        for i in range(0, 3):
            for j in range(0, 3):

                SH += cn[i,j] * numpy.exp(1j*2*numpy.pi*(fx[i,j]*x+fy[i,j]*y))

        phs_lo = phs_lo + SH
        # accumulate subharmonics

    phs_lo = phs_lo.real - phs_lo.real.mean()

    return phs_lo+phs_hi


def ft_phase_screen(r0, N, delta, L0, l0, FFT=None, seed=None):
    """
    Creates a random phase screen with Von Karmen statistics.
    (Schmidt 2010)
    
    Parameters:
        r0 (float): r0 parameter of scrn in metres
        N (int): Size of phase scrn in pxls
        delta (float): size in Metres of each pxl
        L0 (float): Size of outer-scale in metres
        l0 (float): inner scale in metres
        seed (int, optional): seed for random number generator. If provided, 
            allows for deterministic screens  

    .. note::
        The phase screen is returned as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)

    Returns:
        ndarray: numpy array representing phase screen in radians
    """
    delta = float(delta)
    r0 = float(r0)
    L0 = float(L0)
    l0 = float(l0)

    R = numpy.random.default_rng(seed)

    del_f = 1./(N*delta)

    fx = numpy.arange(-N/2., N/2.) * del_f

    (fx, fy) = numpy.meshgrid(fx,fx)
    f = numpy.sqrt(fx**2. + fy**2.)

    fm = 5.92/l0/(2*numpy.pi)
    f0 = 1./L0

    PSD_phi = (0.023*r0**(-5./3.) * numpy.exp(-1*((f/fm)**2)) / (((f**2) + (f0**2))**(11./6)))

    PSD_phi[int(N/2), int(N/2)] = 0

    cn = ((R.normal(size=(N, N))+1j * R.normal(size=(N, N))) * numpy.sqrt(PSD_phi)*del_f)

    return ift2(cn, 1, FFT).real


def ift2(G, delta_f, FFT=None):
    """
    Wrapper for inverse fourier transform

    Parameters:
        G: data to transform
        delta_f: pixel seperation
        FFT (FFT object, optional): An accelerated FFT object
    """

    N = G.shape[0]

    return (
        numpy.fft.fftshift(FFT(numpy.fft.fftshift(G))) * (N * delta_f) ** 2
        if FFT
        else fft.ifftshift(fft.ifft2(fft.fftshift(G))) * (N * delta_f) ** 2
    )
