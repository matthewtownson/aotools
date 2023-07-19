"""
Module containing useful FFT based function and classes
"""
import numpy


def ft(data, delta):
    """
    A properly scaled 1-D FFT

    Parameters:
        data (ndarray): An array on which to perform the FFT
        delta (float): Spacing between elements

    Returns:
        ndarray: scaled FFT
    """
    return (
        numpy.fft.fftshift(
            numpy.fft.fft(numpy.fft.fftshift(data, axes=(-1))), axes=(-1)
        )
        * delta
    )

def ift(data, delta_f):
    """
    Scaled inverse 1-D FFT

    Parameters:
        data (ndarray): Data in Fourier Space to transform
        delta_f (ndarray): Frequency spacing of grid

    Returns:
        ndarray: Scaled data in real space
    """

    return (
        numpy.fft.ifftshift(
            numpy.fft.ifft(numpy.fft.ifftshift(data, axes=(-1))), axes=(-1)
        )
        * data.shape[-1]
        * delta_f
    )


def ft2(data, delta):
    """
    A properly scaled 2-D FFT

    Parameters:
        data (ndarray): An array on which to perform the FFT
        delta (float): Spacing between elements

    Returns:
        ndarray: scaled FFT
    """

    return (
        numpy.fft.fftshift(
            numpy.fft.fft2(numpy.fft.fftshift(data, axes=(-1, -2))),
            axes=(-1, -2),
        )
        * delta**2
    )

def ift2(data, delta_f):
    """
    Scaled inverse 2-D FFT

    Parameters:
        data (ndarray): Data in Fourier Space to transform
        delta_f (ndarray): Frequency spacing of grid

    Returns:
        ndarray: Scaled data in real space
    """
    N = data.shape[-1]
    return (
        numpy.fft.ifftshift(
            numpy.fft.ifft2(
                numpy.fft.ifftshift(data, axes=(-1, -2)), axes=(-1, -2)
            ),
            axes=(-1, -2),
        )
        * (N * delta_f) ** 2
    )

def rft(data, delta):
    """
    A properly scaled real 1-D FFT

    Parameters:
        data (ndarray): An array on which to perform the FFT
        delta (float): Spacing between elements

    Returns:
        ndarray: scaled FFT
    """
    return (
        numpy.fft.fftshift(
            numpy.fft.rfft(numpy.fft.fftshift(data, axes=(-1))), axes=(-1)
        )
        * delta
    )

def irft(data, delta_f):
    """
    Scaled real inverse 1-D FFT

    Parameters:
        data (ndarray): Data in Fourier Space to transform
        delta_f (ndarray): Frequency spacing of grid

    Returns:
        ndarray: Scaled data in real space
    """

    return (
        numpy.fft.ifftshift(
            numpy.fft.irfft(numpy.fft.ifftshift(data, axes=(-1))), axes=(-1)
        )
        * data.shape[-1]
        * delta_f
    )

def rft2(data, delta):
    """
    A properly scaled, real 2-D FFT

    Parameters:
        data (ndarray): An array on which to perform the FFT
        delta (float): Spacing between elements

    Returns:
        ndarray: scaled FFT
    """
    return (
        numpy.fft.fftshift(
            numpy.fft.rfft2(numpy.fft.fftshift(data, axes=(-1, -2))),
            axes=(-1, -2),
        )
        * delta**2
    )

def irft2(data, delta_f):
    """
    Scaled inverse real 2-D FFT

    Parameters:
        data (ndarray): Data in Fourier Space to transform
        delta_f (ndarray): Frequency spacing of grid

    Returns:
        ndarray: Scaled data in real space
    """
    N = data.shape[-1]
    return (
        numpy.fft.ifftshift(
            numpy.fft.irfft2(
                numpy.fft.ifftshift(data, axes=(-1, -2)), axes=(-1, -2)
            ),
            axes=(-1, -2),
        )
        * (N * delta_f) ** 2
    )
