"""
Slope Covariance Matrix Generation
----------------------------------

Slope covariance matrix routines for AO systems observing through Von Karmon turbulence. Such matrices have a variety
of uses, though they are especially useful for creating 'tomographic reconstructors' that can reconstruct some 'psuedo'
WFS measurements in a required direction (where there might be an interesting science target but no guide stars),
given some actual measurements in other directions (where the some suitable guide stars are).

.. warning::
    This code has been tested qualitatively and seems OK, but needs more rigorous testing.

.. codeauthor::
    Andrew Reeves <a.p.reeves@durham.ac.uk>

"""

import multiprocessing

import numpy
import scipy.special


class CovarianceMatrix(object):
    """
    A creator of slope covariance matrices in Von Karmon turbulence, based on the paper by Martin et al, SPIE, 2012.

    Given a list of paramters describing an AO WFS system and the atmosphere above the telescope, this class can
    compute the covariance matrix between all the WFS measurements. This can support LGS sources that exist at a
    finite altitude. When computing the covariance matrix, Python's multiprocessing module is used to spread the work
    between different processes and processing cores.

    On initialisation, this class performs some initial calculations and parameter sorting. To create the
    covariance matrix run the ``make_covariace_matrix`` method. This may take some time depending on your paramters...

    Parameters:
        n_wfs (int): Number of wavefront sensors present.
        pupil_masks (ndarray): A map of the pupil for each WFS which is nx_subaps by ny_subaps. 1 if subap active, 0 if not.
        telescope_diameter (float): Diameter of the telescope
        subap_diameters (ndarray): The diameter of the sub-apertures for each WFS in metres
        gs_altitudes (ndarray): Reciprocal (1/metres) of the Guide star alitude for each WFS
        gs_positions (ndarray): X,Y position of each WFS in arcsecs. Array shape (Wfs, 2)
        wfs_wavelengths (ndarray): Wavelength each WFS observes
        n_layers (int): The number of atmospheric turbulence layers
        layer_altitudes (ndarray): The altitude of each turbulence layer in meters
        layer_r0s (ndarray): The Fried parameter of each turbulence layer
        layer_L0s (ndarray): The outer-scale of each layer in metres
        threads (int, optional): Number of processes to use for calculation. default is 1
    """

    def __init__(
        self, n_wfs, pupil_masks, telescope_diameter, subap_diameters, gs_altitudes, gs_positions, wfs_wavelengths,
        n_layers, layer_altitudes, layer_r0s, layer_L0s, threads=1):


        self.threads = threads

        self.n_wfs = n_wfs
        self.subap_diameters = subap_diameters
        self.wfs_wavelengths = wfs_wavelengths
        self.telescope_diameter = telescope_diameter
        self.pupil_masks = pupil_masks
        self.gs_altitudes = gs_altitudes
        self.gs_positions = gs_positions

        self.n_layers = n_layers
        self.layer_altitudes = layer_altitudes
        self.layer_r0s = layer_r0s
        self.layer_L0s = layer_L0s
        # print("n_layers: {}".format(n_layers))

        self.n_subaps = []
        self.total_subaps = 0
        for wfs_n in range(n_wfs):
            self.n_subaps.append(pupil_masks[wfs_n].sum())
            self.total_subaps += self.n_subaps[wfs_n]
        self.n_subaps = numpy.array(self.n_subaps, dtype="int")
        self.total_subaps = int(self.total_subaps)


    def make_covariance_matrix(self):
        """
        Calculate and build the covariance matrix

        Returns:
            ndarray: Covariance Matrix
        """
        # make a list of the positions of the centre of every sub-aperture
        # for each WFS un units of metres from the centre of the pupil
        self.subap_positions = []
        for wfs_n in range(self.n_wfs):
            wfs_subap_pos = numpy.array(numpy.where(self.pupil_masks[wfs_n] == 1)).T * self.subap_diameters[wfs_n]
            wfs_subap_pos -= self.telescope_diameter/2.
            wfs_subap_pos -= self.subap_diameters[wfs_n]/2.

            self.subap_positions.append(wfs_subap_pos)
            # print("WFS {}, max position: {}m".format(wfs_n, abs(wfs_subap_pos).max()))

        # Create a list with n_layers elements, each of which is a list of the WFS meta-subap positions at that altitude
        self.subap_layer_positions = []
        self.subap_layer_diameters = []
        subap_n = 0
        for layer_altitude in self.layer_altitudes:
            wfs_pos = []
            wfs_subap_diameters = []
            for wfs_n in range(self.n_wfs):

                # Scale for LGS
                if self.gs_altitudes[wfs_n] != 0:
                    # print("Its an LGS!")
                    scale_factor = (1 - layer_altitude/self.gs_altitudes[wfs_n])
                    positions = scale_factor * self.subap_positions[wfs_n].copy()
                else:
                    scale_factor = 1
                    positions = self.subap_positions[wfs_n].copy()

                # translate due to GS position
                gs_pos_rad = numpy.array(self.gs_positions[wfs_n]) * numpy.pi/180/3600
                # print("GS Positions: {} rad".format(gs_pos_rad))
                translation = gs_pos_rad * layer_altitude

                # print("Translation: {} m".format(translation))
                # print("Max position before translation: {} m".format(abs(positions).max()))

                positions += translation
                # print("Max position: {} m".format(abs(positions).max()))


                wfs_pos.append(positions)
                wfs_subap_diameters.append(self.subap_diameters[wfs_n] * scale_factor)

            self.subap_layer_diameters.append(wfs_subap_diameters)
            self.subap_layer_positions.append(wfs_pos)


        if self.threads == 1:
            self._make_covariance_matrix()
        else:

            self._make_covariance_matrix_mp(self.threads)

        self.covariance_matrix = mirror_covariance_matrix(self.covariance_matrix)

        return self.covariance_matrix

    def _make_covariance_matrix(self):
        # Now compile the covariance matrix
        self.covariance_matrix = numpy.zeros((2 * self.total_subaps, 2 * self.total_subaps)).astype("float32")
        for layer_n in range(self.n_layers):
            # print("Compute Layer {}".format(layer_n))

            subap_ni = 0
            for wfs_i in range(self.n_wfs):
                subap_nj = 0
                # Only loop over upper diagonal of covariance matrix as its symmetrical
                for wfs_j in range(wfs_i+1):
                    cov_xx, cov_yy, cov_xy = wfs_covariance(
                            self.n_subaps[wfs_i], self.n_subaps[wfs_j],
                            self.subap_layer_positions[layer_n][wfs_i], self.subap_layer_positions[layer_n][wfs_j],
                            self.subap_layer_diameters[layer_n][wfs_i], self.subap_layer_diameters[layer_n][wfs_j],
                            self.layer_r0s[layer_n], self.layer_L0s[layer_n])

                    subap_ni = self.n_subaps[:wfs_i].sum()
                    subap_nj = self.n_subaps[:wfs_j].sum()

                    # Coordinates of the XX covariance
                    cov_mat_coord_x1 = subap_ni * 2
                    cov_mat_coord_x2 = subap_ni * 2 + self.n_subaps[wfs_i]

                    cov_mat_coord_y1 = subap_nj * 2
                    cov_mat_coord_y2 = subap_nj * 2 + self.n_subaps[wfs_j]

                    # print("covmat coords: ({}: {}, {}: {})".format(cov_mat_coord_x1, cov_mat_coord_x2, cov_mat_coord_y1, cov_mat_coord_y2))
                    r0_scale = ((self.wfs_wavelengths[wfs_i] * self.wfs_wavelengths[wfs_j])
                            / (8 * numpy.pi**2 * self.subap_layer_diameters[layer_n][wfs_i] * self.subap_layer_diameters[layer_n][wfs_j])
                                )

                    self.covariance_matrix[
                            cov_mat_coord_x1: cov_mat_coord_x2, cov_mat_coord_y1: cov_mat_coord_y2
                            ] += cov_xx * r0_scale
                    self.covariance_matrix[
                            cov_mat_coord_x1 + self.n_subaps[wfs_i]: cov_mat_coord_x2 + self.n_subaps[wfs_i],
                            cov_mat_coord_y1: cov_mat_coord_y2] += cov_xy * r0_scale
                    self.covariance_matrix[
                            cov_mat_coord_x1: cov_mat_coord_x2,
                            cov_mat_coord_y1 + self.n_subaps[wfs_j]: cov_mat_coord_y2 + self.n_subaps[wfs_j]
                            ] += numpy.fliplr(numpy.flipud(cov_xy)) * r0_scale
                    self.covariance_matrix[
                            cov_mat_coord_x1 + self.n_subaps[wfs_i]: cov_mat_coord_x2 + self.n_subaps[wfs_i],
                            cov_mat_coord_y1 + self.n_subaps[wfs_j]: cov_mat_coord_y2 + self.n_subaps[wfs_j]
                            ] += cov_yy * r0_scale

    def _make_covariance_matrix_mp(self, threads):
        pool = multiprocessing.Pool(threads)

        # Now compile the covariance matrix
        self.covariance_matrix = numpy.zeros((2 * self.total_subaps, 2 * self.total_subaps), dtype="float32")
        for layer_n in range(self.n_layers):
            # print("Compute Layer {}".format(layer_n))

            args = []
            for wfs_i in range(self.n_wfs):
                # Only loop over upper diagonal of covariance matrix as its symmetrical
                args.extend(
                    (
                        self.n_subaps[wfs_i],
                        self.n_subaps[wfs_j],
                        self.subap_layer_positions[layer_n][wfs_i],
                        self.subap_layer_positions[layer_n][wfs_j],
                        self.subap_layer_diameters[layer_n][wfs_i],
                        self.subap_layer_diameters[layer_n][wfs_j],
                        self.layer_r0s[layer_n],
                        self.layer_L0s[layer_n],
                    )
                    for wfs_j in range(wfs_i + 1)
                )
            self.cov_mats = pool.map(wfs_covariance_mpwrap, args)

            thread_n = 0
            for wfs_i in range(self.n_wfs):
                for wfs_j in range(wfs_i+1):
                    cov_xx, cov_yy, cov_xy = self.cov_mats[thread_n]

                    subap_ni = self.n_subaps[:wfs_i].sum()
                    subap_nj = self.n_subaps[:wfs_j].sum()

                    # Coordinates of the XX covariance
                    cov_mat_coord_x1 = subap_ni * 2
                    cov_mat_coord_x2 = subap_ni * 2 + self.n_subaps[wfs_i]

                    cov_mat_coord_y1 = subap_nj * 2
                    cov_mat_coord_y2 = subap_nj * 2 + self.n_subaps[wfs_j]

                    # print("covmat coords: ({}: {}, {}: {})".format(cov_mat_coord_x1, cov_mat_coord_x2, cov_mat_coord_y1, cov_mat_coord_y2))
                    r0_scale = ((self.wfs_wavelengths[wfs_i] * self.wfs_wavelengths[wfs_j])
                            / (8 * numpy.pi**2 * self.subap_layer_diameters[layer_n][wfs_i] * self.subap_layer_diameters[layer_n][wfs_j])
                                )

                    self.covariance_matrix[
                            cov_mat_coord_x1: cov_mat_coord_x2, cov_mat_coord_y1: cov_mat_coord_y2
                            ] += cov_xx * r0_scale
                    self.covariance_matrix[
                            cov_mat_coord_x1 + self.n_subaps[wfs_i]: cov_mat_coord_x2 + self.n_subaps[wfs_i],
                            cov_mat_coord_y1: cov_mat_coord_y2] += cov_xy * r0_scale
                    self.covariance_matrix[
                            cov_mat_coord_x1: cov_mat_coord_x2,
                            cov_mat_coord_y1 + self.n_subaps[wfs_j]: cov_mat_coord_y2 + self.n_subaps[wfs_j]
                            ] += numpy.fliplr(numpy.flipud(cov_xy)) * r0_scale
                    self.covariance_matrix[
                            cov_mat_coord_x1 + self.n_subaps[wfs_i]: cov_mat_coord_x2 + self.n_subaps[wfs_i],
                            cov_mat_coord_y1 + self.n_subaps[wfs_j]: cov_mat_coord_y2 + self.n_subaps[wfs_j]
                            ] += cov_yy * r0_scale

                    thread_n += 1


    def make_tomographic_reconstructor(self, svd_conditioning=0):
        """
        Creats a tomohraphic reconstructor from the covariance matrix as in Vidal, 2010.
        See the documentation for the function `create_tomographic_covariance_reconstructor` in this module.
        Assumes the 1st WFS given is the one for which reconstruction is required to.

        Parameters:
            svd_conditioning (float): Conditioning for the SVD used in inversion.

        Returns:
            ndarray: A tomohraphic reconstructor.
        """

        self.tomographic_reconstructor = create_tomographic_covariance_reconstructor(
                self.covariance_matrix, self.n_subaps[0], svd_conditioning)

        return self.tomographic_reconstructor


def wfs_covariance_mpwrap(args):
    return wfs_covariance(*args)


def wfs_covariance(n_subaps1, n_subaps2, wfs1_positions, wfs2_positions, wfs1_diam, wfs2_diam, r0, L0):
    """
    Calculates the covariance between 2 WFSs

    Parameters:
        n_subaps1 (int): number of sub-apertures in WFS 1
        n_subaps2 (int): number of sub-apertures in WFS 1
        wfs1_positions (ndarray): Central position of each sub-apeture from telescope centre for WFS 1
        wfs2_positions (ndarray): Central position of each sub-apeture from telescope centre for WFS 2
        wfs1_diam: Diameter of WFS 1 sub-apertures
        wfs2_diam: Diameter of WFS 2 sub-apertures
        r0: Fried parameter of turbulence
        L0: Outer scale of turbulence

    Returns:
        slope covariance of X with X , slope covariance of Y with Y, slope covariance of X with Y
    """

    xy_seperations = calculate_wfs_seperations(n_subaps1, n_subaps2, wfs1_positions, wfs2_positions)

    # print("Min seperation: {}".format(abs(xy_seperations).min()))
    cov_xx = compute_covariance_xx(xy_seperations, wfs1_diam, wfs2_diam, r0, L0)
    cov_yy = compute_covariance_yy(xy_seperations, wfs1_diam, wfs2_diam, r0, L0)
    cov_xy = compute_covariance_xy(xy_seperations, wfs1_diam, wfs2_diam, r0, L0)


    return cov_xx, cov_yy, cov_xy


def calculate_wfs_seperations(n_subaps1, n_subaps2, wfs1_positions, wfs2_positions):
    """
    Calculates the seperation between all sub-apertures in two WFSs

    Parameters:
        n_subaps1 (int): Number of sub-apertures in WFS 1
        n_subaps2 (int): Number of sub-apertures in WFS 2
        wfs1_positions (ndarray): Array of the X, Y positions of the centre of each sub-aperture with respect to the centre of the telescope pupil
        wfs2_positions (ndarray): Array of the X, Y positions of the centre of each sub-aperture with respect to the centre of the telescope pupil

    Returns:
        ndarray: 2-D Array of sub-aperture seperations
    """


    xy_separations = numpy.zeros((n_subaps1, n_subaps2, 2), dtype='float64')

    for i, (x1, y1) in enumerate(wfs1_positions):
        for j, (x2, y2) in enumerate(wfs2_positions):
            xy_separations[i, j] = (x2-x1), (y2-y1)

    xy_separations += 1e-20

    return xy_separations

    # def get_wfs_wfs_covariance(self, ):


def compute_covariance_xx(seperation, subap1_diam, subap2_diam, r0, L0):

    x1 = seperation[..., 0] + (subap2_diam - subap1_diam) * 0.5
    r1 = numpy.sqrt(x1**2 + seperation[..., 1]**2)

    x2 = seperation[..., 0] - (subap2_diam + subap1_diam) * 0.5
    r2 = numpy.sqrt(x2**2 + seperation[..., 1]**2)

    x3 = seperation[..., 0] + (subap2_diam + subap1_diam) * 0.5
    r3 = numpy.sqrt(x3**2 + seperation[..., 1]**2)

    return (
        -2 * structure_function_vk(r1, r0, L0)
        + structure_function_vk(r2, r0, L0)
        + structure_function_vk(r3, r0, L0)
    )

def compute_covariance_yy(seperation, subap1_diam, subap2_diam, r0, L0):

    y1 = seperation[..., 1] + (subap2_diam - subap1_diam) * 0.5
    r1 = numpy.sqrt(seperation[..., 0]**2 + y1**2)

    y2 = seperation[..., 1] - (subap2_diam + subap1_diam) * 0.5
    r2 = numpy.sqrt(seperation[..., 0]**2 + y2**2)

    y3 = seperation[..., 1] + (subap2_diam + subap1_diam) * 0.5
    r3 = numpy.sqrt(seperation[..., 0]**2 + y3**2)

    return (
        -2 * structure_function_vk(r1, r0, L0)
        + structure_function_vk(r2, r0, L0)
        + structure_function_vk(r3, r0, L0)
    )


def compute_covariance_xy(seperation, subap1_diam, subap2_diam, r0, L0):

    x1 = seperation[..., 0] + subap1_diam * 0.5
    y1 = seperation[..., 1] - subap2_diam * 0.5
    r1 = numpy.sqrt(x1**2 + y1**2)

    x2 = seperation[..., 0] - subap1_diam * 0.5
    y2 = seperation[..., 1] + subap2_diam * 0.5
    r2 = numpy.sqrt(x2**2 + y2**2)

    x3 = seperation[..., 0] + subap1_diam * 0.5
    y3 = seperation[..., 1] + subap2_diam * 0.5
    r3 = numpy.sqrt(x3**2 + y3**2)

    x4 = seperation[..., 0] - subap1_diam * 0.5
    y4 = seperation[..., 1] - subap2_diam * 0.5
    r4 = numpy.sqrt(x4**2 + y4**2)

    return (
        -structure_function_vk(r1, r0, L0)
        - structure_function_vk(r2, r0, L0)
        + structure_function_vk(r3, r0, L0)
        + structure_function_vk(r4, r0, L0)
    )

def structure_function_vk(seperation, r0, L0):
    """
    Computes the Von Karmon structure function of atmospheric turbulence

    Parameters:
        seperation (ndarray, float): float or array of data representing seperations between points
        r0 (float): Fried parameter for atmosphere
        L0 (float): Outer scale of turbulence

    Returns:
        ndarray, float: Structure function for seperation(s)
    """
    return (
        0.17253
        * (L0 / (r0)) ** (5.0 / 3.0)
        * (
            1
            - 2
            * numpy.pi ** (5.0 / 6.0)
            * ((seperation) / L0) ** (5.0 / 6.0)
            / scipy.special.gamma(5.0 / 6.0)
            * scipy.special.kv(5.0 / 6.0, (2 * numpy.pi * seperation) / L0)
        )
    )


def structure_function_kolmogorov(separation, r0):
    '''
        Compute the Kolmogorov phase structure function

        Parameters:
            separation (ndarray, float): float or array of data representing
                separations between points
            r0 (float): Fried parameter for atmosphere

        Returns:
            ndarray, float: Structure function for separation(s)
    '''
    return 6.88 * (separation / r0)**(5. / 3.)


def calculate_structure_function(phase, nbOfPoint=None, step=None):
    '''
        Compute the structure function of an 2D array, along the first dimension.
        SF defined as sf[j]= < (phase - phase_shifted_by_j)^2 >
        Translated from a YAO function.

        Parameters:
            phase (ndarray, 2d): 2d-array
            nbOfPoint (int): final size of the structure function vector. Default is phase.shape[1] / 4
            step (int): step in pixel when computing the sf. (step * sampling_phase) gives the sf sampling in meters. Default is 1

        Returns:
            ndarray, float: values for the structure function of the data.
    '''
    if nbOfPoint is None:
        nbOfPoint = phase.shape[1] / 4
    if step is None:
        step = 1
    step = int(step)
    xm = int(numpy.min([nbOfPoint, phase.shape[1] / step - 1]))
    sf_x = numpy.empty(xm)
    for i in range(step, xm * step, step):
        sf_x[int(i / step)] = numpy.mean((phase[0:-i, :] - phase[i:, :])**2)

    return sf_x


def mirror_covariance_matrix(cov_mat):
    """
    Mirrors a covariance matrix around the axis of the diagonal.

    Parameters:
        cov_mat (ndarray): The covariance matrix to mirror
        n_subaps (ndarray): Number of sub-aperture in each WFS
    """

    return numpy.bitwise_or(cov_mat.view("int32"), cov_mat.T.view("int32")).view("float32")

def create_tomographic_covariance_reconstructor(covariance_matrix, n_onaxis_subaps, svd_conditioning=0):
    """
    Calculates a tomographic reconstructor using the method of Vidal, JOSA A, 2010.

    Uses a slope covariance matrix to generate a reconstruction matrix that will convert a collection of measurements
    from WFSs observing in a variety of directions (the "off-axis" directions) to the measurements that would have been
    observed by a WFS observing in a different direction (the "on-axis" direction.
    The given covariance matrix must include the covariance between all WFS measurements,
    including the "psuedo" WFS pointing in the on-axis direction.
    It is assumed that the covariance of measurements from this on-axis direction are first in the covariance matrix.

    To create the tomographic reconstructor it is neccessary to invert the covariance matrix between off-axis WFS
    measurements. An SVD based psueod inversion is used for this (`numpy.linalg.pinv`). A conditioning to this SVD may
    be required to filter potentially unwanted modes.

    Args:
        covariance_matrix (ndarray): A covariance matrix between WFSs
        n_onaxis_subaps (int): Number of sub-aperture in on-axis WFS
        svd_conditioning (float, optional): Conditioning for SVD inversion (default is 0)

    Returns:

    """

    cov_onoff = covariance_matrix[:2 * n_onaxis_subaps, 2 * n_onaxis_subaps:]
    cov_offoff = covariance_matrix[2 * n_onaxis_subaps:, 2 * n_onaxis_subaps:]


    icov_offoff = numpy.linalg.pinv(cov_offoff, rcond=svd_conditioning)

    return cov_onoff.dot(icov_offoff)
