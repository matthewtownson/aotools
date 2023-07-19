from aotools import turbulence
import numpy


def test_calc_slope_temporalps():
    slopes = numpy.random.random((1000, 104))
    mean_spectra, error_spectra = turbulence.calc_slope_temporalps(slopes)
    assert len(mean_spectra) == 500
    assert len(error_spectra) == 500


def test_get_tps_time_axis():
    frame_rate = 100
    n_frames = 1000
    t_values = turbulence.get_tps_time_axis(frame_rate, n_frames)
    assert len(t_values) == 500


# def test_fit_tps():
#     frames = 100
#     frame_rate = 10
#     sub_aperture_diameter = 0.5
#     slopes = numpy.random.random((frames, 104))
#     temporal_power_spectra, error = turbulence.calc_slope_temporalps(slopes)
#     t_axis = turbulence.get_tps_time_axis(frame_rate, frames)
#     turbulence.fit_tps(temporal_power_spectra, t_axis, sub_aperture_diameter)


def test_tps_fit():
    """
    Tests the validaty of a fit to the temporal power spectrum.

    Uses the temporal power spectrum and time-axis data to test the validity of a coherence time. A frequency above which fitting is not performaed should also be given, as noise will be the dominant contributor above this.

    Parameters:
        tps (ndarray): Temporal power spectrum to fit
        t_axis_data (ndarray): Time axis data
        D (float): (sub-) Aperture diameter
        V (float): Integrated wind speed
        f_noise (float): Frequency above which noise dominates.
        A (float): Initial Guess of
    """
    # Parameters
    frame_rate = 100
    n_frames = 1000
    slopes = numpy.random.random((1000, 104))
    tps, error_spectra = turbulence.calc_slope_temporalps(slopes)
    t_axis_data = turbulence.get_tps_time_axis(frame_rate, n_frames)
    D = 0.5
    V = 20
    f_noise = t_axis_data[-2]
    A = 1
    tps_err = None
    plot = False

    t_values = turbulence.get_tps_time_axis(frame_rate, n_frames)

    # Start testing function
    f0 = 0.3 * V/D

    if f0<t_axis_data[0] or f0>f_noise or f_noise>t_axis_data.max():
        return 10**99

    tps_tt_indices = numpy.where((t_axis_data<f0) & (t_axis_data>0))[0]
    tt_t_data = t_axis_data[tps_tt_indices]
    tt_fit = 10**A * tt_t_data**(-2./3)

    # get scaling for next part of distribution so it matches up at cutof freq.
    tps_ho_indices = numpy.where((t_axis_data>f0) & (t_axis_data<f_noise))
    ho_t_data = t_axis_data[tps_ho_indices]
    B = tt_fit[-1]/(ho_t_data[0] ** (-11./3))
    ho_fit = B * ho_t_data ** (-11./3)

    ps_fit = numpy.append(tt_fit, ho_fit)
    fit_t_data = numpy.append(tt_t_data, ho_t_data)
    fit_t_coords = numpy.append(tps_tt_indices, tps_ho_indices)


    if tps_err is None:
        err = numpy.mean((numpy.sqrt(numpy.square(ps_fit - tps[fit_t_coords]))))
    else:
        chi2 = numpy.square((ps_fit - tps[fit_t_coords])/tps_err[fit_t_coords]).sum()
        err = chi2/ps_fit.shape[0]



    if plot:
        print(f"V: {V}, f_noise: {f_noise}, A: {A}")
        pyplot.cla()
        ax = pyplot.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        if tps_err is None:
            ax.loglog(t_axis_data, tps)
        else:
            ax.errorbar(t_axis_data, tps, tps_err)
        ax.plot(fit_t_data, ps_fit, linewidth=2)

        ax.plot([f0]*2, [0.1, tps.max()], color='k')
        pyplot.draw()
        pyplot.pause(0.01)
        print(f"Error Func: {err}")

    return err


def test_tps_fit_minimize_func():
    # Parameters
    frame_rate = 100
    n_frames = 1000
    slopes = numpy.random.random((1000, 104))
    tps, tps_err = turbulence.calc_slope_temporalps(slopes)
    args = [20., 20., 1.]
    t_axis = turbulence.get_tps_time_axis(frame_rate, n_frames)
    D = 0.5
    plot = False

    # Test code
    V, f_noise, A = args

    return test_tps_fit()
