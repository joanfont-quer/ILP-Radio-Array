from collections import defaultdict
import matplotlib.pyplot as plt
import numpy
import ndtest
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.time import Time
from matplotlib.ticker import MaxNLocator
from ska_ost_array_config.array_config import LowSubArray, MidSubArray
from ska_ost_array_config.simulation_utils import (
    generate_mfs_psf,
    get_PSF_shape_and_profile,
    simulate_observation,
)
from ska_ost_array_config.UVW import UVW, plot_baseline_distribution, plot_uv_coverage

def generate_similar_subarrays_uvselection_ks_method(
    full_array, n_sub_arrays, start_at_max=True
):
    """
    Split full_array into similar number of n_sub_arrays.

    Parameters
    ----------
    full_array:
        Subarray object of type ska_ost_array_config.array_config.LowSubArray
        or ska_ost_array_config.array_config.MidSubArray
    n_sub_arrays: int
        Number of smaller subarrays to generate
    start_at_max: bool
        FIXME

    Returns
    -------
    Two dimensional list of subarrays with shape
    (n_sub_arrays x int(floor(len(station_list)/n_sub_arrays)))
    """
    # First, get the list of all stations/dishes in the full array
    station_list = get_antenna_list(full_array).split(",")

    # Get the uv distribution for every station in the array
    # Need to do this here to use the simple fast cost calculation
    # - this is done only once
    n_stations_full = len(station_list)
    ref_time = Time.now()
    zenith = SkyCoord(
        alt=90 * units.deg,
        az=0 * units.deg,
        frame="altaz",
        obstime=ref_time,
        location=full_array.array_config.location,
    ).icrs
    if isinstance(full_array, LowSubArray):
        ref_freq = 200e6
        chan_width = 5.4e3
    elif isinstance(full_array, MidSubArray):
        ref_freq = 1.4e9
        chan_width = 13.4e3
    else:
        err_msg = "Invalid full array object specified"
        raise IOError(err_msg)
    n_chan = 1
    integ_time = 1
    vis_full = simulate_observation(
        array_config=full_array.array_config,
        phase_centre=zenith,
        start_time=ref_time,
        ref_freq=ref_freq,
        chan_width=chan_width,
        n_chan=n_chan,
        integration_time=integ_time,
        duration=integ_time,
    )
    uvw_full = UVW(vis_full, ignore_autocorr=True)

    autocorr = vis_full.uvw.antenna2.data == vis_full.uvw.antenna1.data
    full_array_antenna1 = vis_full.uvw.antenna1.data[~autocorr]
    full_array_antenna2 = vis_full.uvw.antenna2.data[~autocorr]
    full_array_u_m = uvw_full.u_m / 1000.0
    full_array_v_m = uvw_full.v_m / 1000.0

    u_stations = numpy.zeros((n_stations_full, n_stations_full - 1))
    v_stations = numpy.zeros((n_stations_full, n_stations_full - 1))
    for si, s in enumerate(station_list):
        # get the index of this station in the simulated observation uvw list
        array_index = list(vis_full.configuration["names"].data).index(s)

        array_bl_mask = numpy.logical_or(
            full_array_antenna1 == array_index, full_array_antenna2 == array_index
        )

        # store the 2d uv-distribution for this station in the array
        u_stations[si, :] = full_array_u_m[array_bl_mask]
        v_stations[si, :] = full_array_v_m[array_bl_mask]

    # sort the station list by distance so we can select the most distant station
    # first (or least if start_at_max is not set)
    distance = numpy.sqrt(
        vis_full.configuration.xyz.data[:, 0] ** 2
        + vis_full.configuration.xyz.data[:, 1] ** 2
    ).tolist()
    sorted_name_list = [
        name for _, name in sorted(zip(distance, station_list), reverse=start_at_max)
    ]

    # now start picking stations
    sub_arrays = [[] for _ in range(n_sub_arrays)]
    station_list_remaining = station_list.copy()
    stations_used = []

    Nremaining = len(station_list_remaining)
    # pick most distant antenna
    station_i = sorted_name_list[0]

    station_list_remaining.pop(station_list_remaining.index(station_i))
    sub_arrays[0].append(station_i)

    i = 0
    while len(station_list_remaining) > (n_stations_full % n_sub_arrays):
        i += 1
        si = i % n_sub_arrays  # sub_array to assign to

        Nremaining = len(station_list_remaining)
        remaining_station_indices = list(range(Nremaining))

        station_index = station_list.index(station_i)
        # pick the best antenna

        # get uv dist of lst station picked and choose the one out of the remaining
        # stations with the uv distribution (for it to all other stations in the full
        # array - simplification for now) most like its own
        cost_per_station = numpy.zeros(len(station_list_remaining))
        for j, station_j in enumerate(station_list_remaining):
            station_index_j = station_list.index(station_j)

            # calculate the KS statistic for every other station compared to the last
            # one selected
            if station_j == station_i:
                cost_per_station[j] = 1e10  # can't add itself
            else:
                cost_per_station[j] = ndtest.avgmaxdist(
                    u_stations[station_index_j, :],
                    v_stations[station_index_j, :],
                    u_stations[station_index, :],
                    v_stations[station_index, :],
                )

        new_station_index = numpy.argmin(cost_per_station)
        station_i = station_list_remaining[new_station_index]
        station_list_remaining.pop(station_list_remaining.index(station_i))
        sub_arrays[si].append(station_i)

    return sub_arrays


def get_antenna_list(sub_array):
    """Print the names of all the stations/dishes in this subarray"""
    return ",".join(sorted(sub_array.array_config.names.data.tolist()))


def describe_subarray(
    sub_array,
    ref_freq,
    chan_width,
    n_chan,
    integ_time,
    output_file="plot.png",
    title="title",
    n_bins=50,
):
    """
    For the specified subarray, plot
    * array layout,
    * uv coverage, and
    * 1D profiles of the PSF

    Parameters
    -----------
    sub_array: ska_ost_array_config.array_config.SubArray

    ref_freq: float
        Frequency of the first channel in Hz

    chan_width: float
        Channel width in Hz

    n_chan: int
        Number of channels to include in the simulation

    integ_time: float
        Time resolution of the simulated observation

    output_file: str
        File name of the plot. Default: plot.png

    title: str
        Text to display as title on the plot.
        Default: title

    n_bins: int
        Number of bins to use in the baseline distribution plot
        Default: 50
    """
    fig, axes = plt.subplots(2, 3, layout="constrained", figsize=(15, 10))

    # Plot the array layout
    sub_array.plot_array_layout(axes[0][0], scale="kilo", s=5, c="C0")
    axis_limit = numpy.max(
        [
            numpy.max(numpy.absolute(axes[0][0].get_xlim())),
            numpy.max(numpy.absolute(axes[0][0].get_ylim())),
        ]
    )
    axis_limit += 0.1 * axis_limit
    axes[0][0].set_xlim(-axis_limit, axis_limit)
    axes[0][0].set_ylim(-axis_limit, axis_limit)
    axes[0][0].set_aspect("equal")
    axes[0][0].set_box_aspect(1)
    axes[0][0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0][0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0][0].set_title("Array layout")

    # Simulate snapshot and long-track observations
    ref_time = Time.now()
    zenith = SkyCoord(
        alt=90 * units.deg,
        az=0 * units.deg,
        frame="altaz",
        obstime=ref_time,
        location=sub_array.array_config.location,
    ).icrs
    vis_4min = simulate_observation(
        array_config=sub_array.array_config,
        phase_centre=zenith,
        start_time=ref_time,
        ref_freq=ref_freq,
        chan_width=chan_width,
        n_chan=n_chan,
        integration_time=integ_time,
        duration=4 * 60,
    )
    uvw_4min = UVW(vis_4min, ignore_autocorr=True)
    vis_4h = simulate_observation(
        array_config=sub_array.array_config,
        phase_centre=zenith,
        start_time=ref_time,
        ref_freq=ref_freq,
        chan_width=chan_width,
        n_chan=n_chan,
        integration_time=integ_time,
        duration=4 * 3600,
    )
    uvw_4h = UVW(vis_4h, ignore_autocorr=True)

    # Plot the uv coverages (including the complex conjugates) in the top row
    plot_uv_coverage(
        uvw_4min,
        axes=axes[0][1],
        method="metre",
        plot_conj=True,
        scale="kilo",
        c="C0",
        c_conj="C0",
    )
    axes[0][1].set_title("uv coverage (snapshot)")
    axes[0][1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0][1].yaxis.set_major_locator(MaxNLocator(nbins=5))

    plot_uv_coverage(
        uvw_4h,
        axes=axes[0][2],
        method="metre",
        plot_conj=True,
        scale="kilo",
        c="C0",
        c_conj="C0",
    )
    axes[0][2].set_title("uv coverage (4h)")
    axes[0][2].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0][2].yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Plot the PSF sizes in the first two columns of the bottom row
    n_pixels = 1000
    cell_size = uvw_4min.get_cellsize(over_sample=5)
    x_axis = (numpy.arange(n_pixels) * cell_size) - (cell_size * n_pixels / 2)
    x_axis = numpy.degrees(x_axis) * 3600.0
    for weighting in ["uniform", "robust 0", "natural"]:
        psf = generate_mfs_psf(
            vis_4min,
            cellsize=cell_size,
            npixel=n_pixels,
            weighting=weighting.replace(" 0", ""),
            r_value=0.0,
            return_sidelobe_noise=False,
        )
        synth_beam, (bmaj, bmin) = get_PSF_shape_and_profile(psf, return_profile=True)
        axes[1][0].plot(x_axis, bmaj, label=weighting)
        axes[1][1].plot(x_axis, bmin, label=weighting)
    x_limit = 10 * synth_beam["bmaj"] * 3600.0  # [arcsec]
    axes[1][0].legend()
    axes[1][0].set_xlim([-x_limit, x_limit])
    axes[1][0].set_xlabel("[arcsec]")
    axes[1][0].set_ylabel("|PSF|")
    axes[1][0].set_title("PSF major axis")

    axes[1][1].legend()
    axes[1][1].set_xlim([-x_limit, x_limit])
    axes[1][1].set_xlabel("[arcsec]")
    axes[1][1].set_ylabel("|PSF|")
    axes[1][1].set_title("PSF minor axis")

    # Plot a histogram of the baseline distribution in the bottom-right plot
    plot_baseline_distribution(
        uvw_4min,
        axes=axes[1][2],
        method="metre",
        scale="kilo",
        bins=n_bins,
        density=True,
        cumulative=True,
    )
    axes[1][2].set_title("Cumulative Baseline distribution")
    axes[1][2].set_ylabel("Fraction of baselines")

    fig.suptitle(title)

    # plt.tight_layout()
    plt.savefig(output_file, dpi=150)


def plot_similar_array_layout(
    sub_arrays_list, telescope, title="title", output_file="plot.png"
):
    """
    Plot the array layouts for similar subarrays

    Parameters
    -----------
    sub_arrays_list: list of list containing similar subarrays

    telescope: string
        Telescope type (valid choices are "LOW" or "MID")

    output_file: str
        File name of the plot. Default: plot.png

    title: str
        Text to display as title on the plot.
        Default: title
    """
    fig, axes = plt.subplots(figsize=(5, 5))

    assert telescope in [
        "LOW",
        "MID",
    ], "Invalid telescope specified. Must be LOW or MID"

    for idx, station_list in enumerate(sub_arrays_list):
        if telescope == "LOW":
            sub_array = LowSubArray(
                subarray_type="custom", custom_stations=",".join(station_list)
            )
        else:
            sub_array = MidSubArray(
                subarray_type="custom", custom_stations=",".join(station_list)
            )

        # Plot the array layout of all subarrays in different colour
        sub_array.plot_array_layout(axes, scale="kilo", s=5, c=f"C{idx}")

    axes.set_title("Array layout")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)


def describe_similar_subarray(
    sub_arrays_list,
    telescope,
    ref_freq,
    chan_width,
    n_chan,
    integ_time,
    output_file="plot.png",
    title="title",
    n_bins=50,
):
    """
    For the specified similar subarray, plot
    * array layout,
    * uv coverage, and
    * 1D profiles of the PSF

    Parameters
    -----------
    sub_arrays_list:

    telescope: string
        Telescope type (valid choices are "LOW" or "MID")

    ref_freq: float
        Frequency of the first channel in Hz

    chan_width: float
        Channel width in Hz

    n_chan: int
        Number of channels to include in the simulation

    integ_time: float
        Time resolution of the simulated observation

    output_file: str
        File name of the plot. Default: plot.png

    title: str
        Text to display as title on the plot.
        Default: title

    n_bins: int
        Number of bins to use in the baseline distribution plot
        Default: 50
    """
    fig, axes = plt.subplots(2, 3, layout="constrained", figsize=(15, 10))

    assert telescope in [
        "LOW",
        "MID",
    ], "Invalid telescope specified. Must be LOW or MID"

    for idx, sub_array in enumerate(sub_arrays_list):

        # Plot the array layout of all subarrays in different colour
        sub_array.plot_array_layout(axes[0][0], scale="kilo", s=5, c=f"C{idx}")

        # Simulate snapshot and long-track observations
        ref_time = Time.now()
        zenith = SkyCoord(
            alt=90 * units.deg,
            az=0 * units.deg,
            frame="altaz",
            obstime=ref_time,
            location=sub_array.array_config.location,
        ).icrs
        vis_4min = simulate_observation(
            array_config=sub_array.array_config,
            phase_centre=zenith,
            start_time=ref_time,
            ref_freq=ref_freq,
            chan_width=chan_width,
            n_chan=n_chan,
            integration_time=integ_time,
            duration=4 * 60,
        )
        uvw_4min = UVW(vis_4min, ignore_autocorr=True)
        vis_4h = simulate_observation(
            array_config=sub_array.array_config,
            phase_centre=zenith,
            start_time=ref_time,
            ref_freq=ref_freq,
            chan_width=chan_width,
            n_chan=n_chan,
            integration_time=integ_time,
            duration=4 * 3600,
        )
        uvw_4h = UVW(vis_4h, ignore_autocorr=True)

        if idx == 0:
            # Plot the uv coverages (including the complex conjugates) in the top row
            plot_uv_coverage(
                uvw_4min,
                axes=axes[0][1],
                method="metre",
                plot_conj=True,
                scale="kilo",
                c=f"C{idx}",
                c_conj=f"C{idx}",
            )
            axes[0][1].set_title("uv coverage (snapshot)")
            axes[0][1].xaxis.set_major_locator(MaxNLocator(nbins=5))
            axes[0][1].yaxis.set_major_locator(MaxNLocator(nbins=5))
            plot_uv_coverage(
                uvw_4h,
                axes=axes[0][2],
                method="metre",
                plot_conj=True,
                scale="kilo",
                c=f"C{idx}",
                c_conj=f"C{idx}",
            )
            axes[0][2].xaxis.set_major_locator(MaxNLocator(nbins=5))
            axes[0][2].yaxis.set_major_locator(MaxNLocator(nbins=5))

        # Plot the PSF sizes in the first two columns of the bottom row
        # This is done only for the first subarray
        if idx == 1:
            n_pixels = 200
            cell_size = uvw_4min.get_cellsize(over_sample=5)
            x_axis = (numpy.arange(n_pixels) * cell_size) - (cell_size * n_pixels / 2)
            x_axis = numpy.degrees(x_axis) * 3600.0
            for weighting in ["uniform", "robust 0", "natural"]:
                psf = generate_mfs_psf(
                    vis_4min,
                    cellsize=cell_size,
                    npixel=n_pixels,
                    weighting=weighting.replace(" 0", ""),
                    r_value=0.0,
                    return_sidelobe_noise=False,
                )
                _, (bmaj, bmin) = get_PSF_shape_and_profile(psf, return_profile=True)
                axes[1][0].plot(x_axis, bmaj, label=weighting)
                axes[1][1].plot(x_axis, bmin, label=weighting)

        # Plot a histogram of the baseline distribution in the bottom-right plot
        plot_baseline_distribution(
            uvw_4min,
            axes=axes[1][2],
            method="metre",
            scale="kilo",
            bins=n_bins,
            density=True,
            cumulative=True,
        )

    axis_limit = numpy.max(
        [
            numpy.max(numpy.absolute(axes[0][0].get_xlim())),
            numpy.max(numpy.absolute(axes[0][0].get_ylim())),
        ]
    )
    axis_limit += 0.1 * axis_limit
    axes[0][0].set_xlim(-axis_limit, axis_limit)
    axes[0][0].set_ylim(-axis_limit, axis_limit)
    axes[0][0].set_aspect("equal")
    axes[0][0].set_box_aspect(1)
    axes[0][0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0][0].yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Set all the axes labels
    axes[0][0].set_title("Array layout")
    axes[0][2].set_title("uv coverage (4h)")

    axes[1][0].legend()
    axes[1][0].set_xlabel("[arcsec]")
    axes[1][0].set_ylabel("|PSF|")
    axes[1][0].set_title("PSF major axis")

    axes[1][1].legend()
    axes[1][1].set_xlabel("[arcsec]")
    axes[1][1].set_ylabel("|PSF|")
    axes[1][1].set_title("PSF minor axis")

    axes[1][2].set_title("Cumulative Baseline distribution")
    axes[1][2].set_ylabel("Fraction of baselines")

    fig.suptitle(title)

    plt.savefig(output_file, dpi=150)


def assignments_to_string(assignments):
    grouped = defaultdict(list)
    for ant, idx in assignments.items():
        grouped[idx].append(ant)

    return [",".join(sorted(grouped[i])) for i in sorted(grouped)]


def build_subarrays_from_assignments(solution):
    assignment_strings = assignments_to_string(solution)

    subarrays = []
    for assignment in assignment_strings:
        subarray = MidSubArray(subarray_type="custom", custom_stations=assignment)
        subarrays.append(subarray)
    return subarrays
