from ska_ost_array_config.simulation_utils import (
    generate_mfs_psf,
    get_PSF_shape_and_profile,
    simulate_observation
)
from ska_ost_array_config.UVW import UVW
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units
import numpy as np
import matplotlib.pyplot as plt
from utils import build_subarrays_from_assignments


ref_freq = 1.4e9
chan_width = 1e6
n_chan = 4
integ_time = 10
n_pixels = 400


def observation(sub_array):
    ref_time = Time.now()
    zenith = SkyCoord(
        alt=90 * units.deg,
        az=0 * units.deg,
        frame="altaz",
        obstime=ref_time,
        location=sub_array.array_config.location,
        ).icrs

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
    return uvw_4h, vis_4h


def get_psf(vis_4h, cell_size):

    psfs = {}
    for weighting in ["uniform", "robust 0", "natural"]:
        psf = generate_mfs_psf(
            vis_4h,
            cellsize=cell_size,
            npixel=n_pixels,
            weighting=weighting.replace(" 0", ""),
            r_value=0.0,
            return_sidelobe_noise=False,
        )
        _, (bmaj, bmin) = get_PSF_shape_and_profile(psf, return_profile=True)
        psfs[weighting] = (bmaj, bmin, psf)
    return psfs


def compare_all_psf(subarrays):
    psfs_all = {}
    vis_4hs = []
    cell_sizes = []
    for i, subarray in enumerate(subarrays):
        uvw_4h, vis_4h = observation(subarray)
        vis_4hs.append(vis_4h)

        cell_size = uvw_4h.get_cellsize(over_sample=5)
        cell_sizes.append(cell_size)
    global_cell_size = np.min(cell_sizes)
    x_axis = (np.arange(n_pixels) * global_cell_size) - (global_cell_size * n_pixels / 2)
    x_axis = np.degrees(x_axis) * 3600.0
    
    

    for i, subarray in enumerate(subarrays):
        psfs = get_psf(vis_4hs[i], global_cell_size)
        psfs_all[i] = psfs
    
    residuals = {}
    norms = {}

    for weighting in ["uniform", "robust 0", "natural"]:
        residuals[weighting] = []
        norms[weighting] = []

        bmaj0, bmin0, psf0 = psfs_all[0][weighting]
        bmaj1, bmin1, psf1 = psfs_all[1][weighting]

        diff_maj = bmaj0 - bmaj1
        diff_min = bmin0 - bmin1
        diff_psf = psf0["pixels"].data - psf1["pixels"].data
        residuals[weighting] = (diff_maj, diff_min)

        l1 = np.mean(np.abs(diff_psf))
        l2 = np.sqrt(np.mean(diff_psf ** 2))

        lobes = diff_psf[diff_psf < np.max(psf0["pixels"].data) / 2]
        lobe_rms = np.sqrt(np.mean(lobes ** 2))

        norms[weighting] = (l1, l2, lobe_rms)
    
    return residuals, norms, psfs_all, x_axis


def plot_res(residuals, psfs_all, x_axis, key):
    fig, ax = plt.subplots(4, 3, figsize=(15, 15))

    for i, weighting in enumerate(["uniform", "robust 0", "natural"]):
        bmaj0, bmin0, _ = psfs_all[0][weighting]
        bmaj1, bmin1, _ = psfs_all[1][weighting]

        ax[0, i].plot(x_axis, residuals[weighting][0])
        ax[1, i].plot(x_axis, residuals[weighting][1])

        ax[2, i].plot(x_axis, bmaj0)
        ax[2, i].plot(x_axis, bmaj1)

        ax[3, i].plot(x_axis, bmin0)
        ax[3, i].plot(x_axis, bmin1)


        ax[0, i].set_title(f"{weighting} - Bmaj residual")
        ax[1, i].set_title(f"{weighting} - Bmin residual")
        ax[2, i].set_title(f"{weighting} - PSF Major axis")
        ax[3, i].set_title(f"{weighting} - PSF Minor axis")

        ax[0, i].set_ylabel("Residual")
        ax[1, i].set_ylabel("Residual")

    plt.tight_layout()
    plt.savefig(f"PSF_Residuals_{key}.png")


def main():
    solutions = dict(np.load("/share/nas2_3/jfont/ILP-Radio-Array/solutions_heuristic_ska_full.npz", allow_pickle=True))

    for key, solution in solutions.items():
        print(f"Solution key: {key}")

        solution = solution.item() if isinstance(solution.item(), dict) else solution

        subarrays = build_subarrays_from_assignments(solution)
        residuals, norms, psfs_all, x_axis = compare_all_psf(subarrays)

        for weighting in ["uniform", "robust 0", "natural"]:
            l1, l2, lobe_rms = norms[weighting]
            print(f"Weighting: {weighting}")
            print(f"  L1: {l1:.4e}")
            print(f"  L2: {l2:.4e}")
            print(f"  Lobe RMS: {lobe_rms:.4e}")
        print()

        plot_res(residuals, psfs_all, x_axis, key)


if __name__ == '__main__':
    main()
