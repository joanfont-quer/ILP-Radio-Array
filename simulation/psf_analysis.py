import numpy as np
from simulation.psf_simulation import PSFSimulator


class PSFComparator:
    def __init__(self):
        self.simulator = PSFSimulator()

    def compare_subarrays(self, subarrays, weightings=None):
        if weightings is None:
            weightings = ["uniform", "robust 0", "natural"]
        psfs_all = {}
        visibilities = []
        cell_sizes = []

        for i, subarray in enumerate(subarrays):
            print(f"    Simulating subarray {i + 1}/{len(subarrays)}...")
            vis, uvw = self.simulator.simulate_observation(subarray)
            visibilities.append(vis)
            cell_sizes.append(uvw.get_cellsize(over_sample=5))

        global_cell_size = np.min(cell_sizes)
        x_axis = ((np.arange(self.simulator.n_pixels) * global_cell_size) -
                  (global_cell_size * self.simulator.n_pixels / 2))
        x_axis = np.degrees(x_axis) * 3600.0

        for i, vis in enumerate(visibilities):
            psfs = {
                w: self.simulator.generate_psf(global_cell_size, vis=vis, weighting=w)
                for w in weightings
            }
            psfs_all[i] = psfs

        residuals, norms = {}, {}
        for w in weightings:
            bmaj0, bmin0, psf0 = psfs_all[0][w]
            bmaj1, bmin1, psf1 = psfs_all[1][w]

            diff_maj = bmaj0 - bmaj1
            diff_min = bmin0 - bmin1
            diff_psf = psf0["pixels"].data - psf1["pixels"].data

            residuals[w] = (diff_maj, diff_min)
            l1 = np.mean(np.abs(diff_psf))
            l2 = np.sqrt(np.mean(diff_psf ** 2))
            lobes = diff_psf[diff_psf < np.max(psf0["pixels"].data) / 2]
            lobe_rms = np.sqrt(np.mean(lobes ** 2))
            norms[w] = (l1, l2, lobe_rms)

        return residuals, norms, psfs_all, x_axis
