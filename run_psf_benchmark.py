import argparse
from utils import load_config
from simulation.psf_analysis import PSFComparator
import matplotlib.pyplot as plt
import time


def run_comparison(config):
    print(f"Running PSF comparison: {config.NAME}")
    comp = PSFComparator()

    subarrays= config.get_subarrays()

    print(f"    {config.NAME} subarrays: {len(subarrays)}")

    residuals, norms, psfs_all, x_axis = comp.compare_subarrays([subarrays[0], subarrays[1]])

    for w, (l1, l2, lobe_rms) in norms.items():
        print(f"{w}  L1={l1:.4e}, L2={l2:.4e}, Lobe RMS={lobe_rms:.4e}")

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
    plt.savefig(f"PSF_Residuals_{config.NAME}.png")


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description="Compare PSFs between two configurations")
    parser.add_argument("--config", required=True, help="Path to configuration .py file")

    args = parser.parse_args()

    config1 = load_config(args.config)
    run_comparison(config1)
    end = time.time()
    print(f"Runtime: {end - start}")

if __name__ == "__main__":
    main()