from ska_ost_array_config.simulation_utils import generate_mfs_psf, get_PSF_shape_and_profile, simulate_observation
from ska_ost_array_config.UVW import UVW
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units


class PSFSimulator:
    def __init__(self, ref_freq=1.4e9, chan_width=1e6, n_chan=4, integ_time=10, n_pixels=400):
        self.ref_freq = ref_freq
        self.chan_width = chan_width
        self.n_chan = n_chan
        self.integ_time = integ_time
        self.n_pixels = n_pixels
        self.vis = None
        self.uvw = None

    def simulate_observation(self, subarray, duration=4*60):
        ref_time = Time.now()
        zenith = SkyCoord(
            alt=90 * units.deg,
            az=0 * units.deg,
            frame="altaz",
            obstime=ref_time,
            location=subarray.array_config.location,
        ).icrs

        vis = simulate_observation(
            array_config=subarray.array_config,
            phase_centre=zenith,
            start_time=ref_time,
            ref_freq=self.ref_freq,
            chan_width=self.chan_width,
            n_chan=self.n_chan,
            integration_time=self.integ_time,
            duration=duration
        )

        uvw = UVW(vis, ignore_autocorr=True)
        self.vis = vis
        self.uvw = uvw
        return vis, uvw

    def generate_psf(self, cell_size, vis=None, weighting="uniform"):
        if vis is None:
            vis = self.vis
        psf = generate_mfs_psf(
            vis,
            cellsize=cell_size,
            npixel=self.n_pixels,
            weighting=weighting.replace(" 0", ""),
            r_value=0.0,
            return_sidelobe_noise=False
        )

        _, (bmaj, bmin) = get_PSF_shape_and_profile(psf, return_profile=True)
        return bmaj, bmin, psf
