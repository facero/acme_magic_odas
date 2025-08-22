# utility functions to plot the ON, OFF, and exclusion regions on top of a skymap
import logging
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from regions import PointSkyRegion, CircleSkyRegion
from gammapy.maps import Map
from gammapy.makers import WobbleRegionsFinder
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# let us use proper logging
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def create_fake_legend(ax, source_name, excluded_source_name):
    """Make a fake legend representing the ON and OFF regions"""
    markersize = 14
    on = Line2D(
        [0],
        [0],
        markeredgecolor="crimson",
        markerfacecolor="none",
        markeredgewidth=1.5,
        marker="o",
        markersize=markersize,
        ls="",
        label="on region",
    )
    off = Line2D(
        [0],
        [0],
        markeredgecolor="dodgerblue",
        markerfacecolor="none",
        markeredgewidth=1.5,
        marker="o",
        markersize=markersize,
        ls="",
        label="off regions",
    )
    pointing = Line2D(
        [0],
        [0],
        markeredgecolor="goldenrod",
        markerfacecolor="none",
        markeredgewidth=1.5,
        marker="+",
        markersize=(markersize - 2),
        ls="",
        label="pointing",
    )
    source = Line2D(
        [0],
        [0],
        markeredgecolor="crimson",
        markerfacecolor="none",
        markeredgewidth=1.5,
        marker="*",
        markersize=(markersize - 2),
        ls="",
        label=source_name,
    )
    excluded_source = Line2D(
        [0],
        [0],
        markeredgecolor="k",
        markerfacecolor="none",
        markeredgewidth=1.5,
        marker="*",
        markersize=(markersize - 2),
        ls="",
        label=excluded_source_name,
    )
    excluded_region = Line2D(
        [0],
        [0],
        markeredgecolor="k",
        markerfacecolor="none",
        markeredgewidth=1.5,
        marker="o",
        markersize=markersize,
        ls="",
        label="excluded region",
    )
    ax.legend(
        handles=[on, off, pointing, source, excluded_source, excluded_region],
        loc="best",
    )


def plot_on_off_regions_skymap(
    observation,
    skymap_geom,
    on_region,
    exclusion_region,
    n_off_regions,
    source_name,
    excluded_source_name,
    show=False,
):
    """Given an observation, make a skymap overlaying the ON and OFF regions."""

    log.info(f"making ON and OFF regions skymap plot for run {observation.obs_id}")

    # get the pointing coordinates
    pointing_coords = observation.pointing.get_icrs()

    # plotting starts here
    fig = plt.figure()

    # fill the counts map
    counts = Map.from_geom(skymap_geom)
    counts.fill_events(observation.events)
    # define wcs geom from the counts map
    wcs = counts.geom.wcs
    # plot the countmap
    ax = counts.plot(cmap="viridis", add_cbar=True)
    # add the counts label
    ax.images[-1].colorbar.set_label("counts", rotation=270, labelpad=12, fontsize=10)

    # add the pointing and the wobbling
    # wobbling distance
    wobble_radius = observation.pointing.get_icrs().separation(on_region.center).to_value("deg")
    pointing_region = PointSkyRegion(pointing_coords)
    pointing_region.to_pixel(wcs).plot(ax=ax, color="goldenrod", marker="+", markersize=12)
    wobble_circle = CircleSkyRegion(
        center=pointing_coords, radius=Angle(f"{wobble_radius} deg")
    )
    wobble_circle.to_pixel(wcs).plot(ax=ax, edgecolor="goldenrod", ls="--", linewidth=2)

    # plot the ON region
    on_region.to_pixel(wcs).plot(ax=ax, edgecolor="crimson", linewidth=2)

    # plot the source positions and the exclusion mask
    PointSkyRegion(on_region.center).to_pixel(wcs).plot(
        ax=ax, color="crimson", marker="*", markersize=12
    )
    PointSkyRegion(exclusion_region.center).to_pixel(wcs).plot(
        ax=ax, color="k", marker="*", markersize=12
    )
    exclusion_region.to_pixel(wcs).plot(ax=ax, edgecolor="k", ls="-", linewidth=2)

    # create the OFF regions, this is the same process performed by the BackgroundMaker in the dataset
    region_finder = WobbleRegionsFinder(n_off_regions=n_off_regions)
    # find the OFF regions centers
    off_regions, wcs = region_finder.run(
        region=PointSkyRegion(on_region.center), center=pointing_coords
    )

    # plot the OFF regions
    for off_region in off_regions:
        off_region_circle = CircleSkyRegion(
            center=off_region.center, radius=Angle(f"{on_region.radius.value} deg")
        )
        off_region_circle.to_pixel(ax.wcs).plot(
            ax=ax, edgecolor="dodgerblue", linewidth=2
        )

    create_fake_legend(
        ax=ax, source_name=source_name, excluded_source_name=excluded_source_name
    )
    ax.set_title(f"run {observation.obs_id}, {n_off_regions} off regions")

    if show:
        plt.show()

#    fig.savefig(
#        f"run_{observation.obs_id}_theta_max_{on_region.radius.value}_n_off_regions_{n_off_regions}.png"
#    )


def plot_gammapy_lc_points(ax, lc, color, label, alpha=1.0):
    """Get the flux points from a Gammapy LC.
    Correct for the shift in time."""

    mjd = lc.geom.axes["time"].time_mid.value
    mjd_err = 0.5 * (
        lc.geom.axes["time"].time_max.value - lc.geom.axes["time"].time_min.value
    )
    flux = lc.flux.to_unit("cm-2 s-1").data.flatten()
    flux_err = lc.flux_err.to_unit("cm-2 s-1").data.flatten()
    uls_mask = lc.is_ul.data.flatten()
    flux_uls = lc.flux_ul.to_unit("cm-2 s-1").data.flatten()

    # flux points
    ax.errorbar(
        mjd[~uls_mask],
        flux[~uls_mask],
        xerr=mjd_err[~uls_mask],
        yerr=flux_err[~uls_mask],
        ls="",
        marker=".",
        color=color,
        alpha=alpha,
        label=label,
    )

    # upper limits
    ax.errorbar(
        mjd[uls_mask],
        flux_uls[uls_mask],
        yerr=0.25 * np.nanmax(flux_err),
        uplims=True,
        ls="",
        color=color,
        alpha=alpha,
    )


def plot_gammapy_sed(ax, spectral_model, flux_points, color, label):
    """Make a plot of the broadband spectrum and of the flux points.
    As for the convention used in MAGIC, we will plot the broadband model
    from the first to the last energy bin center."""


    e_min = flux_points.energy_ref[0]
    e_max = flux_points.energy_ref[-1]

    plot_kwargs = {
        "energy_bounds": (e_min, e_max),
        "sed_type": "e2dnde",
        "yunits": u.Unit("TeV cm-2 s-1"),
        "xunits": u.GeV,
    }

    # plot spectral model
    spectral_model.plot(
        ax=ax,
        ls="--",
        lw=2,
        color=color,
        **plot_kwargs,
    )
    spectral_model.plot_error(ax=ax, facecolor=color, alpha=0.4, **plot_kwargs)

    flux_points.plot(
        ax=ax,
        ls="",
        markeredgewidth=0,
        color=color,
        label=label,
        sed_type=plot_kwargs["sed_type"],
    )

    ax.set_xlabel(r"$E\,/\,{\rm GeV}$")
    ax.set_ylabel(
        r"$E^2 {\rm d}\phi/{\rm d}E\,/\,({\rm TeV}\,{\rm cm}^{-2}\,{\rm s}^{-1})$"
    )
