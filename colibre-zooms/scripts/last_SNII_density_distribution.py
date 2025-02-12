"""
Plots the SNII density distribution (at last SNII thermal injections).
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt

from unyt import mh

from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser


def critical_density_DVS2012(
    T_K: float = 10.0 ** 7.5,
    M_gas: float = 7.0e4,
    N_ngb: float = 48.0,
    X_H: float = 0.752,
    f_t: float = 10.0,
    mu: float = 0.6,
) -> float:
    """
    Computes the critical density from C. Dalla Vecchia & J. Schaye 2012
    (2012MNRAS.426..140D), below which the gas is subject to (strong) radiation energy
    losses.

    Parameters
    ----------

    T_K: float
        Heating temperature in SNII / AGN feedback [in K]

    M_gas: float
        Gas mass resolution [in Solar masses]

    N_ngb: Target number of gas neighbours in the stellar / BH kernel

    X_H: Average mass fraction of hydrogen

    f_t: float
        The ratio of the cooling time-scale (at temperature T_K) to the sound crossing
        time-scale (through the SPH kernel corresponding to M_gas and n_Hc)

    mu: Mean molecular weight of the heated gas (0.6 assumes the gas is fully ionised)

    Returns
    -------

    n_Hc: float
        The critical density expressed in hydrogen particles per cubic centimetre
    """

    f_X = X_H / (1.0 + X_H) / (1.0 + 3.0 * X_H)  # Eq. 14 in DV&S2012
    g = np.power(X_H, -1.0 / 3.0) * f_X

    # Critical density
    n_Hc = (
        31.0
        * np.power(T_K / 10.0 ** 7.5, 3.0 / 2.0)
        * np.power(f_t / 10.0, -3.0 / 2.0)
        * np.power(M_gas / 7.0e4, -1.0 / 2.0)
        * np.power(N_ngb / 48.0, -1.0 / 2.0)
        * np.power(mu / 0.6, -9.0 / 4.0)
        * np.power(g / 0.14, 3.0 / 2.0)
    )  # Eq. 18 in DV&S2012

    return n_Hc


arguments = ScriptArgumentParser(
    description="Creates a plot showing the distribution of the gas densities recorded "
    "when the gas was last heated by SNII, split by redshift"
)

snapshot_filenames = [
    f"{directory}/{snapshot}"
    for directory, snapshot in zip(arguments.directory_list, arguments.snapshot_list)
]

names = arguments.name_list
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

data = [load(snapshot_filename) for snapshot_filename in snapshot_filenames]
number_of_bins = 256

SNII_density_bins = unyt.unyt_array(
    np.logspace(-5, 6.5, number_of_bins), units="1/cm**3"
)
log_SNII_density_bin_width = np.log10(SNII_density_bins[1].value) - np.log10(
    SNII_density_bins[0].value
)
SNII_density_centers = 0.5 * (SNII_density_bins[1:] + SNII_density_bins[:-1])


# Begin plotting

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
axes = axes.flat

ax_dict = {
    "$z < 1$": axes[0],
    "$1 < z < 3$": axes[1],
    "$z > 3$": axes[2],
}

for label, ax in ax_dict.items():
    ax.loglog()
    ax.text(0.025, 1.0 - 0.025 * 3, label, transform=ax.transAxes, ha="left", va="top")

for color, (snapshot, name) in enumerate(zip(data, names)):

    stars_SNII_densities = snapshot.stars.densities_at_last_supernova_event.to(
        "g/cm**3"
    ) / mh.to("g")

    # swift-colibre master branch as of Feb 26 2021
    try:
        stars_SNII_redshifts = (
            1 / snapshot.stars.last_sniithermal_feedback_scale_factors.value - 1
        )
    # swift-colibre master prior to Feb 26 2021
    except AttributeError:
        stars_SNII_redshifts = (
            1 / snapshot.stars.last_sniifeedback_scale_factors.value - 1
        )

    gas_SNII_densities = snapshot.gas.densities_at_last_supernova_event.to(
        "g/cm**3"
    ) / mh.to("g")

    try:
        gas_SNII_redshifts = (
            1 / snapshot.gas.last_sniithermal_feedback_scale_factors.value - 1
        )
    except AttributeError:
        gas_SNII_redshifts = 1 / snapshot.gas.last_sniifeedback_scale_factors.value - 1

    # Limit only to those gas/stellar particles that were in fact heated by SNII
    stars_SNII_heated = stars_SNII_densities > 0.0
    gas_SNII_heated = gas_SNII_densities > 0.0

    # Select only those parts that were heated by SNII in the past
    stars_SNII_densities = stars_SNII_densities[stars_SNII_heated]
    stars_SNII_redshifts = stars_SNII_redshifts[stars_SNII_heated]
    gas_SNII_densities = gas_SNII_densities[gas_SNII_heated]
    gas_SNII_redshifts = gas_SNII_redshifts[gas_SNII_heated]

    # Segment SNII densities into redshift bins
    stars_SNII_densities_by_redshift = {
        "$z < 1$": stars_SNII_densities[stars_SNII_redshifts < 1],
        "$1 < z < 3$": stars_SNII_densities[
            np.logical_and(stars_SNII_redshifts > 1, stars_SNII_redshifts < 3)
        ],
        "$z > 3$": stars_SNII_densities[stars_SNII_redshifts > 3],
    }

    gas_SNII_densities_by_redshift = {
        "$z < 1$": gas_SNII_densities[gas_SNII_redshifts < 1],
        "$1 < z < 3$": gas_SNII_densities[
            np.logical_and(gas_SNII_redshifts > 1, gas_SNII_redshifts < 3)
        ],
        "$z > 3$": gas_SNII_densities[gas_SNII_redshifts > 3],
    }

    # Compute the critical density from DV&S2012
    SNII_heating_temperature = float(
        snapshot.metadata.parameters["COLIBREFeedback:SNII_delta_T_K"].decode("utf-8")
    )  # in K
    N_ngb_target = snapshot.metadata.hydro_scheme["Kernel target N_ngb"][0]
    X_H = snapshot.metadata.hydro_scheme["Hydrogen mass fraction"][0]
    M_gas = snapshot.metadata.initial_mass_table.gas.to("Msun")  # in Solar Masses

    n_crit = critical_density_DVS2012(
        T_K=SNII_heating_temperature, M_gas=M_gas.value, N_ngb=N_ngb_target, X_H=X_H
    )

    for redshift, ax in ax_dict.items():
        data = np.concatenate(
            [
                stars_SNII_densities_by_redshift[redshift],
                gas_SNII_densities_by_redshift[redshift],
            ]
        )

        H, _ = np.histogram(data, bins=SNII_density_bins)

        # Total number SNII-heated gas particles
        Num_of_obj = np.sum(H)

        # Check to avoid division by zero
        if Num_of_obj:
            y_points = H / log_SNII_density_bin_width / Num_of_obj
        else:
            y_points = np.zeros_like(H)

        ax.plot(
            SNII_density_centers,
            y_points,
            label=name,
            color=f"C{color}",
        )
        ax.axvline(
            np.median(data),
            color=f"C{color}",
            linestyle="dashed",
            zorder=-10,
            alpha=0.5,
        )

        # Add the DV&S2012 line
        ax.axvline(
            n_crit,
            color=f"C{color}",
            linestyle="dotted",
            zorder=-10,
            alpha=0.5,
        )

axes[0].legend(loc="upper right", markerfirst=False)
axes[2].set_xlabel(
    "Density of the gas heated by SNII $\\rho_{\\rm SNII}$ [$n_H$ cm$^{-3}$]"
)
axes[1].set_ylabel("$N_{\\rm bin}$ / d$\\log\\rho_{\\rm SNII}$ / $N_{\\rm total}$")

fig.savefig(f"{arguments.output_directory}/SNII_density_distribution.png")
