import h5py
import numpy as np
from swiftsimio import load
from numba import jit
from unyt import g, cm, mp, erg, s


class interpolate:
    def init(self, X_Ray_table):
        self.table_file = X_Ray_table
        pass

    def load_table(self, band, observing_type):
        self.table = h5py.File(self.table_file, "r")
        self.X_Ray = self.table[band][observing_type][()]
        self.He_bins = self.table["/Bins/He_bins"][()]
        self.missing_elements = self.table["/Bins/Missing_element"][()]
        self.element_masses = self.table["Bins/Element_masses"][()]

        self.density_bins = self.table["/Bins/Density_bins/"][()]
        self.temperature_bins = self.table["/Bins/Temperature_bins/"][()]
        self.redshift_bins = self.table["/Bins/Redshift_bins"][()]
        self.dn = 0.2
        self.dT = 0.1
        self.dz = 0.2

        self.solar_metallicity = self.table["/Bins/Solar_metallicities/"][()]


@jit(nopython=True)
def get_index_1d(bins, subdata):
    eps = 1e-4
    delta = (len(bins) - 1) / (bins[-1] - bins[0])

    idx = np.zeros_like(subdata)
    dx = np.zeros_like(subdata)
    for i, x in enumerate(subdata):
        if x < bins[0] + eps:
            idx[i] = 0
            dx[i] = 0
        elif x < bins[-1] - eps:
            idx[i] = int((x - bins[0]) * delta)
            dx[i] = (x - bins[int(idx[i])]) * delta
        else:
            idx[i] = len(bins) - 2
            dx[i] = 1

    return idx, dx


@jit(nopython=True)
def get_index_1d_irregular(bins, subdata):
    eps = 1e-6

    idx = np.zeros_like(subdata)
    dx = np.zeros_like(subdata)

    for i, x in enumerate(subdata):
        if x < bins[0] + eps:
            idx[i] = 0
            dx[i] = 0
        elif x < bins[-1] - eps:
            min_idx = -1
            for i in range(len(bins)):
                if x - bins[i] <= 0:
                    min_idx = i - 1
                    break
            idx[i] = min_idx
            dx[i] = (x - bins[min_idx]) / (bins[min_idx + 1] - bins[min_idx])
        else:
            idx[i] = len(bins) - 2
            dx[i] = 1

    return idx, dx


@jit(nopython=True)
def get_table_interp(
    dn,
    dT,
    dx_T,
    dx_n,
    idx_T,
    idx_n,
    idx_he,
    dx_he,
    idx_z,
    dx_z,
    X_Ray,
    abundance_to_solar,
):
    f_n_T_Z = np.zeros_like(dx_n)

    for i in range(len(idx_n)):
        t_z = 1 - dx_z[i]
        d_z = dx_z[i]

        # Compute temperature offset relative to bin
        t_T = 1 - dx_T[i]
        d_T = dx_T[i]

        # Compute density offset relative to bin
        t_n = 1 - dx_n[i]
        d_n = dx_n[i]

        # Compute Helium offset relative to bin
        t_he = 1 - dx_he[i]
        d_he = dx_he[i]

        # Do the actual 4D linear interpolation
        f_n_T = (
            t_T
            * t_n
            * t_he
            * t_z
            * X_Ray[idx_z[i], idx_he[i], :, idx_T[i], idx_n[i]]
        )
        f_n_T += (
            t_T
            * t_n
            * d_he
            * t_z
            * X_Ray[idx_z[i], idx_he[i] + 1, :, idx_T[i], idx_n[i]]
        )
        f_n_T += (
            t_T
            * d_n
            * t_he
            * t_z
            * X_Ray[idx_z[i], idx_he[i], :, idx_T[i], idx_n[i] + 1]
        )
        f_n_T += (
            d_T
            * t_n
            * t_he
            * t_z
            * X_Ray[idx_z[i], idx_he[i], :, idx_T[i] + 1, idx_n[i]]
        )
        f_n_T += (
            t_T
            * d_n
            * d_he
            * t_z
            * X_Ray[idx_z[i], idx_he[i] + 1, :, idx_T[i], idx_n[i] + 1]
        )
        f_n_T += (
            d_T
            * t_n
            * d_he
            * t_z
            * X_Ray[idx_z[i], idx_he[i] + 1, :, idx_T[i] + 1, idx_n[i]]
        )
        f_n_T += (
            d_T
            * d_n
            * t_he
            * t_z
            * X_Ray[idx_z[i], idx_he[i], :, idx_T[i] + 1, idx_n[i] + 1]
        )
        f_n_T += (
            d_T
            * d_n
            * d_he
            * t_z
            * X_Ray[idx_z[i], idx_he[i] + 1, :, idx_T[i] + 1, idx_n[i] + 1]
        )

        f_n_T += (
            t_T
            * t_n
            * t_he
            * d_z
            * X_Ray[idx_z[i] + 1, idx_he[i], :, idx_T[i], idx_n[i]]
        )
        f_n_T += (
            t_T
            * t_n
            * d_he
            * d_z
            * X_Ray[idx_z[i] + 1, idx_he[i] + 1, :, idx_T[i], idx_n[i]]
        )
        f_n_T += (
            t_T
            * d_n
            * t_he
            * d_z
            * X_Ray[idx_z[i] + 1, idx_he[i], :, idx_T[i], idx_n[i] + 1]
        )
        f_n_T += (
            d_T
            * t_n
            * t_he
            * d_z
            * X_Ray[idx_z[i] + 1, idx_he[i], :, idx_T[i] + 1, idx_n[i]]
        )
        f_n_T += (
            t_T
            * d_n
            * d_he
            * d_z
            * X_Ray[idx_z[i] + 1, idx_he[i] + 1, :, idx_T[i], idx_n[i] + 1]
        )
        f_n_T += (
            d_T
            * t_n
            * d_he
            * d_z
            * X_Ray[idx_z[i] + 1, idx_he[i] + 1, :, idx_T[i] + 1, idx_n[i]]
        )
        f_n_T += (
            d_T
            * d_n
            * t_he
            * d_z
            * X_Ray[idx_z[i] + 1, idx_he[i], :, idx_T[i] + 1, idx_n[i] + 1]
        )
        f_n_T += (
            d_T
            * d_n
            * d_he
            * d_z
            * X_Ray[idx_z[i] + 1, idx_he[i] + 1, :, idx_T[i] + 1, idx_n[i] + 1]
        )

        # Add each metal contribution individually
        f_n_T_Z_temp = np.power(10, f_n_T[-1])
        for j in range(len(f_n_T) - 1):
            f_n_T_Z_temp += np.power(10, f_n_T[j]) * abundance_to_solar[i, j]

        f_n_T_Z[i] = np.log10(f_n_T_Z_temp)

    return f_n_T_Z


def interpolate_X_Ray(
    densities,
    temperatures,
    element_mass_fractions,
    redshift,
    masses,
    band=None,
    observing_type=None,
    fill_value=None,
    X_Ray_table=None,
):
    scale_factor = 1 / (1 + redshift)
    data_n = np.log10(
        element_mass_fractions.hydrogen
        * (1 / scale_factor ** 3)
        * densities.to(g * cm ** -3)
        / mp
    )
    data_T = np.log10(temperatures)
    volumes = masses / ((1 / scale_factor ** 3) * densities)

    if band == None:
        print(
            'Please specify the band you would like to generate emissivities for\n \
              Using the "band = " keyword\n\n \
              Available options are:\n \
              "erosita-low" (0.2-2.3 keV)\n \
              "erosita-high" (2.3-8.0 keV)\n \
              "ROSAT" (0.5-2.0 keV)'
        )
        raise KeyError

    if observing_type == None:
        print(
            'Please specify whether you would like to generate photon or energie emissivities\n \
              Using the "observing_type = " keyword\n\n \
              Available options are:\n \
              "energies"\n \
              "photons"\n \
              "energies_intrinsic"\n \
              "photons_intrinsic"'
        )
        raise KeyError

    if X_Ray_table == None:
        print("Please specify the location of the X-Ray table file")
        raise KeyError

    # Initialise interpolation class
    interp = interpolate(X_Ray_table)
    interp.load_table(band, observing_type)

    # Initialise the emissivity array which will be returned
    emissivities = np.zeros_like(data_n, dtype=float)

    # Create density mask, round to avoid numerical errors
    density_mask = (data_n >= np.round(interp.density_bins.min(), 1)) & (
        data_n <= np.round(interp.density_bins.max(), 1)
    )
    # Create temperature mask, round to avoid numerical errors
    temperature_mask = (
        data_T >= np.round(interp.temperature_bins.min(), 1)
    ) & (data_T <= np.round(interp.temperature_bins.max(), 1))

    # Combine masks
    joint_mask = density_mask & temperature_mask

    # Check if within density and temperature bounds
    density_bounds = np.sum(density_mask) == density_mask.shape[0]
    temperature_bounds = np.sum(temperature_mask) == temperature_mask.shape[0]
    if ~(density_bounds & temperature_bounds):
        # If no fill_value is set, return an error with some explanation
        if fill_value == None:
            raise ValueError(
                "Temperature or density are outside of the interpolation range and no fill_value is supplied\n \
                               Temperature ranges between log(T) = 5 and log(T) = 9.5\n \
                               Density ranges between log(nH) = -8 and log(nH) = 6\n \
                               Set the kwarg 'fill_value = some value' to set all particles outside of the interpolation range to 'some value'\n \
                               Or limit your particle data set to be within the interpolation range"
            )
        else:
            emissivities[~joint_mask] = fill_value

    # If only a single redshift is received, use it for all particles
    if redshift.size == 1:
        redshift = np.ones_like(data_n) * redshift

    mass_fraction = np.zeros((len(data_n[joint_mask]), 9))

    # get individual mass fraction
    mass_fraction[:, 0] = element_mass_fractions.hydrogen[joint_mask]
    mass_fraction[:, 1] = element_mass_fractions.helium[joint_mask]
    mass_fraction[:, 2] = element_mass_fractions.carbon[joint_mask]
    mass_fraction[:, 3] = element_mass_fractions.nitrogen[joint_mask]
    mass_fraction[:, 4] = element_mass_fractions.oxygen[joint_mask]
    mass_fraction[:, 5] = element_mass_fractions.neon[joint_mask]
    mass_fraction[:, 6] = element_mass_fractions.magnesium[joint_mask]
    mass_fraction[:, 7] = element_mass_fractions.silicon[joint_mask]
    mass_fraction[:, 8] = element_mass_fractions.iron[joint_mask]

    # Find density offsets
    idx_n, dx_n = get_index_1d(interp.density_bins, data_n[joint_mask])
    print("nh", data_n[joint_mask][0], idx_n[0], dx_n[0])

    # Find temperature offsets
    idx_T, dx_T = get_index_1d(interp.temperature_bins, data_T[joint_mask])
    print("T", data_T[joint_mask][0], idx_T[0], dx_T[0])
    # Find element offsets
    # mass of ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    # element_masses = [1.008, 4.003, 12.01, 14.01, 16., 20.18, 24.31, 28.09, 55.85]

    # Calculate the abundance wrt to solar
    abundances = (
        mass_fraction / np.expand_dims(mass_fraction[:, 0], axis=1)
    ) * (interp.element_masses[0] / np.array(interp.element_masses))

    # Calculate abundance offsets using solar abundances
    abundance_to_solar = abundances / 10 ** interp.solar_metallicity

    # Add columns for Calcium and Sulphur and add Iron at the end
    abundance_to_solar = np.c_[
        abundance_to_solar[:, :-1],
        abundance_to_solar[:, -2],
        abundance_to_solar[:, -2],
        abundance_to_solar[:, -1],
    ]

    # Find helium offsets
    idx_he, dx_he = get_index_1d_irregular(
        interp.He_bins, np.log10(abundances[:, 1])
    )
    print("he", idx_he[0], dx_he[0])

    # Find redshift offsets
    idx_z, dx_z = get_index_1d(interp.redshift_bins, redshift)
    print("Start interpolation")
    emissivities[joint_mask] = get_table_interp(
        interp.dn,
        interp.dT,
        dx_T,
        dx_n,
        idx_T.astype(int),
        idx_n.astype(int),
        idx_he.astype(int),
        dx_he,
        idx_z.astype(int),
        dx_z,
        interp.X_Ray,
        abundance_to_solar[:, 2:],
    )

    # Convert from erg cm^3 s^-1 to erg cm^-3 s^-1
    # To do so we multiply by nH^2, this is the actual nH not the nearest bin
    # It allows to extrapolate in density space without too much worry
    # log(emissivity * nH^2) = log(emissivity) + 2*log(nH)
    emissivities[joint_mask] += 2 * data_n[joint_mask]

    # Test if volumes have the correct units
    luminosities = np.zeros_like(emissivities)
    if "photon" in observing_type:
        luminosities[joint_mask] = (
            np.power(10, emissivities[joint_mask])
            * s ** -1
            * cm ** -3
            * volumes[joint_mask]
        )
    elif "energies" in observing_type:
        luminosities[joint_mask] = (
            np.power(10, emissivities[joint_mask])
            * erg
            * s ** -1
            * cm ** -3
            * volumes[joint_mask]
        )

    return luminosities
