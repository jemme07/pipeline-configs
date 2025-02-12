"""
Registration of extra quantities for velociraptor catalogues (i.e. quantities
derived from the catalogue's internal properties).
This file calculates:
    + sSFR (30, 100 kpc) (specific_sfr_gas_{x}_kpc)
        This is the specific star formation rate of gas within those apertures.
        Only calculated for galaxies with a stellar mass greater than zero.
    + is_passive and is_active (30, 100 kpc) (is_passive_{x}_kpc)
        Boolean that determines whether or not a galaxy is passive. Marginal
        specific star formation rate is 1e-11 year^-1.
    + sfr_halo_mass (30, 100 kpc) (sfr_halo_mass_{x}_kpc)
        Star formation rate divided by halo mass with the star formation rate
        computed within apertures.
    + 12 + log(O/H) ({gas_sf_twelve_plus_log_OH_{x}_kpc, 30, 100 kpc)
        12 + log(O/H) based on metallicities. These should be removed at some point
        once velociraptor has a more sensible way of dealing with metallicity
        units.
    + metallicity_in_solar (star_metallicity_in_solar_{x}_kpc, 30, 100 kpc)
        Metallicity in solar units (relative to metal_mass_fraction).
    + stellar_mass_to_halo_mass_{x}_kpc for 30 and 100 kpc
        Stellar Mass / Halo Mass (both mass_200crit and mass_bn98) for 30 and 100 kpc
        apertures.
    + average of log of stellar birth densities (average_of_log_stellar_birth_density)
        velociraptor outputs the log of the quantity we need, so we take exp(...) of it
"""

# Define aperture size in kpc
aperture_sizes = [30, 100]

# Solar metal mass fraction used in Plöckinger S. & Schaye J. (2020)
solar_metal_mass_fraction = 0.0134

# Solar value for O/H
twelve_plus_log_OH_solar = 8.69

# Solar Fe abundance (from Wiersma et al 2009a)
solar_fe_abundance = 2.82e-5


def register_spesific_star_formation_rates(self, catalogue, aperture_sizes):

    # Lowest sSFR below which the galaxy is considered passive
    marginal_ssfr = unyt.unyt_quantity(1e-11, units=1 / unyt.year)

    # Loop over apertures
    for aperture_size in aperture_sizes:

        # Get the halo mass
        halo_mass = catalogue.masses.mass_200crit

        # Get stellar mass
        stellar_mass = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")
        # Need to mask out zeros, otherwise we get RuntimeWarnings
        good_stellar_mass = stellar_mass > unyt.unyt_quantity(0.0, stellar_mass.units)

        # Get star formation rate
        star_formation_rate = getattr(
            catalogue.apertures, f"sfr_gas_{aperture_size}_kpc"
        )

        # Compute specific star formation rate using the "good" stellar mass
        ssfr = unyt.unyt_array(np.zeros(len(star_formation_rate)), units=1 / unyt.year)
        ssfr[good_stellar_mass] = (
            star_formation_rate[good_stellar_mass] / stellar_mass[good_stellar_mass]
        )

        # Name (label) of the derived field
        ssfr.name = f"Specific SFR ({aperture_size} kpc)"

        # Mask for the passive objects
        is_passive = unyt.unyt_array(
            (ssfr < 1.01 * marginal_ssfr).astype(float), units="dimensionless"
        )
        is_passive.name = "Passive Fraction"

        # Mask for the active objects
        is_active = unyt.unyt_array(
            (ssfr > 1.01 * marginal_ssfr).astype(float), units="dimensionless"
        )
        is_active.name = "Active Fraction"

        # Get the specific star formation rate (per halo mass instead of stellar mass)
        sfr_M200 = star_formation_rate / halo_mass
        sfr_M200.name = "Star formation rate divided by halo mass"

        # Register derived fields with specific star formation rates
        setattr(self, f"specific_sfr_gas_{aperture_size}_kpc", ssfr)
        setattr(self, f"is_passive_{aperture_size}_kpc", is_passive)
        setattr(self, f"is_active_{aperture_size}_kpc", is_active)
        setattr(self, f"sfr_halo_mass_{aperture_size}_kpc", sfr_M200)

    return


def register_global_mask(self, catalogue):

    mfof = catalogue.masses.mass_fof
    m_bg = catalogue.masses.mass_interloper

    # ensure halos contribute < 10% of Mfof
    low_interloper_halos = m_bg < 0.1 * mfof

    setattr(
        self, f"low_interloper_halos", low_interloper_halos,
    )


def register_star_metallicities(self, catalogue, aperture_sizes, Z_sun):

    # Loop over apertures
    for aperture_size in aperture_sizes:

        try:
            metal_mass_fraction_star = (
                getattr(catalogue.apertures, f"zmet_star_{aperture_size}_kpc") / Z_sun
            )
            metal_mass_fraction_star.name = (
                f"Star Metallicity $Z_*$ rel. to "
                f"$Z_\\odot={Z_sun}$ ({aperture_size} kpc)"
            )
            setattr(
                self,
                f"star_metallicity_in_solar_{aperture_size}_kpc",
                metal_mass_fraction_star,
            )
        except AttributeError:
            pass

    return


def register_stellar_to_halo_mass_ratios(self, catalogue, aperture_sizes):

    # Loop over apertures
    for aperture_size in aperture_sizes:

        # Get the stellar mass in the aperture of a given size
        stellar_mass = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")

        # M200 critical
        halo_M200crit = catalogue.masses.mass_200crit
        smhm = stellar_mass / halo_M200crit
        smhm.name = f"$M_* / M_{{\\rm 200crit}}$ ({aperture_size} kpc)"
        setattr(self, f"stellar_mass_to_halo_mass_200crit_{aperture_size}_kpc", smhm)

        # BN98
        halo_MBN98 = catalogue.masses.mass_bn98
        smhm = stellar_mass / halo_MBN98
        smhm.name = f"$M_* / M_{{\\rm BN98}}$ ({aperture_size} kpc)"
        setattr(self, f"stellar_mass_to_halo_mass_bn98_{aperture_size}_kpc", smhm)

    return


def register_dust(self, catalogue, aperture_sizes):

    # Loop over apertures
    for aperture_size in aperture_sizes:

        # Metal mass fractions of the gas
        metal_frac = getattr(catalogue.apertures, f"zmet_gas_{aperture_size}_kpc")

        try:
            # Fetch dust fields
            dust_mass_silicates = getattr(
                catalogue.dust_masses, f"silicates_mass_{aperture_size}_kpc"
            )
            dust_mass_graphite = getattr(
                catalogue.dust_masses, f"graphite_mass_{aperture_size}_kpc"
            )
            # All dust mass
            dust_mass_total = dust_mass_graphite + dust_mass_silicates

        # In case run without dust
        except AttributeError:
            dust_mass_total = unyt.unyt_array(np.zeros_like(metal_frac), units="Msun")

        # Fetch gas mass
        gas_mass = getattr(catalogue.apertures, f"mass_gas_{aperture_size}_kpc")

        # Add label to the dust mass field
        dust_mass_total.name = f"$M_{{\\rm dust}}$ ({aperture_size} kpc)"

        # Compute dust to gas fraction
        dust_to_gas = dust_mass_total / gas_mass
        # Label for the dust-fraction derived field
        dust_to_gas.name = f"$\\mathcal{{DTG}}$ ({aperture_size} kpc)"

        # Compute to metal ratio
        dust_to_metals = dust_to_gas / metal_frac
        dust_to_metals.name = f"$\\mathcal{{DTM}}$ ({aperture_size} kpc)"

        # Compute dust to stellar ratio
        dust_to_stars = dust_mass_total / catalogue.apertures.mass_star_100_kpc
        dust_to_stars.name = f"$M_{{\\rm dust}}/M_*$ ({aperture_size} kpc)"

        # Register derived fields with dust
        setattr(self, f"total_dust_masses_{aperture_size}_kpc", dust_mass_total)
        setattr(self, f"dust_to_metal_ratio_{aperture_size}_kpc", dust_to_metals)
        setattr(self, f"dust_to_gas_ratio_{aperture_size}_kpc", dust_to_gas)
        setattr(self, f"dust_to_stellar_ratio_{aperture_size}_kpc", dust_to_stars)

    return


def register_oxygen_to_hydrogen(self, catalogue, aperture_sizes):
    # Loop over aperture average-of-linear O-abundances
    for aperture_size in aperture_sizes:
        # register lnearly averaged O abundances
        for short_phase, long_phase in zip(
            ["_total", ""], ["Total (Diffuse + Dust)", "Diffuse"]
        ):
            # Fetch O over H times gas mass computed in apertures.  The factor of 16 (the
            # mass ratio between O and H) has already been accounted for.
            log_O_over_H_times_gas_mass = getattr(
                catalogue.lin_element_ratios_times_masses,
                f"lin_O_over_H{short_phase}_times_gas_mass_{aperture_size}_kpc",
            )
            # Fetch gas mass in apertures
            gas_cold_dense_mass = getattr(
                catalogue.cold_dense_gas_properties,
                f"cold_dense_gas_mass_{aperture_size}_kpc",
            )

            # Compute gas-mass weighted O over H
            log_O_over_H = unyt.unyt_array(
                np.zeros_like(gas_cold_dense_mass), "dimensionless"
            )
            # Avoid division by zero
            mask = gas_cold_dense_mass > 0.0 * gas_cold_dense_mass.units
            log_O_over_H[mask] = np.log10(
                log_O_over_H_times_gas_mass[mask] / gas_cold_dense_mass[mask]
            )

            # Convert to units used in observations
            O_abundance = unyt.unyt_array(12 + log_O_over_H, "dimensionless")
            O_abundance.name = f"SF {long_phase} Gas $12+\\log_{{10}}({{\\rm O/H}})$ ({aperture_size} kpc)"

            # Register the field
            setattr(
                self,
                f"gas_o_abundance{short_phase}_avglin_{aperture_size}_kpc",
                O_abundance,
            )
            setattr(self, f"has_cold_dense_gas_{aperture_size}_kpc", mask)

        # register average-of-log O-abundances (high and low particle floors)
        for floor, floor_label in zip(
            ["low", "high"], ["Min = $10^{{-4}}$", "Min = $10^{{-3}}$"]
        ):
            # Fetch O over H times gas mass computed in apertures.  The factor of 16 (the
            # mass ratio between O and H) has already been accounted for.
            log_O_over_H_times_gas_mass = getattr(
                catalogue.log_element_ratios_times_masses,
                f"log_O_over_H_times_gas_mass_{floor}floor_{aperture_size}_kpc",
            )

            # Fetch gas mass in apertures
            gas_cold_dense_mass = getattr(
                catalogue.cold_dense_gas_properties,
                f"cold_dense_gas_mass_{aperture_size}_kpc",
            )

            # Compute gas-mass weighted O over H
            log_O_over_H = unyt.unyt_array(
                np.zeros_like(gas_cold_dense_mass), "dimensionless"
            )
            # Avoid division by zero
            mask = gas_cold_dense_mass > 0.0 * gas_cold_dense_mass.units
            log_O_over_H[mask] = (
                log_O_over_H_times_gas_mass[mask] / gas_cold_dense_mass[mask]
            )

            # Convert to units used in observations
            O_abundance = unyt.unyt_array(12 + log_O_over_H, "dimensionless")
            O_abundance.name = f"SF Gas Diffuse $12+\\log_{{10}}({{\\rm O/H}})$ ({floor_label}, {aperture_size} kpc)"

            # Register the field
            setattr(
                self, f"gas_o_abundance_avglog_{floor}_{aperture_size}_kpc", O_abundance
            )

    return


def register_iron_to_hydrogen(self, catalogue, aperture_sizes, fe_solar_abundance):
    # Loop over apertures
    for aperture_size in aperture_sizes:
        # Fetch linear Fe over H times stellar mass computed in apertures. The
        # mass ratio between Fe and H has already been accounted for.
        lin_Fe_over_H_times_star_mass = getattr(
            catalogue.lin_element_ratios_times_masses,
            f"lin_Fe_over_H_times_star_mass_{aperture_size}_kpc",
        )
        # Fetch stellar mass in apertures
        star_mass = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")

        # Compute stellar-mass weighted Fe over H
        Fe_over_H = unyt.unyt_array(np.zeros_like(star_mass), "dimensionless")
        # Avoid division by zero
        mask = star_mass > 0.0 * star_mass.units
        Fe_over_H[mask] = lin_Fe_over_H_times_star_mass[mask] / star_mass[mask]
        # Convert to units used in observations
        Fe_abundance = unyt.unyt_array(Fe_over_H / fe_solar_abundance, "dimensionless")
        Fe_abundance.name = f"Stellar $10^{{\\rm [Fe/H]}}$ ({aperture_size} kpc)"

        # Register the field
        setattr(self, f"star_fe_abundance_avglin_{aperture_size}_kpc", Fe_abundance)

        # register average-of-log Fe-abundances (high and low particle floors)
        for floor, floor_label in zip(
            ["low", "high"], ["Min = $10^{{-4}}$", "Min = $10^{{-3}}$"]
        ):

            # Fetch Fe over H times stellar mass computed in apertures. The
            # mass ratio between Fe and H has already been accounted for.
            log_Fe_over_H_times_star_mass = getattr(
                catalogue.log_element_ratios_times_masses,
                f"log_Fe_over_H_times_star_mass_{floor}floor_{aperture_size}_kpc",
            )
            # Fetch stellar mass in apertures
            star_mass = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")

            # Compute stellar-mass weighted Fe over H
            Fe_over_H = unyt.unyt_array(np.zeros_like(star_mass), "dimensionless")
            # Avoid division by zero
            mask = star_mass > 0.0 * star_mass.units
            Fe_over_H[mask] = pow(
                10.0, log_Fe_over_H_times_star_mass[mask] / star_mass[mask]
            )
            # Convert to units used in observations
            Fe_abundance = unyt.unyt_array(
                Fe_over_H / fe_solar_abundance, "dimensionless"
            )
            Fe_abundance.name = (
                f"Stellar $10^{{\\rm [Fe/H]}}$ ({floor_label}, {aperture_size} kpc)"
            )

            # Register the field
            setattr(
                self,
                f"star_fe_abundance_avglog_{floor}_{aperture_size}_kpc",
                Fe_abundance,
            )

    return


def register_cold_dense_gas_metallicity(
    self, catalogue, aperture_sizes, Z_sun, log_twelve_plus_logOH_solar
):
    # Loop over apertures
    for aperture_size in aperture_sizes:
        # Fetch gas metal masses in apertures
        lin_diffuse_metallicity = getattr(
            catalogue.cold_dense_gas_properties,
            f"cold_dense_diffuse_metal_mass_{aperture_size}_kpc",
        )
        # Fetch gas mass in apertures
        gas_cold_dense_mass = getattr(
            catalogue.cold_dense_gas_properties,
            f"cold_dense_gas_mass_{aperture_size}_kpc",
        )

        # Compute gas-mass weighted metallicity, floor at a non-zero metallicity 1e-8
        twelve_plus_logOH = unyt.unyt_array(
            np.zeros_like(gas_cold_dense_mass) + 1e-8, "dimensionless"
        )
        # Avoid division by zero
        mask = gas_cold_dense_mass > 0.0 * gas_cold_dense_mass.units

        # convert absolute metallicity to 12+log10(O/H), assuming solar abundance patterns
        twelve_plus_logOH[mask] = (
            np.log10(
                lin_diffuse_metallicity[mask] / (Z_sun * gas_cold_dense_mass[mask])
            )
            + log_twelve_plus_logOH_solar
        )

        # Register the field
        O_abundance = unyt.unyt_array(twelve_plus_logOH, "dimensionless")
        O_abundance.name = (
            f"SF Gas $12+\\log_{{10}}({{\\rm O/H}})$ from $Z$ ({aperture_size} kpc)"
        )

        setattr(self, f"gas_o_abundance_fromz_avglin_{aperture_size}_kpc", O_abundance)

    return


def register_hi_masses(self, catalogue, aperture_sizes):

    # Loop over aperture sizes
    for aperture_size in aperture_sizes:

        # Fetch HI mass
        HI_mass = getattr(
            catalogue.gas_hydrogen_species_masses, f"HI_mass_{aperture_size}_kpc"
        )

        # Label of the derived field
        HI_mass.name = f"$M_{{\\rm HI}}$ ({aperture_size} kpc)"

        # Register derived field
        setattr(self, f"gas_HI_mass_{aperture_size}_kpc", HI_mass)

    return


def register_dust_to_hi_ratio(self, catalogue, aperture_sizes):

    # Loop over aperture sizes
    for aperture_size in aperture_sizes:

        # Fetch HI mass in apertures
        HI_mass = getattr(
            catalogue.gas_hydrogen_species_masses, f"HI_mass_{aperture_size}_kpc"
        )

        try:
            # Fetch dust fields
            dust_mass_silicates = getattr(
                catalogue.dust_masses, f"silicates_mass_{aperture_size}_kpc"
            )
            dust_mass_graphite = getattr(
                catalogue.dust_masses, f"graphite_mass_{aperture_size}_kpc"
            )
            # All dust mass
            dust_mass_total = dust_mass_graphite + dust_mass_silicates

        # In case run without dust
        except AttributeError:
            dust_mass_total = unyt.unyt_array(np.zeros_like(HI_mass), units="Msun")

        # Compute dust-to-HI mass ratios
        dust_to_hi_ratio = unyt.unyt_array(np.zeros_like(HI_mass), "dimensionless")
        # Avoid division by zero
        mask = HI_mass > 0.0 * HI_mass.units
        dust_to_hi_ratio[mask] = dust_mass_total[mask] / HI_mass[mask]

        # Add label
        dust_to_hi_ratio.name = f"M_{{\\rm dust}}/M_{{\\rm HI}} ({aperture_size} kpc)"

        # Register field
        setattr(self, f"gas_dust_to_hi_ratio_{aperture_size}_kpc", dust_to_hi_ratio)

    return


def register_h2_masses(self, catalogue, aperture_sizes):

    # Loop over aperture sizes
    for aperture_size in aperture_sizes:

        # Fetch H2 mass
        H2_mass = getattr(
            catalogue.gas_hydrogen_species_masses, f"H2_mass_{aperture_size}_kpc"
        )

        # Label of the derived field
        H2_mass.name = f"$M_{{\\rm H_2}}$ ({aperture_size} kpc)"

        # Compute H2 mass with correction due to He
        He_mass = getattr(catalogue.gas_H_and_He_masses, f"He_mass_{aperture_size}_kpc")
        H_mass = getattr(catalogue.gas_H_and_He_masses, f"H_mass_{aperture_size}_kpc")

        H2_mass_with_He = H2_mass * (1.0 + He_mass / H_mass)

        # Add label
        H2_mass_with_He.name = f"$M_{{\\rm H_2}}$ (incl. He, {aperture_size} kpc)"

        # Register field
        setattr(self, f"gas_H2_mass_{aperture_size}_kpc", H2_mass)
        setattr(self, f"gas_H2_plus_He_mass_{aperture_size}_kpc", H2_mass_with_He)

    return


def register_cold_gas_mass_ratios(self, catalogue, aperture_sizes):

    # Loop over aperture sizes
    for aperture_size in aperture_sizes:

        # Fetch HI and H2 masses in apertures of a given size
        HI_mass = getattr(
            catalogue.gas_hydrogen_species_masses, f"HI_mass_{aperture_size}_kpc"
        )
        H2_mass = getattr(
            catalogue.gas_hydrogen_species_masses, f"H2_mass_{aperture_size}_kpc"
        )

        # Compute neutral H mass (HI + H2)
        neutral_H_mass = HI_mass + H2_mass
        # Add label
        neutral_H_mass.name = f"$M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"

        # Fetch total stellar mass
        stellar_mass = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")
        # Fetch mass of star-forming gas
        sf_mass = getattr(catalogue.apertures, f"mass_gas_sf_{aperture_size}_kpc")

        # Compute neutral H mass to stellar mass ratio
        neutral_H_to_stellar_fraction = neutral_H_mass / stellar_mass
        neutral_H_to_stellar_fraction.name = (
            f"$M_{{\\rm HI + H_2}} / M_*$ ({aperture_size} kpc)"
        )

        # Compute molecular H mass to molecular plus stellar mass fraction
        molecular_H_to_molecular_plus_stellar_fraction = H2_mass / (
            H2_mass + stellar_mass
        )
        molecular_H_to_molecular_plus_stellar_fraction.name = (
            f"$M_{{\\rm H_2}} / (M_* + M_{{\\rm H_2}})$ ({aperture_size} kpc)"
        )

        # Compute molecular H mass to neutral H mass ratio
        molecular_H_to_neutral_fraction = H2_mass / neutral_H_mass
        molecular_H_to_neutral_fraction.name = (
            f"$M_{{\\rm H_2}} / M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"
        )

        # Compute neutral H mass to baryonic mass fraction
        neutral_H_to_baryonic_fraction = neutral_H_mass / (
            neutral_H_mass + stellar_mass
        )
        neutral_H_to_baryonic_fraction.name = (
            f"$M_{{\\rm HI + H_2}}/((M_*+ M_{{\\rm HI + H_2}})$ ({aperture_size} kpc)"
        )

        # Compute HI mass to neutral H mass ratio
        HI_to_neutral_H_fraction = HI_mass / neutral_H_mass
        HI_to_neutral_H_fraction.name = (
            f"$M_{{\\rm HI}}/M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"
        )

        # Compute H2 mass to neutral H mass ratio
        H2_to_neutral_H_fraction = H2_mass / neutral_H_mass
        H2_to_neutral_H_fraction.name = (
            f"$M_{{\\rm H_2}}/M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"
        )

        # Compute SF mass to SF mass plus stellar mass ratio
        sf_to_sf_plus_stellar_fraction = unyt.unyt_array(
            np.zeros_like(neutral_H_mass), units="dimensionless"
        )
        # Select only good mass
        star_plus_sf_mass_mask = sf_mass + stellar_mass > 0.0 * sf_mass.units
        sf_to_sf_plus_stellar_fraction[star_plus_sf_mass_mask] = sf_mass[
            star_plus_sf_mass_mask
        ] / (sf_mass[star_plus_sf_mass_mask] + stellar_mass[star_plus_sf_mass_mask])
        sf_to_sf_plus_stellar_fraction.name = (
            f"$M_{{\\rm SF}}/(M_{{\\rm SF}} + M_*)$ ({aperture_size} kpc)"
        )

        # Select only the star-forming gas mass that is greater than zero
        sf_mask = sf_mass > 0.0 * sf_mass.units

        # Compute neutral H mass to SF gas mass ratio
        neutral_H_to_sf_fraction = unyt.unyt_array(
            np.zeros_like(neutral_H_mass), units="dimensionless"
        )
        neutral_H_to_sf_fraction[sf_mask] = neutral_H_mass[sf_mask] / sf_mass[sf_mask]
        neutral_H_to_sf_fraction.name = (
            f"$M_{{\\rm HI + H_2}}/M_{{\\rm SF}}$ ({aperture_size} kpc)"
        )

        # Compute HI mass to SF gas mass ratio
        HI_to_sf_fraction = unyt.unyt_array(
            np.zeros_like(HI_mass), units="dimensionless"
        )
        HI_to_sf_fraction[sf_mask] = HI_mass[sf_mask] / sf_mass[sf_mask]
        HI_to_sf_fraction.name = f"$M_{{\\rm HI}}/M_{{\\rm SF}}$ ({aperture_size} kpc)"

        # Compute H2 mass to SF gas mass ratio
        H2_to_sf_fraction = unyt.unyt_array(
            np.zeros_like(H2_mass), units="dimensionless"
        )
        H2_to_sf_fraction[sf_mask] = H2_mass[sf_mask] / sf_mass[sf_mask]
        H2_to_sf_fraction.name = f"$M_{{\\rm H_2}}/M_{{\\rm SF}}$ ({aperture_size} kpc)"

        # Compute SF gas mss to stellar mass ratio
        sf_to_stellar_fraction = unyt.unyt_array(
            np.zeros_like(neutral_H_mass), units="dimensionless"
        )
        # Select only good stellar mass
        m_star_mask = stellar_mass > 0.0 * stellar_mass.units
        sf_to_stellar_fraction[m_star_mask] = (
            sf_mass[m_star_mask] / stellar_mass[m_star_mask]
        )
        sf_to_stellar_fraction.name = f"$M_{{\\rm SF}}/M_*$ ({aperture_size} kpc)"

        # Finally, register all the above fields
        setattr(
            self, f"gas_neutral_H_mass_{aperture_size}_kpc", neutral_H_mass,
        )

        setattr(
            self,
            f"gas_neutral_H_to_stellar_fraction_{aperture_size}_kpc",
            neutral_H_to_stellar_fraction,
        )
        setattr(
            self,
            f"gas_molecular_H_to_molecular_plus_stellar_fraction_{aperture_size}_kpc",
            molecular_H_to_molecular_plus_stellar_fraction,
        )
        setattr(
            self,
            f"gas_molecular_H_to_neutral_fraction_{aperture_size}_kpc",
            molecular_H_to_neutral_fraction,
        )
        setattr(
            self,
            f"gas_neutral_H_to_baryonic_fraction_{aperture_size}_kpc",
            neutral_H_to_baryonic_fraction,
        )
        setattr(
            self,
            f"gas_HI_to_neutral_H_fraction_{aperture_size}_kpc",
            HI_to_neutral_H_fraction,
        )
        setattr(
            self,
            f"gas_H2_to_neutral_H_fraction_{aperture_size}_kpc",
            H2_to_neutral_H_fraction,
        )
        setattr(
            self,
            f"gas_sf_to_sf_plus_stellar_fraction_{aperture_size}_kpc",
            sf_to_sf_plus_stellar_fraction,
        )
        setattr(
            self,
            f"gas_neutral_H_to_sf_fraction_{aperture_size}_kpc",
            neutral_H_to_sf_fraction,
        )
        setattr(self, f"gas_HI_to_sf_fraction_{aperture_size}_kpc", HI_to_sf_fraction)
        setattr(self, f"gas_H2_to_sf_fraction_{aperture_size}_kpc", H2_to_sf_fraction)
        setattr(
            self,
            f"gas_sf_to_stellar_fraction_{aperture_size}_kpc",
            sf_to_stellar_fraction,
        )
        setattr(self, f"has_neutral_gas_{aperture_size}_kpc", neutral_H_mass > 0.0)

    return


def register_species_fractions(self, catalogue, aperture_sizes):

    # Loop over aperture sizes
    for aperture_size in aperture_sizes:

        # Compute galaxy area (pi r^2)
        gal_area = (
            2
            * np.pi
            * getattr(
                catalogue.projected_apertures,
                f"projected_1_rhalfmass_star_{aperture_size}_kpc",
            )
            ** 2
        )

        # Stellar mass
        M_star = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")
        M_star_projected = getattr(
            catalogue.projected_apertures, f"projected_1_mass_star_{aperture_size}_kpc"
        )

        # Selection functions for the xGASS and xCOLDGASS surveys, used for the H species
        # fraction comparison. Note these are identical mass selections, but are separated
        # to keep survey selections explicit and to allow more detailed selection criteria
        # to be added for each.

        self.xgass_galaxy_selection = np.logical_and(
            M_star > unyt.unyt_quantity(10 ** 9, "Solar_Mass"),
            M_star < unyt.unyt_quantity(10 ** (11.5), "Solar_Mass"),
        )
        self.xcoldgass_galaxy_selection = np.logical_and(
            M_star > unyt.unyt_quantity(10 ** 9, "Solar_Mass"),
            M_star < unyt.unyt_quantity(10 ** (11.5), "Solar_Mass"),
        )

        # Register stellar mass density in apertures
        mu_star = M_star_projected / gal_area
        mu_star.name = f"$M_{{*, {aperture_size} {{\\rm kpc}}}} / \\pi R_{{*, {aperture_size} {{\\rm kpc}}}}^2$"

        # Atomic hydrogen mass in apertures
        HI_mass = getattr(
            catalogue.gas_hydrogen_species_masses, f"HI_mass_{aperture_size}_kpc"
        )
        HI_mass.name = f"HI Mass ({aperture_size} kpc)"

        # Molecular hydrogen mass in apertures
        H2_mass = getattr(
            catalogue.gas_hydrogen_species_masses, f"H2_mass_{aperture_size}_kpc"
        )
        H2_mass.name = f"H$_2$ Mass ({aperture_size} kpc)"

        # Compute H2 mass with correction due to He
        He_mass = getattr(catalogue.gas_H_and_He_masses, f"He_mass_{aperture_size}_kpc")
        H_mass = getattr(catalogue.gas_H_and_He_masses, f"H_mass_{aperture_size}_kpc")

        H2_mass_with_He = H2_mass * (1.0 + He_mass / H_mass)
        H2_mass_with_He.name = f"$M_{{\\rm H_2}}$ (incl. He, {aperture_size} kpc)"

        # Atomic hydrogen to stellar mass in apertures
        hi_to_stellar_mass = HI_mass / M_star
        hi_to_stellar_mass.name = f"$M_{{\\rm HI}} / M_*$ ({aperture_size} kpc)"

        # Molecular hydrogen mass to stellar mass in apertures
        h2_to_stellar_mass = H2_mass / M_star
        h2_to_stellar_mass.name = f"$M_{{\\rm H_2}} / M_*$ ({aperture_size} kpc)"

        # Molecular hydrogen mass to stellar mass in apertures
        h2_plus_he_to_stellar_mass = H2_mass_with_He / M_star
        h2_plus_he_to_stellar_mass.name = (
            f"$M_{{\\rm H_2}} / M_*$ (incl. He, {aperture_size} kpc)"
        )

        # Neutral H / stellar mass
        neutral_to_stellar_mass = hi_to_stellar_mass + h2_to_stellar_mass
        neutral_to_stellar_mass.name = (
            f"$M_{{\\rm HI + H_2}} / M_*$ ({aperture_size} kpc)"
        )

        # Register all derived fields
        setattr(self, f"mu_star_{aperture_size}_kpc", mu_star)

        setattr(self, f"hi_to_stellar_mass_{aperture_size}_kpc", hi_to_stellar_mass)
        setattr(self, f"h2_to_stellar_mass_{aperture_size}_kpc", h2_to_stellar_mass)
        setattr(
            self,
            f"h2_plus_he_to_stellar_mass_{aperture_size}_kpc",
            h2_plus_he_to_stellar_mass,
        )
        setattr(
            self,
            f"neutral_to_stellar_mass_{aperture_size}_kpc",
            neutral_to_stellar_mass,
        )

    return


def register_stellar_birth_densities(self, catalogue):

    try:
        average_log_n_b = catalogue.stellar_birth_densities.logaverage
        stellar_mass = catalogue.apertures.mass_star_100_kpc

        exp_average_log_n_b = unyt.unyt_array(
            np.exp(average_log_n_b), units=average_log_n_b.units
        )

        # Ensure haloes with zero stellar mass have zero stellar birth densities
        no_stellar_mass = stellar_mass <= unyt.unyt_quantity(0.0, stellar_mass.units)
        exp_average_log_n_b[no_stellar_mass] = unyt.unyt_quantity(
            0.0, average_log_n_b.units
        )

        # Label of the derived field
        exp_average_log_n_b.name = "Stellar Birth Density (average of log)"

        # Register the derived field
        setattr(self, "average_of_log_stellar_birth_density", exp_average_log_n_b)

    # In case stellar birth densities are not present in the catalogue
    except AttributeError:
        pass

    return


# Register derived fields
register_spesific_star_formation_rates(self, catalogue, aperture_sizes)
register_star_metallicities(self, catalogue, aperture_sizes, solar_metal_mass_fraction)
register_stellar_to_halo_mass_ratios(self, catalogue, aperture_sizes)
register_dust(self, catalogue, aperture_sizes)
register_oxygen_to_hydrogen(self, catalogue, aperture_sizes)
register_cold_dense_gas_metallicity(
    self, catalogue, aperture_sizes, solar_metal_mass_fraction, twelve_plus_log_OH_solar
)
register_iron_to_hydrogen(self, catalogue, aperture_sizes, solar_fe_abundance)
register_hi_masses(self, catalogue, aperture_sizes)
register_h2_masses(self, catalogue, aperture_sizes)
register_dust_to_hi_ratio(self, catalogue, aperture_sizes)
register_cold_gas_mass_ratios(self, catalogue, aperture_sizes)
register_species_fractions(self, catalogue, aperture_sizes)
register_stellar_birth_densities(self, catalogue)
register_global_mask(self, catalogue)
