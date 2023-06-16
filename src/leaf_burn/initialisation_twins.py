from datetime import datetime
from typing import Tuple

from hydroshoot.architecture import add_soil_surface_mesh, get_leaves, get_mtg_base
from hydroshoot.energy import set_form_factors_simplified
from hydroshoot.initialisation import (remove_stem_geometry, add_rhyzosphere_concentric_cylinders, traversal,
                                       calc_nitrogen_distribution, set_photosynthetic_capacity)
from hydroshoot.io import HydroShootInputs, HydroShootHourlyInputs
from hydroshoot.irradiance import irradiance_distribution, hsCaribu, set_optical_properties
from hydroshoot.params import Params
from openalea.mtg.mtg import MTG

from leaf_burn.funcs import set_wind_speed_profile, set_temperature_profile


def init_model(g: MTG, g_clone: MTG, inputs: HydroShootInputs) -> Tuple[MTG, MTG]:
    params = inputs.params
    g.node(g.root).vid_collar = get_mtg_base(g, vtx_label=params.mtg_api.collar_label)
    g.node(g.root).vid_base = g.node(g.root).vid_collar

    # Add rhyzosphere concentric cylinders
    if params.soil.rhyzo_solution:
        if not any(item.startswith('rhyzo') for item in g.property('label').values()):
            g = add_rhyzosphere_concentric_cylinders(
                g=g,
                cylinders_radii=params.soil.rhyzo_radii,
                soil_dimensions=params.soil.soil_dimensions,
                soil_class=params.soil.soil_class,
                vtx_label=params.mtg_api.collar_label,
                length_conv=1. / params.simulation.conv_to_meter)

    # Add form factors
    if params.simulation.is_energy_budget:
        if inputs.form_factors is not None:
            for ff in ('ff_sky', 'ff_leaves', 'ff_soil'):
                g.properties()[ff] = inputs.form_factors[ff]
        else:
            if not all([s in g.property_names() for s in ('ff_sky', 'ff_leaves', 'ff_soil')]):
                print('Computing form factors...')
                g_clone = set_form_factors_simplified(
                    g=g_clone,
                    pattern=params.irradiance.pattern,
                    infinite=True,
                    leaf_lbl_prefix=params.mtg_api.leaf_lbl_prefix,
                    turtle_sectors=params.irradiance.turtle_sectors,
                    icosphere_level=params.irradiance.icosphere_level,
                    unit_scene_length=params.simulation.unit_scene_length)

                for ff in ('ff_sky', 'ff_leaves', 'ff_soil'):
                    g.properties()[ff] = {k: v for k, v in g_clone.property(ff).items() if k in g.VtxList(Scale=3)}

    # Initialize sap flow to 0
    for vtx_id in traversal.pre_order2(g, g.node(g.root).vid_base):
        g.node(vtx_id).Flux = 0.

    # Add soil surface
    if 'Soil' not in g.properties()['label'].values():
        side_length = inputs.soil_size if inputs.soil_size is not None and inputs.soil_size > 0 else 500.
        g = add_soil_surface_mesh(g=g, side_length=side_length)
        g_clone = add_soil_surface_mesh(g=g_clone, side_length=side_length)

    # Remove undesired geometry for light and energy calculations
    if not inputs.is_ppfd_interception_calculated:
        remove_stem_geometry(g)
        remove_stem_geometry(g_clone)

    # Attach optical properties to MTG elements
    g = set_optical_properties(
        g=g,
        wave_band='SW',
        leaf_lbl_prefix=params.mtg_api.leaf_lbl_prefix,
        stem_lbl_prefix=params.mtg_api.stem_lbl_prefix,
        opt_prop=params.irradiance.opt_prop)
    g_clone = set_optical_properties(
        g=g_clone,
        wave_band='SW',
        leaf_lbl_prefix=params.mtg_api.leaf_lbl_prefix,
        stem_lbl_prefix=params.mtg_api.stem_lbl_prefix,
        opt_prop=params.irradiance.opt_prop)

    # Calculate leaf Nitrogen per unit surface area according to Prieto et al. (2012)
    if 'Na' not in g.property_names():
        if inputs.leaf_nitrogen is not None:
            g.properties()['Na'] = inputs.leaf_nitrogen
        else:
            print('Computing Nitrogen profile...')
            inputs.gdd_since_budbreak = calc_nitrogen_distribution(
                g=g_clone,
                gdd_since_budbreak=inputs.gdd_since_budbreak,
                weather=inputs.weather,
                params=params)
            for prop in ('Ei10', 'Na'):
                g.properties()[prop] = {k: v for k, v in g_clone.property(prop).items() if
                                        k in get_leaves(g=g, leaf_lbl_prefix=params.mtg_api.leaf_lbl_prefix)}

    set_photosynthetic_capacity(
        g=g,
        photo_n_params=inputs.params.exchange.par_photo_N,
        deactivation_enthalopy=inputs.params.exchange.par_photo['dHd'],
        leaf_lbl_prefix=inputs.params.mtg_api.leaf_lbl_prefix)
    return g, g_clone


def init_hourly(g: MTG, g_clone: MTG, inputs_hourly: HydroShootHourlyInputs, leaf_ppfd: dict,
                params: Params) -> (MTG, float):
    # Add a date index to g
    g.date = datetime.strftime(inputs_hourly.date, "%Y%m%d%H%M%S")

    g = set_wind_speed_profile(g=g, wind_speed_ref=inputs_hourly.weather['u'].iloc[0])

    ##############################################################################
    g = set_temperature_profile(g=g, temperature_air_ref=inputs_hourly.weather['Tac'].iloc[0], temperature_ground=10)

    if leaf_ppfd is not None:
        diffuse_to_total_irradiance_ratio = leaf_ppfd[g.date]['diffuse_to_total_irradiance_ratio']
        g.properties()['Ei'] = leaf_ppfd[g.date]['Ei']
        g.properties()['Eabs'] = leaf_ppfd[g.date]['Eabs']
    else:
        # Compute irradiance distribution over the scene
        caribu_source, diffuse_to_total_irradiance_ratio = irradiance_distribution(
            meteo=inputs_hourly.weather,
            geo_location=params.simulation.geo_location,
            irradiance_unit=params.irradiance.E_type,
            time_zone=params.simulation.tzone,
            turtle_sectors=params.irradiance.turtle_sectors,
            turtle_format=params.irradiance.turtle_format,
            sun2scene=inputs_hourly.sun2scene,
            rotation_angle=params.planting.scene_rotation)

        # Compute irradiance interception and absorbtion
        g_clone, _ = hsCaribu(
            mtg=g_clone,
            unit_scene_length=params.simulation.unit_scene_length,
            source=caribu_source, direct=False,
            infinite=True, nz=50, ds=0.5,
            pattern=params.irradiance.pattern)

        g.properties()['Ei'] = {k: v for k, v in g_clone.property('Ei').items() if k in g.VtxList(Scale=3)}
        g.properties()['Eabs'] = {k: v for k, v in g_clone.property('Eabs').items() if k in g.VtxList(Scale=3)}

    g.properties()['Rg'] = {k: v / (0.48 * 4.6) for k, v in g.properties()['Ei'].items()}

    return g, diffuse_to_total_irradiance_ratio
