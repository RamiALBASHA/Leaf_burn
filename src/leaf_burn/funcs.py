from math import exp, log
from pathlib import Path

from hydroshoot import architecture, display
from openalea.mtg import mtg, traversal
from openalea.plantgl.all import Vector3, Translated, surface
from openalea.plantgl.scenegraph import Scene
from pandas import DataFrame

from sims.config import Pot, Config

cfg = Config()


def build_mtg(path_file: Path, is_show_scene: bool = True) -> (mtg.MTG, Scene):
    grapevine_mtg = architecture.vine_mtg(file_path=path_file)

    for v in traversal.iter_mtg2(grapevine_mtg, grapevine_mtg.root):
        architecture.vine_mtg_properties(grapevine_mtg, v)
        architecture.vine_mtg_geometry(grapevine_mtg, v)
        architecture.vine_transform(grapevine_mtg, v)

    # Display of the plant mock-up (result in 'fig_01_plant_mock_up.png')
    mtg_scene = display.visu(grapevine_mtg, def_elmnt_color_dict=True, scene=Scene(), view_result=is_show_scene)
    return grapevine_mtg, mtg_scene


def add_pots(g: mtg.MTG, pot: Pot) -> mtg.MTG:
    vids_base = list(g.property('baseXYZ').keys())
    shifts_y = [0, pot.radius_upper * 2] * int(len(vids_base) / 2)
    for shift_y, vid_base in zip(shifts_y, g.property('baseXYZ').keys()):
        vid_pot = g.add_component(complex_id=vid_base, label='other', edge_type='/')
        mesh = architecture.slim_cylinder(length=pot.height, radius_base=pot.radius_lower, radius_top=pot.radius_upper)
        x, y = g.node(vid_base).properties()['baseXYZ'][:2]

        mesh = Translated(Vector3([x, y - shift_y, 0]), mesh)
        g.node(vid_pot).geometry = mesh

    return g


def calc_local_wind_speed(wind_speed_ref: float, leaf_area_index: float, extinction_coefficient: float) -> float:
    """Calculates wind speed below a cumulative leaf area index from canopy top.

    Args:
        wind_speed_ref: (m s-1) wind speed at canopy top
        leaf_area_index: (m2 m-2) cumulative downwards leaf area index
        extinction_coefficient: (m2 m-2) extinction coefficient of wind speed

    Returns:
        (m s-1) wind speed below the given leaf area index

    References:
        Wang and Leuning (1998) https://doi.org/10.1016/S0168-1923(98)00061-6

    """
    return wind_speed_ref * exp(- extinction_coefficient * leaf_area_index)


def get_leaf_area_profile(g: mtg.MTG) -> DataFrame:
    conv_to_m = {'mm': 1.e-3, 'cm': 1.e-2, 'm': 1}[cfg.unit_digit]
    height_attr = cfg.leaf_height_attribute
    area_ground = cfg.spacing_on_row * cfg.spacing_between_rows * (conv_to_m ** 2)
    leaf_height, leaf_area = zip(*[(g.node(vid).properties()[height_attr][2],
                                    surface(g.node(vid).geometry) * (conv_to_m ** 2) * 2)  # twin pots
                                   for vid in architecture.get_leaves(g)])
    leaf_area = list(leaf_area)

    z_sorted = list(reversed(sorted(leaf_height)))
    df = DataFrame(dict(
        height=z_sorted,
        leaf_area=[leaf_area[z_sorted.index(z)] for z in leaf_height]),
        index=range(len(z_sorted)))
    df.loc[:, 'leaf_area_cum'] = df['leaf_area'].cumsum()
    df.loc[:, 'leaf_area_index_cum'] = df['leaf_area_cum'] / area_ground

    return df


def set_wind_speed_profile(g: mtg.MTG, wind_speed_ref: float) -> mtg.MTG:
    """sets wind speed through the canopy

    Args:
        g: single plant mtg
        wind_speed_ref: (m s-1) wind speed at reference height

    Returns:
        updated grapevine mtg
    """
    df = get_leaf_area_profile(g)
    height_attr = cfg.leaf_height_attribute
    k = cfg.extinction_coefficient_wind

    for vid in architecture.get_leaves(g):
        height = g.node(vid).properties()[height_attr][2]
        lai = df[df.loc[:, 'height'] == height]['leaf_area_index_cum'].iloc[0]
        g.node(vid).u = calc_local_wind_speed(
            wind_speed_ref=wind_speed_ref,
            leaf_area_index=lai,
            extinction_coefficient=k)

    return g


def calc_local_air_temperature(height: float, temperature_air_ref: float, temperature_ground: float) -> float:
    """Calculates the air temperature profile inside the canopy

    Args:
        height: (m) height above the ground
        temperature_air_ref: (°C) air temperature at reference height
        temperature_ground: (°C) ground temperature

    Returns:
        (°C) air temperature at given depth

    Notes:
        Temperature profiles deduced from normalized soil-air temperature difference date provided by
            Heilman et al. (1994) Fig 3., https://doi.org/10.1016/0168-1923(94)90102-3
    """
    return temperature_ground - log(max(1.e-6, height) / 1.e-6) / 14.604 * (temperature_ground - temperature_air_ref)


def set_air_temperature_profile(g: mtg.MTG, temperature_air_ref: float, temperature_ground: float) -> mtg.MTG:
    """Sets air temperature through the canopy

    Args:
        g: single plant mtg
        temperature_air_ref: (m s-1) air temperature at reference height
        temperature_ground: (m s-1) air temperature at ground surface

    Returns:
        updated grapevine mtg
    """
    conv_to_m = {'mm': 1.e-3, 'cm': 1.e-2, 'm': 1}[cfg.unit_digit]
    height_attr = cfg.leaf_height_attribute

    for vid in architecture.get_leaves(g):
        g.node(vid).Tac = calc_local_air_temperature(
            height=g.node(vid).properties()[height_attr][2] * conv_to_m,
            temperature_air_ref=temperature_air_ref,
            temperature_ground=temperature_ground)

    return g
