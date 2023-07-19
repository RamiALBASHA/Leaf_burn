from datetime import date
from pathlib import Path

from hydroshoot.display import visu, DEFAULT_COLORS
from hydroshoot.irradiance import irradiance_distribution
from hydroshoot.params import Params
from openalea.mtg.mtg import MTG
from openalea.plantgl.all import Viewer
from openalea.plantgl.scenegraph import Scene
from pandas import read_csv, to_datetime

from leaf_burn import funcs
from sims import config


def show_the_sun(mtg_obj: MTG, name: str):
    cfg = config.Config()

    weather = read_csv(cfg.path_weather, sep=';', decimal='.', index_col='time')
    weather.index = to_datetime(weather.index)
    weather_heatwave = weather[weather.index.date == date(2019, 6, 28)]

    for expo_id, expo_angle in cfg.expositions:
        user_params = cfg.params
        user_params['planting']['row_angle_with_south'] = expo_angle

        params = Params('', user_params=user_params)
        scene = visu(mtg_obj, def_elmnt_color_dict=True, scene=Scene())
        for date_sim in weather_heatwave.index:
            caribu_source, diffuse_to_total_irradiance_ratio = irradiance_distribution(
                meteo=weather_heatwave[weather_heatwave.index == date_sim],
                geo_location=params.simulation.geo_location,
                irradiance_unit=params.irradiance.E_type,
                time_zone=params.simulation.tzone,
                turtle_sectors=params.irradiance.turtle_sectors,
                turtle_format=params.irradiance.turtle_format,
                sun2scene=scene,
                rotation_angle=params.planting.scene_rotation)
        Viewer.saveSnapshot(str(path_digit / f'{expo_id}_{name}_sun_course.png'))

    pass


if __name__ == '__main__':
    path_root = Path(__file__).parent
    path_digit = path_root / 'source/digit'

    element_colors = DEFAULT_COLORS
    element_colors.update({'other': (0, 0, 0)})
    for f in ('all_pots_demo', 'all_pots'):
        g, _ = funcs.build_mtg(path_file=path_digit / f'{f}.csv', is_show_scene=True)
        g = funcs.add_pots(g=g, pot=config.Pot())
        scene = visu(g, elmnt_color_dict=element_colors, scene=Scene(), view_result=True)
        Viewer.saveSnapshot(str(path_digit / f'{f}.png'))
        show_the_sun(mtg_obj=g, name=f)
