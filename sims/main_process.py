from datetime import datetime
from itertools import product
from json import load
from multiprocessing import Pool
from typing import Iterable

from hydroshoot.architecture import load_mtg

from leaf_burn import hydroshoot_wrapper
from sims.config import Config

cfg = Config()


def run_hydroshoot(id_plant: str, is_cst_air_temperature: bool, is_cst_wind_speed: bool, scenario_id: str):
    path_preprocessed_dir = cfg.path_preprocessed_data / id_plant

    g, scene = load_mtg(
        path_mtg=str(path_preprocessed_dir / f'initial_mtg_{id_plant}.pckl'),
        path_geometry=str(path_preprocessed_dir / f'geometry_{id_plant}.bgeom'))

    with open(path_preprocessed_dir / f'{id_plant}_static.json') as f:
        static_inputs = load(f)
    with open(path_preprocessed_dir / f'{id_plant}_dynamic.json') as f:
        dynamic_inputs = load(f)

    scenario_val = cfg.scenarios[scenario_id]
    params = cfg.params
    params['exchange']['par_gs']['D0'] = scenario_val['d0']
    params['exchange']['par_photo']['photo_inhibition'] = scenario_val['photo_inhibition']

    path_output = cfg.path_output_dir / scenario_id / (
        f'{"tcst" if is_cst_air_temperature else "tvar"}_{"ucst" if is_cst_wind_speed else "uvar"}') / id_plant
    path_output.mkdir(parents=True, exist_ok=True)

    hydroshoot_wrapper.run(
        g=g,
        wd=path_preprocessed_dir.parent,
        path_weather=cfg.path_weather,
        params=params,
        is_cst_air_temperature_profile=is_cst_air_temperature,
        is_cst_wind_speed_profile=is_cst_wind_speed,
        plant_id=f'{id_plant}1',
        scene=scene,
        is_write_result=True,
        is_write_mtg=True,
        path_output_dir=path_output / 'time_series.csv',
        psi_soil_init=-0.1,
        form_factors=static_inputs['form_factors'],
        leaf_nitrogen=static_inputs['Na'],
        leaf_ppfd=dynamic_inputs)

    pass


def run_sims(args):
    return run_hydroshoot(*args)


def mp(sim_args: Iterable, nb_cpu: int = 2):
    with Pool(nb_cpu) as p:
        p.map(run_sims, sim_args)


if __name__ == '__main__':
    names_plant = ['belledenise', 'plantdec', 'poulsard', 'raboso', 'salice']
    is_constant_air_temperature = [False, True]
    is_constant_wind_speed = [False, True]
    time_on = datetime.now()
    mp(sim_args=product(names_plant, is_constant_air_temperature, is_constant_wind_speed, cfg.scenarios), nb_cpu=12)
    time_off = datetime.now()
    print(f"--- Total runtime: {(time_off - time_on).seconds} sec ---")
