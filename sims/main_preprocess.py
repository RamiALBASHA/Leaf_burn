import re
from datetime import datetime
from itertools import product
from json import dump
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

from hydroshoot import io, display
from hydroshoot.architecture import mtg_save_geometry, save_mtg
from openalea.plantgl.scenegraph import Scene
from pandas import read_csv, DataFrame

from config import Config
from leaf_burn import funcs, initialisation_twins
from leaf_burn.utils import copy_mtg, extract_mtg

cfg = Config()


def run_preprocess(id_plant: str, df: DataFrame):
    path_digit = cfg.path_digit / f'digit_{id_plant}.csv'
    df[df["Plant"].apply(lambda x: x.startswith(id_plant))].to_csv(path_digit, sep=';', decimal='.', index=False)

    g, scene = funcs.build_mtg(path_file=path_digit, is_show_scene=False)
    # g = funcs.add_pots(g=g, pot=Pot())

    path_preprocessed_data = cfg.path_preprocessed_data / id_plant
    path_preprocessed_data.mkdir(parents=True, exist_ok=True)
    print("Computing 'static' data...")

    inputs = io.HydroShootInputs(
        path_project=cfg.path_root,
        path_weather=cfg.path_weather,
        user_params=cfg.params,
        scene=scene,
        is_write_result=False,
        path_output_file=Path(),
        psi_soil=-.01)
    io.verify_inputs(g=g, inputs=inputs)

    g_clone = copy_mtg(g)
    g = extract_mtg(g, plant_id=f'{id_plant}1')

    print("Computing 'static' data...")

    g, g_clone = initialisation_twins.init_model(g=g, g_clone=g_clone, inputs=inputs)
    static_data = {'form_factors': {s: g.property(s) for s in ('ff_sky', 'ff_leaves', 'ff_soil')}}
    static_data.update({'Na': g.property('Na')})
    with open(path_preprocessed_data / f'{id_plant}_static.json', mode='w') as f_prop:
        dump(static_data, f_prop, indent=2)

    scene_single = display.visu(g, def_elmnt_color_dict=True, scene=Scene(), view_result=False)

    mtg_save_geometry(scene=scene_single, file_path=path_preprocessed_data, index=f'_{id_plant}')
    save_mtg(g=g, scene=scene_single, file_path=path_preprocessed_data, filename=f'initial_mtg_{id_plant}.pckl')

    print("Computing 'dynamic' data...")
    dynamic_data = {}

    inputs_hourly = io.HydroShootHourlyInputs(psi_soil=inputs.psi_soil_forced, sun2scene=inputs.sun2scene)

    params = inputs.params
    for date in params.simulation.date_range:
        print("=" * 72)
        print(f'Date: {date}\n')

        inputs_hourly.update(g=g, date_sim=date, hourly_weather=inputs.weather[inputs.weather.index == date],
                             psi_pd=inputs.psi_pd, params=params)

        g, diffuse_to_total_irradiance_ratio = initialisation_twins.init_hourly(
            g=g, g_clone=g_clone, inputs_hourly=inputs_hourly, leaf_ppfd=inputs.leaf_ppfd, params=params)

        dynamic_data.update({g.date: {
            'diffuse_to_total_irradiance_ratio': diffuse_to_total_irradiance_ratio,
            'Ei': g.property('Ei'),
            'Eabs': g.property('Eabs')}})

    with open(path_preprocessed_data / f'{id_plant}_dynamic.json', mode='w') as f_prop:
        dump(dynamic_data, f_prop, indent=2)

    pass


def run_sims(args):
    print(args)
    return run_preprocess(*args)


def mp(sim_args: Iterable, nb_cpu: int = 2):
    with Pool(nb_cpu) as p:
        p.map(run_sims, sim_args)


if __name__ == '__main__':
    path_root = Path(__file__).parent.resolve()

    digit_df = read_csv(cfg.path_digit / 'all_pots.csv', sep=';', decimal='.', header=0)
    # names_plant = set([" ".join(re.findall("[a-zA-Z]+", s)) for s in digit_df['Plant'].unique()])
    names_plant = ['belledenise', 'plantdec', 'poulsard', 'raboso', 'salice']

    # run_preprocess(id_plant=list(names_plant)[0])
    time_on = datetime.now()
    mp(sim_args=product(names_plant, [digit_df]), nb_cpu=12)
    time_off = datetime.now()
    print(f"--- Total runtime: {(time_off - time_on).seconds} sec ---")
