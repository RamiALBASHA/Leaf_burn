from datetime import date
from itertools import product
from math import pi
from pathlib import Path
from pickle import load

import openalea.plantgl.all as pgl
from hydroshoot.architecture import get_leaves, load_mtg, slim_cylinder
from hydroshoot.display import DEFAULT_LABELS, visu, DEFAULT_COLORS
from hydroshoot.utilities import vapor_pressure_deficit
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from openalea.mtg.mtg import MTG
from pandas import read_csv, DataFrame, to_datetime

from leaf_burn import funcs
from sims.config import Config, Pot

PLANT_IDS = ['belledenise', 'plantdec', 'poulsard', 'raboso', 'salice']
DEFAULT_LABELS = {k: v.replace('[', '(').replace(']', ')') for k, v in DEFAULT_LABELS.items()}
DEFAULT_LABELS.update({'vpd': 'VPD (kPa)'})


def plot_property_map(prop: str, g: MTG, ax: pyplot.Subplot, prop2: str, label: str = None) -> tuple:
    assert prop2 in ('Eabs', 'Ei', 'gs')
    x, y, c = zip(*[(g.node(vid).properties()[prop], g.node(vid).TopPosition[2], g.node(vid).properties()[prop2])
                    for vid in get_leaves(g)])
    im_ = ax.scatter(x, y, c=c, vmin=0, vmax=(1800 if prop2 in ("Eabs", "Ei") else 0.5), cmap='hot', label=label)
    #    ax.scatter(*zip(*[(g.node(vid).properties()[prop], g.node(vid).TopPosition[2])
    #                      for vid in get_leaves(g)]), label=id_plant)
    return ax, im_


def plot_leaf_temperature_profile(id_plant, g: MTG, ax: pyplot.Subplot) -> pyplot.Subplot:
    ax.scatter(*zip(*[(g.node(vid).Tlc, g.node(vid).TopPosition[2]) for vid in get_leaves(g)]), label=id_plant)
    return ax


def plot_canopy_absorbed_irradiance(weather: DataFrame):
    fig, ax = pyplot.subplots()
    ax.plot(*zip(*enumerate(weather['Rg'].values)), 'k-', label='incident')

    for expo_id, _ in cfg.expositions:
        path_output_dir = cfg.path_output_dir / expo_id / list(cfg.scenarios.keys())[0] / 'tvar_uvar'

        for i, plant in enumerate(PLANT_IDS):
            res = read_csv(path_output_dir / plant / f'time_series.csv', sep=';', decimal='.')
            ax.plot(*zip(*enumerate(res['Rg'])), marker='None', color='grey', alpha=0.5, label='absorbed')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:2], labels=labels[:2])
        ax.set(xlabel='local hour', ylabel=r'$\mathregular{(W\/m^{-2}_{ground}})}$')
        fig.tight_layout()
        fig.savefig(cfg.path_output_dir / expo_id / 'irradiance.png')

    pass


def plot_property(weather: DataFrame, path_output_dir: Path, prop: str, prop2: str = None):
    fig, axs = pyplot.subplots(nrows=4, ncols=6, sharex='all', sharey='all', figsize=(15, 9.6))

    for hour, ax in enumerate(axs.flatten()):
        w = weather[weather.index == f'2019-06-28 {hour:02d}:00']
        tair = w['Tac'].iloc[0]
        vpd_air = vapor_pressure_deficit(temp_air=tair, temp_leaf=tair, rh=w['hs'].iloc[0])

        height, air_temperature = [], []
        for i, plant in enumerate(PLANT_IDS):
            pth = path_output_dir / plant / f'mtg20190628{hour:02d}0000.pckl'
            with open(pth, mode='rb') as f:
                g, _ = load(f)
            z_leaf, t_air = zip(*[(g.node(i).TopPosition[2], t) for i, t in g.property('Tac').items()])
            height += z_leaf
            air_temperature += t_air
            ax, im = plot_property_map(prop=prop, g=g, ax=ax, prop2='gs')

        x_text = 0.9 if hour in (list(range(10)) + list(range(21, 24))) else 0.1
        align_text = 'right' if hour in (list(range(10)) + list(range(21, 24))) else 'left'

        ax.text(0.75, 0.9, f'{hour:02d}:00', transform=ax.transAxes)
        ax.text(x_text, 0.1, f'VPDa={vpd_air:.1f}', transform=ax.transAxes, horizontalalignment=align_text)
        ax.text(x_text, 0.2, f'Rg={w["Rg"].iloc[0]:.0f}', transform=ax.transAxes, horizontalalignment=align_text)
        if prop == 'Tlc':
            ax.xaxis.grid(True, which='minor')
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            height, air_temperature = zip(*sorted(zip(height, air_temperature)))
            ax.plot(air_temperature, height, 'r-', label='air')

    axs[-1, 2].set(xlabel=DEFAULT_LABELS[prop])
    axs[-1, 2].xaxis.set_label_coords(1.1, -0.2, transform=axs[-1, 2].transAxes)
    axs[1, 0].set(ylabel='Leaf height [cm]', ylim=[0, axs[1, 0].get_ylim()[-1]])
    axs[-1, 2].xaxis.set_label_coords(1.1, -0.2, transform=axs[-1, 2].transAxes)
    axs[1, 0].yaxis.set_label_coords(-0.325, 0, transform=axs[1, 0].transAxes)
    axs[-1, -1].legend(loc='center right')
    if prop2 is not None:
        cbar_ax = inset_axes(axs[-1, 0], width="30%", height="5%", loc=2)
        fig.colorbar(im, cax=cbar_ax, label=DEFAULT_LABELS[prop2], orientation='horizontal')

    scenario = path_output_dir.name
    fig.suptitle(scenario)
    fig.tight_layout()
    fig.savefig(path_output_dir.parent / f'{prop}_{prop2}_{scenario}.png')

    pass


def plot_reponse_to_temperature():
    fig, axs = pyplot.subplots(ncols=2, sharex='all')

    for expo_id, _ in cfg.expositions:
        for scen in cfg.scenarios.keys():
            for is_cst_t, is_cst_w in product((True, False), (True, False)):
                path_output_dir = cfg.path_output_dir / expo_id / scen / (
                    f"{'tcst' if is_cst_t else 'tvar'}_{'ucst' if is_cst_w else 'uvar'}")
                for hour in range(24):
                    for i, plant in enumerate(PLANT_IDS):
                        pth = path_output_dir / plant / f'mtg20190628{hour:02d}0000.pckl'
                        with open(pth, mode='rb') as f:
                            g, _ = load(f)
                        axs[0].scatter(*zip(*[(g.node(vid).Tlc, g.node(vid).An) for vid in get_leaves(g)]), marker='.',
                                       c='r')
                        axs[1].scatter(*zip(*[(g.node(vid).Tlc, g.node(vid).gs) for vid in get_leaves(g)]), marker='.',
                                       c='r')
            axs[0].set(xlabel='leaf temperature (°C)',
                       ylabel=' '.join(['Net carbon assimilation', r"$\mathregular{(\mu mol\/m^{-2}\/s^{-1})}$"]))
            axs[1].set(xlabel='leaf temperature (°C)',
                       ylabel=' '.join(['Stomatal conductance', r"$\mathregular{(mol\/m^{-2}\/s^{-1})}$"]))
    fig.tight_layout()
    fig.savefig(cfg.path_output_dir / 'an_vs_tleaf.png')

    pass


def plot_temperature_vs_light(weather: DataFrame):
    fig, axs = pyplot.subplots(ncols=4, nrows=6, sharex='all', sharey='all', figsize=(18 / 2.5, 20 / 2.54))

    ppfd_tot = {hour: [] for hour in range(24)}
    tleaf_tot = {hour: [] for hour in range(24)}
    for expo_id, _ in cfg.expositions:
        for scen in cfg.scenarios.keys():
            for is_cst_t, is_cst_w in ((False, False),):
                path_output_dir = cfg.path_output_dir / expo_id / scen / (
                    f"{'tcst' if is_cst_t else 'tvar'}_{'ucst' if is_cst_w else 'uvar'}")
                for ax, hour in zip(axs.flatten(), range(24)):
                    for i, plant in enumerate(PLANT_IDS):
                        pth = path_output_dir / plant / f'mtg20190628{hour:02d}0000.pckl'
                        with open(pth, mode='rb') as f:
                            g, _ = load(f)
                        ppfd, tleaf = zip(*[(g.node(vid).Eabs, g.node(vid).Tlc) for vid in get_leaves(g)])
                        ppfd_tot[hour] += ppfd
                        tleaf_tot[hour] += tleaf

    for ax, hour in zip(axs.flatten(), range(24)):
        w = weather[weather.index == f'2019-06-28 {hour:02d}:00']
        ax.hlines(w['Tac'].iloc[0], 0, 2000, color='b', linestyles='--', label='air')
        ax.scatter(ppfd_tot[hour], tleaf_tot[hour], marker='.', c='r', label='leaf')

    [ax.text(0.65, 0.1, f'{hour:02d}:00', transform=ax.transAxes) for (hour, ax) in zip(range(24), axs.flatten())]
    axs[3, 0].set(ylabel="Temperature (°C)")
    axs[3, 0].yaxis.set_label_coords(-.3, 1.0, transform=axs[3, 0].transAxes)

    axs[-1, 1].set(xlabel=' '.join(['Absorbed irradiance', r"$\mathregular{(\mu mol_{photon}\/m^{-2}\/s^{-1})}$"]))
    axs[-1, 1].xaxis.set_label_coords(1.05, -0.35, transform=axs[-1, 1].transAxes)

    # hls, lbs = axs[-1, -1].get_legend_handles_labels()
    axs[-1, -1].legend(fontsize=8)
    fig.tight_layout()
    for fmt in ('png', 'eps', 'svg', 'pdf'):
        fig.savefig(cfg.path_output_dir / f'tleaf_vs_ppfd.{fmt}', format=fmt, dpi=1200)

    df = DataFrame(dict(ppfd=ppfd_tot[16], tleaf=tleaf_tot[16]), index=range(len(ppfd_tot[16])))
    df.sort_values(by='ppfd', inplace=True)
    print(f"max tleaf of shaded leaves = {df[df['ppfd'] < 100]['tleaf'].max():.1f} °C at 16:00")
    print(f"max tleaf of sunlit leaves = {df['tleaf'].max():.1f} °C at 16:00")

    pass


def plot_temperature_profile(hour: int, path_output_dir: Path, weather: DataFrame, prop='Tlc', prop2='gs'):
    fig, ax = pyplot.subplots()

    w = weather[weather.index == f'2019-06-28 {hour:02d}:00']
    tair = w['Tac'].iloc[0]
    vpd_air = vapor_pressure_deficit(temp_air=tair, temp_leaf=tair, rh=w['hs'].iloc[0])

    height, air_temperature = [], []
    for i, plant in enumerate(PLANT_IDS):
        pth = path_output_dir / plant / f'mtg20190628{hour:02d}0000.pckl'
        with open(pth, mode='rb') as f:
            g, _ = load(f)
            z_leaf, t_air = zip(*[(g.node(i).TopPosition[2], t) for i, t in g.property('Tac').items()])
            height += z_leaf
            air_temperature += t_air
        ax, im = plot_property_map(prop=prop, g=g, ax=ax, prop2='gs', label=plant)

    ax.text(0.05, 0.3, f'{hour:02d}:00', transform=ax.transAxes)
    ax.text(0.05, 0.2, f'VPDa = {vpd_air:.1f} kPa', transform=ax.transAxes)
    ax.text(0.05, 0.1, ' '.join([f'Rg = {w["Rg"].iloc[0]:.0f}', r"$\mathregular{W\/m^{-2}}$"]), transform=ax.transAxes)
    if prop == 'Tlc':
        ax.xaxis.grid(True, which='minor')
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        height, air_temperature = zip(*sorted(zip(height, air_temperature)))
        ax.plot(air_temperature, height, 'r-', label='air')

    # ax.set(xlabel='Leaf temperature (°C)', ylabel='Leaf height (m)', xlim=(30, 50))
    ax.set(xlabel='Temperature (°C)' if prop == 'Tlc' else DEFAULT_LABELS[prop],
           ylabel='Leaf height (cm)',
           xlim=(35, 60), ylim=[0, ax.get_ylim()[-1]])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[-2:], labels=['leaf'] + [labels[-1]], loc='lower right')
    if prop2 is not None:
        cbar_ax = inset_axes(ax, width="30%", height="5%", loc=1)
        fig.colorbar(im, cax=cbar_ax, label=DEFAULT_LABELS[prop2], orientation='horizontal')

    scenario = path_output_dir.name
    fig.suptitle(scenario)
    fig.tight_layout()
    fig.savefig(path_output_dir.parent / f'{prop}_{prop2}_{scenario}_{hour}.png')

    pass


def plot_mockup(hour: int, path_output_dir: Path):
    for i, plant in enumerate(PLANT_IDS):
        pth = path_output_dir / plant / f'mtg20190628{hour:02d}0000.pckl'
        g, scene = load_mtg(path_mtg=str(pth),
                            path_geometry=cfg.path_preprocessed_data / f'{plant}/geometry_{plant}.bgeom')
        x_base, y_base = g.node(g.node(g.root).vid_base).BotPosition[:2]
        for v in get_leaves(g):
            n = g.node(v)
            g.node(v).TopPosition = [n.TopPosition[0] - x_base, n.TopPosition[1] - y_base, n.TopPosition[2]]
            g.node(v).BotPosition = [n.BotPosition[0] - x_base, n.BotPosition[1] - y_base, n.BotPosition[2]]
            mesh = n.geometry
            if n.label.startswith('L'):
                mesh = pgl.Translated(pgl.Vector3([-x_base, -y_base, 0.0]), mesh)
                g.node(v).geometry = mesh

        scene = visu(g, plot_prop='Tlc', min_value=35, max_value=60)
        pgl.Viewer.saveSnapshot(str(path_output_dir / f'{plant}_mkp.png'))
    pass


def plot_scene(hour: int, scenario_stomatal_behavior: str, path_output_dir: Path,
               is_cst_air_temperature: bool = False, is_cst_wind_speed: bool = False):
    path_digit = Path(r'C:\Users\albashar\Documents\dvp\Leaf_burn\sims\source\digit')
    g_all, _ = funcs.build_mtg(path_file=path_digit / 'all_pots_demo.csv', is_show_scene=True)
    g_all = funcs.add_pots(g=g_all, pot=Pot())

    element_colors = {k: (40, 40, 40) for k in list(DEFAULT_COLORS.keys()) + ['other']}
    scene = visu(g_all, elmnt_color_dict=element_colors, scene=pgl.Scene(), view_result=True)

    dx_positions = dict(
        belledenise=-5,
        plantdec=21,
        poulsard=-18,
        raboso=8,
        salice=34)

    pot = Pot()
    for i, (expo_id, expo_angle) in enumerate(cfg.expositions):
        for plant in PLANT_IDS:

            pth_mtg = cfg.path_output_dir / expo_id / scenario_stomatal_behavior / (
                f'{"tcst" if is_cst_air_temperature else "tvar"}_{"ucst" if is_cst_wind_speed else "uvar"}') / plant / (
                          f'mtg20190628{hour:02d}0000.pckl')
            pth_geom = cfg.path_preprocessed_data / expo_id / f'{plant}/geometry_{plant}.bgeom'
            for i_pos in range(9):

                g, _ = load_mtg(path_mtg=pth_mtg, path_geometry=pth_geom)

                x_base, y_base = g.node(g.node(g.root).vid_base).BotPosition[:2]
                vid_base = g.node(g.root).vid_base
                vid_pot = g.add_component(complex_id=vid_base, label='other', edge_type='/')
                mesh_pot = slim_cylinder(length=pot.height, radius_base=pot.radius_lower, radius_top=pot.radius_upper)
                mesh_pot = pgl.Translated(pgl.Vector3([x_base, y_base, 0]), mesh_pot)
                g.node(vid_pot).geometry = mesh_pot

                theta = 0 if i == 0 else pi
                dx = -5 if i == 0 else dx_positions[plant]
                dy = 0.1 if i == 0 else -20.1  # -20

                for v in get_leaves(g):
                    n = g.node(v)
                    # g.node(v).TopPosition = [n.TopPosition[0] - x_base, n.TopPosition[1] - y_base, n.TopPosition[2]]
                    # g.node(v).BotPosition = [n.BotPosition[0] - x_base, n.BotPosition[1] - y_base, n.BotPosition[2]]
                    mesh = n.geometry
                    if n.label.startswith('L'):
                        mesh = pgl.EulerRotated(theta, 0, 0, mesh)
                        mesh = pgl.Translated(pgl.Vector3([dx + (i_pos - 3) * 65, dy, 0.0]), mesh)
                        g.node(v).geometry = mesh

                scene = visu(g, plot_prop='Tlc', min_value=35, max_value=60, scene=scene)
                pyplot.close()

            if plant == PLANT_IDS[0]:
                scene = visu(g, plot_prop='Tlc', min_value=35, max_value=60, scene=scene)
        fig = pyplot.figure(1)
        cbar_ax = fig.get_axes()[0]
        cbar_ax.set_xlabel('Leaf temperature (°C)')
        cbar_ax.set_xticklabels([int(x) for x in cbar_ax.get_xticks()], ha='center', va='top')
        fig.set_size_inches(6 / 2.54, 2 / 2.54)
        fig.savefig(str(path_output_dir / 'cbar.png'))
        pgl.Viewer.saveSnapshot(str(path_output_dir / 'mkp.png'))

    pass


if __name__ == '__main__':
    cfg = Config()
    weather_input = read_csv(cfg.path_weather, sep=";", decimal=".", index_col='time')
    weather_input.index = to_datetime(weather_input.index)
    weather_input = weather_input[weather_input.index.date == date(2019, 6, 28)]

    plot_temperature_vs_light(weather=weather_input)
    plot_reponse_to_temperature()
    plot_canopy_absorbed_irradiance(weather=weather_input)

    for expo_id, _ in cfg.expositions:
        plot_temperature_profile(
            hour=16,
            path_output_dir=cfg.path_output_dir / expo_id / 'intermediate' / 'tvar_uvar',
            weather=weather_input,
            prop='Tlc',
            prop2='gs')

    plot_scene(
        hour=16,
        scenario_stomatal_behavior='intermediate',
        path_output_dir=cfg.path_output_dir.parent)

    for expo_id, _ in cfg.expositions:
        for sim_scen in cfg.scenarios.keys():
            for is_cst_temperature, is_cst_wind in product((True, False), (True, False)):
                name_index = f"{'tcst' if is_cst_temperature else 'tvar'}_{'ucst' if is_cst_wind else 'uvar'}"
                path_output = cfg.path_output_dir / expo_id / sim_scen / name_index
                # plot_property(weather=weather_input, path_output_dir=path_output, prop='vpd', prop2='gs')
                plot_property(weather=weather_input, path_output_dir=path_output, prop='Tlc', prop2='gs')
                # plot_property(weather=weather_input, path_output_dir=path_output, prop='An', prop2='Eabs')
