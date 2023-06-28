from itertools import product
from pathlib import Path
from pickle import load

from hydroshoot.architecture import get_leaves
from hydroshoot.utilities import vapor_pressure_deficit
from matplotlib import pyplot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from openalea.mtg.mtg import MTG
from pandas import read_csv, DataFrame

from sims.config import Config

PLANT_IDS = ['belledenise', 'plantdec', 'poulsard', 'raboso', 'salice']


def plot_property(prop: str, g: MTG, ax: pyplot.Subplot, prop2: str) -> tuple:
    assert prop2 in ('Eabs', 'Ei', 'gs')
    x, y, c = zip(*[(g.node(vid).properties()[prop], g.node(vid).TopPosition[2], g.node(vid).properties()[prop2])
                    for vid in get_leaves(g)])
    im_ = ax.scatter(x, y, c=c, vmin=0, vmax=(1800 if prop2 in ("Eabs", "Ei") else 0.5), cmap='hot')
    #    ax.scatter(*zip(*[(g.node(vid).properties()[prop], g.node(vid).TopPosition[2])
    #                      for vid in get_leaves(g)]), label=id_plant)
    return ax, im_


def plot_leaf_temperature_profile(id_plant, g: MTG, ax: pyplot.Subplot) -> pyplot.Subplot:
    ax.scatter(*zip(*[(g.node(vid).Tlc, g.node(vid).TopPosition[2]) for vid in get_leaves(g)]), label=id_plant)
    return ax


def plot_temperature(weather: DataFrame, path_output_dir: Path):
    fig, axs = pyplot.subplots(nrows=4, ncols=6, sharex='all', sharey='all', figsize=(15, 9.6))

    for hour, ax in enumerate(axs.flatten()):
        w = weather[weather['time'] == f'2019-06-28 {hour:02d}:00']
        tair = w['Tac'].iloc[0]
        vpd_air = vapor_pressure_deficit(temp_air=tair, temp_leaf=tair, rh=w['hs'].iloc[0])
        for i, plant in enumerate(PLANT_IDS):
            pth = path_output_dir / plant / f'mtg20190628{hour:02d}0000.pckl'
            with open(pth, mode='rb') as f:
                g, _ = load(f)
            ax, im = plot_property(prop="Tlc", g=g, ax=ax, prop2='gs')
            # ax = plot_leaf_ppfd_profile(id_plant=f"plant {i + 1}", g=g, ax=ax)
        ax.text(0.75, 0.9, f'{hour:02d}:00', transform=ax.transAxes)
        ax.text(0.1, 0.1, f'VPDa={vpd_air:.1f}', transform=ax.transAxes)
        ax.vlines(tair, 0, 200, label='air')
        # ax.set(xlabel='Leaf temperature (°C)', ylabel='Leaf height (m)', xlim=(30, 50))
    axs[-1, 2].set(xlabel='Leaf temperature (°C)', xlim=(0, 60))
    axs[-1, 2].xaxis.set_label_coords(1.1, -0.2, transform=axs[-1, 2].transAxes)
    axs[1, 0].set(ylabel='Leaf height (cm)')
    axs[-1, 2].xaxis.set_label_coords(1.1, -0.2, transform=axs[-1, 2].transAxes)
    axs[1, 0].yaxis.set_label_coords(-0.325, 0, transform=axs[1, 0].transAxes)
    axs[-1, -1].legend(loc='lower right')
    cbar_ax = inset_axes(axs[-1, 0], width="30%", height="5%", loc=2)
    fig.colorbar(im, cax=cbar_ax, label='gs\n(mol/m2/s)', orientation='horizontal')
    scenario = path_output_dir.name
    fig.suptitle(scenario)
    fig.tight_layout()
    fig.savefig(path_output_dir.parent / f'temperature_profile_{scenario}.png')

    pass


if __name__ == '__main__':
    cfg = Config()
    weather_input = read_csv(cfg.path_weather, sep=";", decimal=".")

    # for is_cst_temperature, is_cst_wind in zip((False, True), (False, True)):
    for is_cst_temperature, is_cst_wind in product((True, False), (True, False)):
        name_index = f"{'tcst' if is_cst_temperature else 'tvar'}_{'ucst' if is_cst_wind else 'uvar'}"
        path_output = cfg.path_output_dir / name_index
        plot_temperature(weather=weather_input, path_output_dir=path_output)
