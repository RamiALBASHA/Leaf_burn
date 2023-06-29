from datetime import date
from itertools import product
from pathlib import Path
from pickle import load

from hydroshoot.architecture import get_leaves
from hydroshoot.display import DEFAULT_LABELS
from hydroshoot.utilities import vapor_pressure_deficit
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from openalea.mtg.mtg import MTG
from pandas import read_csv, DataFrame, to_datetime

from sims.config import Config

PLANT_IDS = ['belledenise', 'plantdec', 'poulsard', 'raboso', 'salice']
DEFAULT_LABELS.update({'vpd': 'VPD [kPa]'})


def plot_property_map(prop: str, g: MTG, ax: pyplot.Subplot, prop2: str) -> tuple:
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


def plot_canopy_absorbed_irradiance(weather: DataFrame):
    fig, ax = pyplot.subplots()
    ax.plot(*zip(*enumerate(weather['Rg'].values)), 'k-', label='incident')
    for is_cst_t, is_cst_w in product((True, False), (True, False)):
        path_output_dir = cfg.path_output_dir / f"{'tcst' if is_cst_t else 'tvar'}_{'ucst' if is_cst_w else 'uvar'}"

        for i, plant in enumerate(PLANT_IDS):
            res = read_csv(path_output_dir / plant / f'time_series.csv', sep=';', decimal='.')
            ax.plot(*zip(*enumerate(res['Rg'])), marker='None', color='grey', alpha=0.5, label='absorbed')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:2], labels=labels[:2])
    ax.set(xlabel='local hour', ylabel=r'$\mathregular{(W\/m^{-2}_{ground}})}$')
    fig.tight_layout()
    fig.savefig(cfg.path_output_dir / 'irradiance.png')

    pass


def plot_property(weather: DataFrame, path_output_dir: Path, prop: str, prop2: str = None):
    fig, axs = pyplot.subplots(nrows=4, ncols=6, sharex='all', sharey='all', figsize=(15, 9.6))

    for hour, ax in enumerate(axs.flatten()):
        w = weather[weather.index == f'2019-06-28 {hour:02d}:00']
        tair = w['Tac'].iloc[0]
        vpd_air = vapor_pressure_deficit(temp_air=tair, temp_leaf=tair, rh=w['hs'].iloc[0])
        for i, plant in enumerate(PLANT_IDS):
            pth = path_output_dir / plant / f'mtg20190628{hour:02d}0000.pckl'
            with open(pth, mode='rb') as f:
                g, _ = load(f)
            ax, im = plot_property_map(prop=prop, g=g, ax=ax, prop2='gs')
            # ax = plot_leaf_ppfd_profile(id_plant=f"plant {i + 1}", g=g, ax=ax)
        ax.text(0.75, 0.9, f'{hour:02d}:00', transform=ax.transAxes)
        ax.text(0.1, 0.1, f'VPDa={vpd_air:.1f}', transform=ax.transAxes)
        if prop == 'Tlc':
            ax.vlines(tair, 0, 200, label='air')
            ax.xaxis.grid(True, which='minor')
            ax.xaxis.set_minor_locator(MultipleLocator(5))
        # ax.set(xlabel='Leaf temperature (°C)', ylabel='Leaf height (m)', xlim=(30, 50))
    axs[-1, 2].set(xlabel=DEFAULT_LABELS[prop])
    axs[-1, 2].xaxis.set_label_coords(1.1, -0.2, transform=axs[-1, 2].transAxes)
    axs[1, 0].set(ylabel='Leaf height [cm]')
    axs[-1, 2].xaxis.set_label_coords(1.1, -0.2, transform=axs[-1, 2].transAxes)
    axs[1, 0].yaxis.set_label_coords(-0.325, 0, transform=axs[1, 0].transAxes)
    axs[-1, -1].legend(loc='lower right')
    if prop2 is not None:
        cbar_ax = inset_axes(axs[-1, 0], width="30%", height="5%", loc=2)
        fig.colorbar(im, cax=cbar_ax, label=DEFAULT_LABELS[prop2], orientation='horizontal')

    scenario = path_output_dir.name
    fig.suptitle(scenario)
    fig.tight_layout()
    fig.savefig(path_output_dir.parent / f'{prop}_{prop2}_{scenario}.png')

    pass


if __name__ == '__main__':
    cfg = Config()
    weather_input = read_csv(cfg.path_weather, sep=";", decimal=".", index_col='time')
    weather_input.index = to_datetime(weather_input.index)
    weather_input = weather_input[weather_input.index.date == date(2019, 6, 28)]

    plot_canopy_absorbed_irradiance(weather=weather_input)

    for is_cst_temperature, is_cst_wind in product((True, False), (True, False)):
        name_index = f"{'tcst' if is_cst_temperature else 'tvar'}_{'ucst' if is_cst_wind else 'uvar'}"
        path_output = cfg.path_output_dir / name_index
        plot_property(weather=weather_input, path_output_dir=path_output, prop='Tlc', prop2='Eabs')
        plot_property(weather=weather_input, path_output_dir=path_output, prop='An', prop2='Eabs')
