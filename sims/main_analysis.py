from pathlib import Path
from pickle import load

from hydroshoot.architecture import get_leaves
from matplotlib import pyplot
from openalea.mtg.mtg import MTG

from sims.config import Config


def plot_leaf_temperature_profile(id_plant, g: MTG, ax: pyplot.Subplot) -> pyplot.Subplot:
    ax.scatter(*zip(*[(g.node(vid).Tlc, g.node(vid).TopPosition[2]) for vid in get_leaves(g)]), label=id_plant)
    return ax


if __name__ == '__main__':
    cfg = Config()
    hour = 13
    fig, ax = pyplot.subplots()
    for i, plant in enumerate(['belledenise', 'plantdec', 'poulsard', 'raboso', 'salice']):
        with open(cfg.path_output_dir / plant / f'mtg20190628{hour}0000.pckl', mode='rb') as f:
            g, _ = load(f)
        ax = plot_leaf_temperature_profile(id_plant=f"plant {i + 1}", g=g, ax=ax)
    ax.set(xlabel='Leaf temperature (°C)', ylabel='Leaf height (m)')
    ax.legend()
    fig.savefig(Path(__file__).parent / 'temperature_profile.png')
