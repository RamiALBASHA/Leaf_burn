from pathlib import Path

from hydroshoot.display import visu, DEFAULT_COLORS
from openalea.plantgl.all import Viewer
from openalea.plantgl.scenegraph import Scene

from config import Pot
from leaf_burn import funcs

if __name__ == '__main__':
    path_root = Path(__file__).parent
    path_digit = path_root / 'source/digit'

    element_colors = DEFAULT_COLORS
    element_colors.update({'other': (0, 0, 0)})
    for f in ('all_pots_demo', 'all_pots'):
        g, _ = funcs.build_mtg(path_file=path_digit / f'{f}.csv', is_show_scene=True)
        g = funcs.add_pots(g=g, pot=Pot())
        scene = visu(g, elmnt_color_dict=element_colors, scene=Scene(), view_result=True)
        Viewer.saveSnapshot(str(path_digit / f'{f}.png'))
