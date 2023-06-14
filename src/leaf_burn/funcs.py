from pathlib import Path

from hydroshoot import architecture, display
from openalea.mtg import mtg, traversal
from openalea.plantgl.all import Vector3, Translated
from openalea.plantgl.scenegraph import Scene

from sims.config import Pot


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
