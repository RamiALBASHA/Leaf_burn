from openalea.mtg.mtg import MTG


def trim_mtg(g: MTG, vtx_id: int):
    """This function trims branches at ALL scales greater than zero."""
    if g.node(vtx_id).scale() != 0:
        if g.node(vtx_id).nb_components() == 0:
            complex_id = g.node(vtx_id).complex()._vid
            g.node(vtx_id).remove_tree()
            trim_mtg(g, complex_id)
        else:
            trim_mtg(g, g.node(vtx_id).components()[0]._vid)
    pass


def copy_mtg(g: MTG) -> MTG:
    geom = {vid: g.node(vid).geometry for vid in g.property('geometry')}
    g.remove_property('geometry')
    g_copy = g.copy()
    g.add_property('geometry')
    g.property('geometry').update(geom)
    g_copy.add_property('geometry')
    g_copy.property('geometry').update(geom)
    return g_copy


def extract_mtg(g: MTG, plant_id: int) -> MTG:
    g_single = copy_mtg(g=g)

    branch_vid_to_remove = [vid for vid in g.VtxList(Scale=1) if g.node(vid).label != f'plant{plant_id}']
    for vid in branch_vid_to_remove:
        trim_mtg(g=g_single, vtx_id=vid)

    return g_single
