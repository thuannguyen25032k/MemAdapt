from embodiedbench.memory.spatial_memory import SpatialMemory


def test_scenegraph_basic():
    sm = SpatialMemory()
    cup = sm.add_or_update_object(name="cup", node_type="object")
    bottle = sm.add_or_update_object(name="bottle", node_type="object")
    assert sm.find_node("cup") is not None
    assert sm.find_node("bottle") is not None
    assert len(sm.nodes) == 2
