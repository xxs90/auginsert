from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, find_elements, xml_path_completion
import os

class PegHoleObject(MujocoXMLObject):
    def __init__(self, name, type="peg", top_shape="diamond", body_shape="cube", hinge_pos="0 0 0", hinge_axis="0 0 1"):
        assert type in ["peg", "hole"]

        super().__init__(
            os.path.join(os.path.dirname(__file__), f"assets/interactables/{top_shape}_{body_shape}_{type}.xml"),
            name=name,
            # joints=[dict(type="free")],
            # TODO: damping=10 once hinges are figured out
            joints=[dict(type="hinge", damping="1000", pos=hinge_pos, axis=hinge_axis)],
            obj_type="all",
            duplicate_collision_geoms=False
        )