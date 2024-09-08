from robosuite.models.arenas import Arena
import os

class CapTheBottleArena(Arena):
    """Empty workspace."""

    def __init__(self, texture='light-wood'):
        super().__init__(os.path.join(os.path.dirname(__file__),f"assets/arena/ctb_arena_{texture}.xml"))
