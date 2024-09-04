from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Floor, Door, Key, Lava, Ball, Box
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import COLORS

class GamingEnv(MiniGridEnv):
    """
    ## Description

    This environment is a gaming room for testing the "gaming problem".
    There is no goal or reward in this env. A small penalty is
    subtracted for the number of steps to reach the goal. This environment works
    as a gaming environment for another real task env to provide merely endless novelty. 
    All the objects in the environment are randomly initialized.

    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Gaming-15x15-v0`

    """

    def __init__(
        self,
        size=15,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps=200,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.obj_types = [Box, Floor, Door, Key, Ball]
        self.colors = list(COLORS.keys())

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "explore and get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place random objects to increase obs novelty
        for i in range(2, width-2):
            for j in range(2, height-2):
                obj_ind = self._rand_int(0,5)
                obj_type = self.obj_types[obj_ind]
                color_ind = self._rand_int(0,6)
                if obj_ind > 0:
                    self.put_obj(obj_type(color=self.colors[color_ind]), i, j)
                else:
                    contain_ind = self._rand_int(0,5)
                    self.put_obj(obj_type(color=self.colors[color_ind], contains=self.obj_types[contain_ind](color=self.colors[color_ind])), i, j)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "explore and get to the green goal square"
