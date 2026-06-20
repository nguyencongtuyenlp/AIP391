from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    ZOOM_IN = 4
    ZOOM_OUT = 5
    STOP = 6


ACTION_NAMES = {
    Action.LEFT: "left",
    Action.RIGHT: "right",
    Action.UP: "up",
    Action.DOWN: "down",
    Action.ZOOM_IN: "zoom_in",
    Action.ZOOM_OUT: "zoom_out",
    Action.STOP: "stop",
}

NUM_ACTIONS = len(Action)
