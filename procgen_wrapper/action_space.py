from enum import Enum


class ProcgenAction(Enum):
    pass


class MazesAction(ProcgenAction):
    Up = 5
    Left = 1
    Down = 3
    Right = 7


class LeaperAction(ProcgenAction):
    Up = 5
    Left = 1
    Down = 3
    Right = 7
    Wait = 4
