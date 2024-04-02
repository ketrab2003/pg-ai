from __future__ import annotations
import random
from typing import TYPE_CHECKING

from exceptions import AgentException
if TYPE_CHECKING:
    from connect4 import Connect4


class RandomAgent:
    def __init__(self, my_token='o'):
        self.my_token = my_token

    def decide(self, connect4: Connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        return random.choice(connect4.possible_drops())
