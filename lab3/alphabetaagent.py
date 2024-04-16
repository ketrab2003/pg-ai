from __future__ import annotations
from typing import TYPE_CHECKING

from exceptions import AgentException
if TYPE_CHECKING:
    from connect4 import Connect4


class AlphaBetaAgent:
    def __init__(self, my_token='o', depth=3, enable_heuristics=True):
        self.my_token = my_token
        self.depth = depth
        self.enable_heuristics = enable_heuristics

    def decide(self, connect4: Connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        
        best_value = -2
        best_column = None
        for n_column in connect4.possible_drops():
            connect4.drop_token(n_column)
            value = self.alphabeta(connect4, False, self.depth)
            connect4.undrop_token(n_column)
            if value > best_value:
                best_value = value
                best_column = n_column

        # print(f'best_value: {best_value} best_column: {best_column}')
        return best_column

    def alphabeta(self, connect4: Connect4, maximizing: bool, depth: int, alpha=-2, beta=2) -> int:
        if connect4.game_over:
            if connect4.wins == self.my_token:
                return 1
            elif connect4.wins is not None:
                return -1
            else:
                return 0
        if depth <= 0:
            if self.enable_heuristics:
                return connect4.evaluate(self.my_token)
            else:
                return 0

        if maximizing:
            value = -1
            for n_column in connect4.possible_drops():
                connect4.drop_token(n_column)
                value = max(value, self.alphabeta(connect4, False, depth - 1, alpha, beta))
                connect4.undrop_token(n_column)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = 1
            for n_column in connect4.possible_drops():
                connect4.drop_token(n_column)
                value = min(value, self.alphabeta(connect4, True, depth - 1, alpha, beta))
                connect4.undrop_token(n_column)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value