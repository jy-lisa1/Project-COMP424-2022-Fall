import copy
from agents.agent import Agent
from store import register_agent
from collections import defaultdict
import math
import random
import time

# python simulator.py --player_1 random_agent --player_2 mcts_agent --autoplay --autoplay_runs 1

@register_agent("mcts_agent")
class MCTS_Agent(Agent):

    def __init__(self):
        super(MCTS_Agent, self).__init__()
        self.autoplay = True
        self.name = "mcts_Agent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    

    class MonteCarloTreeSearchNode():
        def __init__(self, state, parent=None, parent_action=None):
            self.state: MCTS_Agent.State = state
            self.parent = parent
            self.parent_action = parent_action
            self.children = []
            self._number_of_visits = 0
            self._results = defaultdict(int)
            self._results[1] = 0
            self._results[-1] = 0
            self._untried_actions = None
            self._untried_actions = self.untried_actions()
            return

        def untried_actions(self):
            self._untried_actions = self.state.get_legal_actions()
            return self._untried_actions

        def q(self):
            wins = self._results[1]
            loses = self._results[-1]
            return wins - loses

        def n(self):
            return self._number_of_visits

        def expand(self):
            action = self._untried_actions.pop()
            next_state = self.state.move(action)
            child_node = MCTS_Agent.MonteCarloTreeSearchNode(next_state, parent=self, parent_action=action)
            self.children.append(child_node)
            return child_node 
        
        def is_terminal_node(self):
            return self.state.is_game_over()

        def rollout(self):
            current_rollout_state = self.state
            while not current_rollout_state.is_game_over():
                possible_moves = current_rollout_state.get_legal_actions()
                if len(possible_moves) == 0:
                    return -1
                action = self.rollout_policy(possible_moves)
                current_rollout_state = current_rollout_state.move(action)
            return current_rollout_state.game_result()

        def backpropagate(self, result):
            self._number_of_visits += 1.
            self._results[result] += 1.
            if self.parent:
                self.parent.backpropagate(result)

        def is_fully_expanded(self):
            return len(self._untried_actions) == 0

        def best_child(self, c_param=0.1):
            choices_weights = [(c.q() / c.n()) + c_param * math.sqrt((2 * math.log(self.n()) / c.n())) for c in self.children]
            return self.children[choices_weights.index(max(choices_weights))]

        def rollout_policy(self, possible_moves):
            return possible_moves[random.randint(0,len(possible_moves)-1)]

        def _tree_policy(self):
            current_node = self
            while not current_node.is_terminal_node():

                if len(self.state.get_legal_actions()) == 0:
                    return current_node

                if not current_node.is_fully_expanded():
                    return current_node.expand()
                else:
                    current_node = current_node.best_child()
            return current_node

        def best_action(self):
            simulation_no = 30
            
            for i in range(simulation_no):
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
            
            return self.best_child(c_param=0.)

    class State:
        def __init__(self, chess_board, my_pos, adv_pos, max_step):
            self.chess_board = chess_board
            self.my_pos = my_pos
            self.adv_pos = adv_pos
            self.max_step = max_step
            self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        def check_valid_step(self, chess_board, start_pos, end_pos, adv_pos, barrier_dir, max_step):
            """
            Check if the step the agent takes is valid (reachable and within max steps).

            Parameters
            ----------
            start_pos : tuple
                The start position of the agent.
            end_pos : np.ndarray
                The end position of the agent.
            barrier_dir : int
                The direction of the barrier.
            """
            # Endpoint already has barrier or is boarder
            r, c = end_pos
            barrier_dir = self.dir_map[barrier_dir]

            if chess_board[r, c, barrier_dir]:
                return False
            if start_pos[0] == end_pos[0] and start_pos[1] == end_pos[1]:
                return True

            # BFS
            state_queue = [(start_pos, 0)]
            visited = {tuple(start_pos)}
            is_reached = False
            while state_queue and not is_reached:
                cur_pos, cur_step = state_queue.pop(0)
                r, c = cur_pos
                if cur_step == max_step:
                    break
                for dir, move in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                    if chess_board[r, c, dir]:
                        continue

                    next_pos = (cur_pos[0]+move[0], cur_pos[1]+move[1])

                    next_equals_adv = next_pos[0] == adv_pos[0] and next_pos[1] == adv_pos[1]
                    next_equals_end = next_pos[0] == end_pos[0] and next_pos[1] == end_pos[1]
                    if next_equals_adv or tuple(next_pos) in visited:
                        continue
                    if next_equals_end:
                        is_reached = True
                        break

                    visited.add(tuple(next_pos))
                    state_queue.append((next_pos, cur_step + 1))

            return is_reached

        def get_legal_actions(self):
            """
            Check if the step the agent takes is valid (reachable and within max steps).

            Parameters
            ----------
            start_pos : tuple
                The start position of the agent.
            end_pos : np.ndarray
                The end position of the agent.
            barrier_dir : int
                The direction of the barrier.
            """
            chess_board = self.chess_board
            start_pos = self.my_pos
            adv_pos = self.adv_pos
            max_step = self.max_step
            # BFS
            state_queue = [(start_pos, 0)]
            visited = set()
            visitedCoords = {tuple(start_pos)}

            # initialize visited with all possible directions
            for direction in [0, 1, 2, 3]:
                if not chess_board[start_pos[0],start_pos[1], direction]:
                    visited.add((start_pos[0], start_pos[1], direction)) # stores possible positions

            while state_queue:
                cur_pos, cur_step = state_queue.pop(0)
                row, column = cur_pos
                if cur_step == max_step:
                    continue

                # try stepping in each direction, don't if there's a barrier in the way
                for dir, move in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                    if chess_board[row, column, dir]:
                        continue

                    next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])

                    next_equals_adv = next_pos[0] == adv_pos[0] and next_pos[1] == adv_pos[1]
                    
                    # loop check and check if its the opponents position
                    if next_equals_adv or tuple(next_pos) in visitedCoords:
                        continue

                    # add next position to visited and to queue
                    visitedCoords.add(tuple(next_pos))
                    for direction in [0, 1, 2, 3]:
                        if not chess_board[next_pos[0],next_pos[1], direction]:
                            visited.add((next_pos[0],next_pos[1],direction))
                            state_queue.append((next_pos, cur_step + 1))
            return list(visited)

        def endgame_details(self):
            # setup variables
            chess_board = self.chess_board
            my_pos = self.my_pos
            adv_pos = self.adv_pos
            boardSize = chess_board.shape[0]
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

            # Union-Find
            father = dict()
            for r in range(boardSize):
                for c in range(boardSize):
                    father[(r, c)] = (r, c)

            def find(pos):
                if father[pos] != pos:
                    father[pos] = find(father[pos])
                return father[pos]

            def union(pos1, pos2):
                father[pos1] = pos2

            for r in range(boardSize):
                for c in range(boardSize):
                    for dir, move in enumerate(
                        moves[1:3]
                    ):  # Only check down and right
                        if chess_board[r, c, dir + 1]:
                            continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)

            for r in range(boardSize):
                for c in range(boardSize):
                    find((r, c))
            my_r = find(tuple(my_pos))
            adv_r = find(tuple(adv_pos))
            my_score = list(father.values()).count(my_r)
            adv_score = list(father.values()).count(adv_r)
            if my_r == adv_r:
                return False, my_score, adv_score
            return True, my_score, adv_score

        def is_game_over(self):
            '''
            Modify according to your game or 
            needs. It is the game over condition
            and depends on your game. Returns
            true or false
            '''
            return self.endgame_details()[0]

        def game_result(self):
            '''
            Modify according to your game or 
            needs. Returns 1 or 0 or -1 depending
            on your state corresponding to win,
            tie or a loss.
            '''
            my_score = self.endgame_details()[1]
            adv_score = self.endgame_details()[2]
            if my_score == adv_score:
                return 0
            elif my_score < adv_score:
                return -1
            else:
                return 1

        def move(self,action):
            '''
            Modify according to your game or 
            needs. Changes the state of your 
            board with a new value. For a normal
            Tic Tac Toe game, it can be a 3 by 3
            array with all the elements of array
            being 0 initially. 0 means the board 
            position is empty. If you place x in
            row 2 column 3, then it would be some 
            thing like board[2][3] = 1, where 1
            represents that x is placed. Returns 
            the new state after making a move.
            '''
            new_board = copy.deepcopy(self.chess_board)
            x = action[0]
            y = action[1]
            dir = action[2]
            new_board[x,y,dir] = True
            new_state = MCTS_Agent.State(new_board, (x,y), self.adv_pos, self.max_step)
            return new_state


    def step(self, chess_board, my_pos, adv_pos, max_step):
        
        initial_state = MCTS_Agent.State(chess_board, my_pos, adv_pos, max_step)
        root = MCTS_Agent.MonteCarloTreeSearchNode(state = initial_state)
        selected_node: MCTS_Agent.MonteCarloTreeSearchNode = root.best_action()
        x, y, dir = selected_node.parent_action
        return ((x,y),dir)