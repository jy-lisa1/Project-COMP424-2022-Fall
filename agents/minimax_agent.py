# Student agent: Add your own agent here
import copy
from agents.agent import Agent
from store import register_agent
import sys

# python simulator.py --player_1 random_agent --player_2 student_agent --autoplay --autoplay_runs 1

@register_agent("minimax_agent")
class MinimaxAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    def __init__(self):
        super(MinimaxAgent, self).__init__()
        self.autoplay = True
        self.name = "MinimaxAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    class State:
        def __init__(self, chess_board, my_pos, adv_pos):
            self.isVisited = False
            self.isTerminal = False
            self.utility = 10000
            self.chess_board = chess_board
            self.my_pos = my_pos
            self.adv_pos = adv_pos
            self.children = []

    def getValidSteps(self, chess_board, start_pos, adv_pos, max_step):
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
        return visited

    def check_endgame(self, chess_board, my_pos, adv_pos):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # setup variables
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

    def checkStateEndgame(self, state):
        gameDone, myScore, advScore = self.check_endgame(state.chess_board, state.my_pos, state.adv_pos)
        state.isTerminal = gameDone
        state.utility = advScore - myScore
        return state

    def evaluate(self, chess_board, my_pos, adv_pos):
        
        # distance to opponent
        heuristic = 10000
        adv_x = adv_pos[0]
        adv_y = adv_pos[1]
        my_x = my_pos[0]
        my_y = my_pos[1]
        x_diff = abs(adv_x - my_x)
        y_diff = abs(adv_y - my_y)
        distance = x_diff + y_diff
        heuristic = distance
        return heuristic

    def miniMaxDecision(self, chess_board, my_pos, adv_pos, max_step):
        """
        Uses minimax value function to decide the best move
        """
        validSteps = self.getValidSteps(chess_board, my_pos, adv_pos, max_step)
        value = {}
        # get the utility of each possible move
        for row, column, dir in validSteps:
            newBoard = copy.deepcopy(chess_board)
            newBoard[row,column,dir] = True
            newState = self.State(newBoard, adv_pos, (row,column))
            value[(row,column,dir)] = self.miniMaxValue(newState, max_step, False, 2)
        bestMove = min(value, key=value.get)
        # print("I choose: ", bestMove, value[bestMove])
        return bestMove

    def miniMaxValue(self, state, max_step, myTurn, n):
        """
        recursive, returns the utility of a given state, with the lowest being the best
        """
        value = {}

        gameDone, myScore, advScore = self.check_endgame(state.chess_board, state.my_pos, state.adv_pos)
        state.isTerminal = gameDone
        state.utility = myScore - advScore
        if myTurn: utility = self.evaluate(state.chess_board, state.my_pos, state.adv_pos)
        else: utility = self.evaluate(state.chess_board, state.adv_pos, state.my_pos)

        if gameDone: 
            if myTurn: 
                score = advScore - myScore
                # weight losses as bad
                if score >= 0: score = score + 10000
            else: 
                score = myScore - advScore
            return score
        if n == 0: 
            return utility
        if utility > 5:
            return utility
        
        validSteps = self.getValidSteps(state.chess_board, state.my_pos, state.adv_pos, max_step)
        children = set()
        for row,column,dir in validSteps:
            newBoard = copy.deepcopy(state.chess_board)
            newBoard[row,column,dir] = True
            # add a child state where the other player makes the decision so swap positions
            children.add(self.State(newBoard, state.adv_pos, (row,column)))
        for child in children:
            # get the value of the child state, alternating between myTurn and not MyTurn
            value[child] = self.miniMaxValue(child, max_step, not(myTurn), n-1)
        
        # troubleshooting
        if (value == {}):
            if myTurn: 
                score = advScore - myScore
                # weight losses as bad
                if score >= 0: score = score + 10000 
            else: 
                score = myScore - advScore
            return score

        if myTurn: 
            score = min(value.values())
            bestMove = min(value, key=value.get)
        else: 
            score = max(value.values())
            bestMove = max(value, key=value.get)
        #if (n==1): print(bestMove.my_pos, score)
        return score

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        bestCoordinates = self.miniMaxDecision(chess_board,my_pos,adv_pos,max_step)
            
        bestStep = (bestCoordinates[0],bestCoordinates[1])
        bestDir = bestCoordinates[2]
        return (bestStep,bestDir)
