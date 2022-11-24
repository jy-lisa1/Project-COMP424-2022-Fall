# Student agent: Add your own agent here
import copy
from agents.agent import Agent
from store import register_agent
import sys

# python simulator.py --player_1 random_agent --player_2 student_agent --autoplay --autoplay_runs 1

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.autoplay = True
        self.name = "StudentAgent"
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
        row, column = end_pos
        barrier_dir = self.dir_map[barrier_dir]

        # check if there is already a barrier
        if chess_board[row, column, barrier_dir]:
            return False

        # check if you are already at the end goal
        if start_pos[0] == end_pos[0] and start_pos[1] == end_pos[1]:
            return True

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            row, column = cur_pos
            if cur_step == max_step:
                break

            # try stepping in each direction, don't if there's a barrier in the way
            for dir, move in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                if chess_board[row, column, dir]:
                    continue

                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                next_equals_adv = next_pos[0] == adv_pos[0] and next_pos[1] == adv_pos[1]
                next_equals_end = next_pos[0] == end_pos[0] and next_pos[1] == end_pos[1]
                
                # loop check
                if next_equals_adv or tuple(next_pos) in visited:
                    continue

                # check if end goal reached
                if next_equals_end:
                    is_reached = True
                    break

                # add next position to visited and to queue
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

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
        visited = {tuple(start_pos)} # stores possible positions

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
                if next_equals_adv or tuple(next_pos) in visited:
                    continue

                # add next position to visited and to queue
                visited.add(tuple(next_pos))
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

        direction = "u"
        n = len(chess_board[1])
        adv_x = adv_pos[0]
        adv_y = adv_pos[1]
        my_x = my_pos[0]
        my_y = my_pos[1]

        validSteps = self.getValidSteps(chess_board, my_pos, adv_pos, max_step)

        # get distance from us to adv at each possible next move
        distance_from_adv = [[10000 for x in range(n)] for y in range(n)] 
        min_distance = 10000
        best_coordinates = [my_x, my_y]

        for (row,column) in validSteps: # check valid position
            for dir in ["u","r","d","l"]:
                        if not(chess_board[row, column, self.dir_map[dir]]): # check valid barrier placement

                            # see if we're about to box ourselves in
                            num_walls_around_us = 0
                            for direction in [0,1,2,3]:
                                if chess_board[row,column,direction]:
                                    num_walls_around_us += 1
                            if num_walls_around_us >=2:
                                continue

                            # try each move to see if we can win the game in one step
                            validDirections = []
                            for dir in ["u","r","d","l"]:
                                if not(chess_board[row,column, self.dir_map[dir]]): # check valid barrier placement
                                    validDirections.append(dir)

                            # check each direction to see if would win
                            for dir in validDirections:
                                newBoard = copy.deepcopy(chess_board)
                                newBoard[row,column,self.dir_map[dir]] = True
                                gameDone, myScore, advScore = self.check_endgame(newBoard, (row,column), adv_pos)
                                if gameDone:
                                    result = myScore - advScore
                                    if result > 0:
                                        return (row,column), self.dir_map[dir]

                            # else run at opponent
                            x_diff = abs(adv_x - row)
                            y_diff = abs(adv_y - column)
                            distance = x_diff + y_diff
                            distance_from_adv[row][column] = distance
                            if x_diff + y_diff < min_distance:
                                min_distance = distance
                                best_coordinates[0] = row
                                best_coordinates[1] = column
        
        # take a step towards the opponent
        validDirections = []
        for dir in ["u","r","d","l"]:
            if not(chess_board[best_coordinates[0],best_coordinates[1], self.dir_map[dir]]): # check valid barrier placement
                validDirections.append(dir)
        my_pos = tuple(best_coordinates)

        #else:
        direction = validDirections[0] # change this later
            
        return my_pos, self.dir_map[direction]
