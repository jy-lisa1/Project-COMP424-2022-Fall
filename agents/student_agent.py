# Student agent: Add your own agent here
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
                if cur_step + 1 == max_step: continue
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return visited

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
        for row in range(0,n):
            for column in range(0,n):
                if (row,column) in validSteps: # check valid position
                    for dir in ["u","r","d","l"]:
                        if not(chess_board[row,column, self.dir_map[dir]]): # check valid barrier placement

                            # see if we're about to box ourselves in
                            num_walls_around_us = 0
                            for direction in [0,1,2,3]:
                                if chess_board[row,column,direction]:
                                    num_walls_around_us += 1
                            if num_walls_around_us >=2:
                                continue

                            # set this move as our best one
                            x_diff = abs(adv_x - row)
                            y_diff = abs(adv_y - column)
                            distance = x_diff + y_diff
                            distance_from_adv[row][column] = distance
                            if x_diff + y_diff < min_distance:
                                min_distance = distance
                                best_coordinates[0] = row
                                best_coordinates[1] = column
        
        available_directions = []
        for dir in ["u","r","d","l"]:
            if not(chess_board[best_coordinates[0],best_coordinates[1], self.dir_map[dir]]): # check valid barrier placement
                available_directions.append(dir)
        my_pos = tuple(best_coordinates)

        # try to determine the best direction to place our barrier

        if adv_x < best_coordinates[0] and "l" in available_directions:
            direction = "l"

        elif adv_x > best_coordinates[0] and "r" in available_directions:
            direction = "r"

        elif adv_y < best_coordinates[1] and "d" in available_directions:
            direction = "d"

        elif adv_y > best_coordinates[1] and "u" in available_directions:
            direction = "u"

        else:
            direction = available_directions[0] # change this later
            
        return my_pos, self.dir_map[direction]
