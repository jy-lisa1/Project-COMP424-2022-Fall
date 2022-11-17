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
        x_diff = adv_x - my_x
        y_diff = adv_y - my_y

        # get distance from us to adv at each possible next move
        distance_from_adv = [[10000 for x in range(n)] for y in range(n)] 
        min_distance = 10000
        best_coordinates = [my_x, my_y]
        for r in range(0,n):
            for c in range(0,n):
                for dir in ["u","r","d","l"]:
                    if self.check_valid_step(chess_board,my_pos,(r,c),adv_pos,dir,max_step):
                        x_diff = abs(adv_x - r)
                        y_diff = abs(adv_y - c)
                        distance = x_diff + y_diff
                        distance_from_adv[r][c] = distance
                        if x_diff + y_diff < min_distance:
                            min_distance = distance
                            best_coordinates[0] = r
                            best_coordinates[1] = c
        
        for dir in ["u","r","d","l"]:
            if self.check_valid_step(chess_board,my_pos,tuple(best_coordinates),adv_pos,dir,max_step):
                my_pos = tuple(best_coordinates)
                direction = dir
                break

        # dummy return
        return my_pos, self.dir_map[direction]

