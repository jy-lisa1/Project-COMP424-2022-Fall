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
        n = len(chess_board[0])
        valid_steps = []

        for r in range(n):
            for c in range(n):
                for dir in ["r","l","u","d"]:
                    if self.check_valid_step(chess_board,start_pos,(r,c),adv_pos,dir,max_step):
                        valid_steps.append(tuple([r,c,self.dir_map[dir]]))
        return valid_steps

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

    def evaluate(self, chess_board, my_pos, adv_pos):
        """ lower value is better here! """
        
        heuristic = 10000
        adv_x = adv_pos[0]
        adv_y = adv_pos[1]
        my_x = my_pos[0]
        my_y = my_pos[1]

        # check if we're endgame
        if self.check_endgame(chess_board, my_pos, adv_pos)[0]:
            return -10000

        # see if we're about to box ourselves in
        num_walls_around_us = 0
        for direction in [0,1,2,3]:
            if chess_board[my_x,my_y,direction]:
                num_walls_around_us += 1
            if num_walls_around_us >=2:   # return 10000 right away since this is very bad
                return 10000       
        
        # manhattan distance to opponent
        x_diff = abs(adv_x - my_x)
        y_diff = abs(adv_y - my_y)
        distance = x_diff + y_diff
        heuristic = distance

        return heuristic

    def minimax(self, chessboard, isMax, my_pos, adv_pos, depth, max_depth, max_step, alpha, beta):
        
        # edit the evaluation since our heuristic function returns smaller valeus for better results
        score = 10000-self.evaluate(chessboard, my_pos, adv_pos)  
        endgame_details = self.check_endgame(chessboard, my_pos, adv_pos)
        
        if depth == max_depth:  # at leaf node but not endgame
            return score

        if endgame_details[0]: # reached endgame
            my_score = endgame_details[1]
            adv_score = endgame_details[2]
            if my_score > adv_score and isMax:
                return 10000           
            elif my_score > adv_score and not isMax:
                return -10000            
            elif my_score < adv_score and isMax:
                return -10000            
            elif my_score < adv_score and not isMax:
                return 10000           
            else: # tie
                return 0                
    
        # If this maximizer's move
        if (isMax) :    
            best = -10000
            
            # Traverse all cells
            valid_steps = self.getValidSteps(chessboard, my_pos, adv_pos, max_step)
            for step in valid_steps:
                x = step[0]
                y = step[1]
                dir = step[2]
                chessboard[x,y,dir] = True  # make the move
                
                # Call minimax recursively and choose the max value
                best = max(best, self.minimax(chessboard, not isMax, (x,y), adv_pos, depth+1,
                           max_depth, max_step, alpha, beta))
                
                chessboard[x,y,dir] = False # Undo the move
                
                alpha = max(alpha, best)    # alpha-beta pruning
                if beta <= alpha:
                    break
            return best

        # If this minimizer's move
        else :
            best = 10000
            
            # Traverse all cells
            valid_steps = self.getValidSteps(chessboard, my_pos, adv_pos, max_step)
            for step in valid_steps:
                x = step[0]
                y = step[1]
                dir = step[2]
                chessboard[x,y,dir] = True  # make the move
                
                # Call minimax recursively and choose the max value
                best = min(best, self.minimax(chessboard, not isMax, (x,y), adv_pos, depth+1,
                                              max_depth, max_step, alpha, beta))
                
                chessboard[x,y,dir] = False # Undo the move
                beta = min(beta, best)      # alpha-beta pruning
                if beta <= alpha:
                    break
            return best

    def step(self, chess_board, my_pos, adv_pos, max_step):
        bestVal = -100000000
        bestMove = (my_pos,0)
        max_depth = 2

        # Traverse all cells, evaluate minimax function for all empty cells
        # return the cell with optimal value.
        valid_steps = self.getValidSteps(chess_board, my_pos, adv_pos, max_step)
        print(len(valid_steps))
        for step in valid_steps:
            x = step[0]
            y = step[1]
            dir = step[2]

            # make the move
            chess_board[x,y,dir] = True

            # compute evaluation function for this move.
            moveVal = self.minimax(chess_board, True, my_pos, adv_pos, 0, max_depth, max_step, -50000, 50000)
            
            # undo the move
            chess_board[x,y,dir] = False
            
            # If the value of the current move is more than the best value, then update best
            if (moveVal > bestVal):               
                bestMove = ((x,y),dir)
                bestVal = moveVal
    
        return bestMove

    def step1(self, chess_board, my_pos, adv_pos, max_step):
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

        # get distance from us to adv at each possible next move
        distance_from_adv = [[10000 for x in range(n)] for y in range(n)] 
        min_distance = 10000
        best_coordinates = [my_pos[0], my_pos[1]]
        for r in range(0,n):
            for c in range(0,n):
                for dir in ["u","r","d","l"]:
                    if self.check_valid_step(chess_board,my_pos,(r,c),adv_pos,dir,max_step):
                        distance = self.evaluate(chess_board,(r,c),adv_pos)
                        distance_from_adv[r][c] = distance
                        if distance < min_distance:
                            min_distance = distance
                            best_coordinates[0] = r
                            best_coordinates[1] = c
        
        available_directions = []
        for dir in ["u","r","d","l"]:
            if self.check_valid_step(chess_board,my_pos,tuple(best_coordinates),adv_pos,dir,max_step):
                available_directions.append(dir)
                my_pos = tuple(best_coordinates)
                break

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

