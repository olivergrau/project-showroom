"""
Tic Tac Toe Player
"""
import copy
import math
import random
from operator import itemgetter

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]
    
    # return [[X, X, EMPTY],
    #         [EMPTY, O, O],
    #         [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    num_x = 0
    num_o = 0
    for i in range(3):        
        for j in range(3):
            if board[i][j] == EMPTY:
                continue
                
            if board[i][j] == X:
                num_x += 1
            else:
                num_o += 1
    
    # board is empty, X should begin per definition            
    if num_x == 0 and num_o == 0:
        return X
    
    if num_x > num_o:
        return O
    
    return X
    

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    available_actions = set()
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                available_actions.add((i, j))
                
    return available_actions           


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # is the action valid (out of boundary test)
    if action[0] < 0 or action[0] >= 3:
        raise ValueError("Invalid action")
    
    if action[1] < 0 or action[1] >= 3:
        raise ValueError("Invalid action")
    
    # if result gets an already taken move
    if board[action[0]][action[1]] != EMPTY:
        raise ValueError("Invalid action")
    
    # deep clone the board first, so you only return a detached copy of the board
    result_board = copy.deepcopy(board)
    
    # find out who is the player
    current_player = player(board)
    
    # set valid action to deep copy
    result_board[action[0]][action[1]] = current_player
    
    return result_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    
    # if board is in still in progress
    if not terminal(board):
        return None
    
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] is not None:
            return board[row][0]

    # check the columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] is not None:
            return board[0][col]

    # check the diagonals
    if board[0][0] == board[1][1] == board[2][2] is not None:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] is not None:
        return board[0][2]

    # board can only be in state of a tie
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    
    [[EMPTY, EMPTY, EMPTY],
     [EMPTY, EMPTY, EMPTY],
     [EMPTY, EMPTY, EMPTY]]
    """
    # check the rows
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] is not None:
            return True
    
    # check the columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] is not None:
            return True
    
    # check the diagonals
    if board[0][0] == board[1][1] == board[2][2] is not None:
        return True
    if board[0][2] == board[1][1] == board[2][0] is not None:
        return True
    
    # check for a tie
    for row in board:
        if None in row:
            return False

    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    # You may assume utility will only be called on a board if terminal(board) is True.
    the_winner = winner(board)
    if the_winner is None:
        return 0
    
    if the_winner == X:
        return 1
    
    if the_winner == O:
        return -1
    

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    
    # if human player has chosen O the board is empty when the AI has its first turn, so choose randomly one, instead 
    # of calculate all possible plays (too much)
    if board == initial_state():
        return random.randrange(0, 2), random.randrange(0, 2)
    
    # the maximizing player is X
    if player(board) == X:
        scores = set()
        for action in actions(board):
            m = min_value(result(board, action))
            
            if m == 1:
                return action
            
            scores.add((m, action))
        
        return max(scores, key=itemgetter(0))[1]

    # the minimizing player is O
    scores = set()
    for action in actions(board):
        m = max_value(result(board, action))

        if m == -1:
            return action

        scores.add((m, action))

    return min(scores, key=itemgetter(0))[1]


def max_value(board):
    """
    The maximizing player picks action a in Actions(s) that produces the highest value of Min-Value(Result(s, a)).
    """
    
    v = -math.inf
    
    if terminal(board):
        return utility(board)
    
    for action in actions(board):
        v = max(v, min_value(result(board, action)))
        
    return v


def min_value(board):
    """
    The minimizing player picks action a in Actions(s) that produces the lowest value of Max-Value(Result(s, a))
    """
    v = math.inf

    if terminal(board):
        return utility(board)

    for action in actions(board):
        v = min(v, max_value(result(board, action)))

    return v