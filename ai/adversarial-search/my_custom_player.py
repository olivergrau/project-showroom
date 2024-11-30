import os
import threading
import time
from isolation import Isolation, StopSearch, DebugState
from isolation.isolation import Action
from sample_players import DataPlayer
import random

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def __init__(self, player_id):
        super().__init__(player_id)
        self.node_count = 0  # To count all nodes visited
        self.pruned_node_count = 0  # To count pruned nodes

        self.debug_log = False
        
    def log(self, message, debug=False):        
        """ Print a message if debugging is enabled """
        if self.debug_log or debug:
            print(message)
            
    def get_action(self, state: Isolation):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """

        self.node_count = 0  # Reset node count
        self.pruned_node_count = 0  # Reset pruned node count
        
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:                            
            self.iterative_deepening_search(
                state, heuristic=self.hybrid_heuristic, search_method=self.alpha_beta_search)

    def iterative_deepening_search(self, state: Isolation, heuristic, search_method):
        best_move = None

        patience_counter = 0  # Initialize the patience counter
        max_patience = 5  # Number of iterations to tolerate no new node expansion
        
        # Get the current process and thread ID
        process_id = os.getpid()
        thread_id = threading.get_ident()
    
        # Iteratively deepen the search
        depth = 1
        last_node_count = 0  # To track node expansion across iterations
        try:
            while True:
                # Reset node counts for each depth
                self.node_count = 0
                self.pruned_node_count = 0

                # Measure time before starting the depth search
                start_time = time.time()
    
                # Perform the search at the current depth                
                best_move = search_method(state, depth, heuristic)
                self.queue.put(best_move)  # Store the best move found so far
    
                # Measure time after completing the search for the current depth
                end_time = time.time()
                elapsed_time = end_time - start_time
    
                # Log the details with process/thread ID and elapsed time
                self.log(
                    f"Process ID: {process_id}, Thread ID: {thread_id}, "
                    f"Search: {search_method.__name__} with Heuristic: {heuristic.__name__}, "
                    f"Depth {depth}: Nodes evaluated: {self.node_count}, Pruned: {self.pruned_node_count}, "
                    f"Elapsed Time: {elapsed_time:.4f} seconds", False
                )  # Log after each depth

                # Check if there are any new nodes evaluated
                if self.node_count == last_node_count:
                    patience_counter += 1
                    self.log(f"No new nodes evaluated at Depth {depth}. Patience Counter: {patience_counter}")
                else:
                    patience_counter = 0  # Reset patience counter if new nodes were evaluated
    
                # If patience limit is exceeded, break out of the loop
                if patience_counter >= max_patience:
                    self.log(f"Patience exceeded at Depth {depth}. Stopping search.", False)
                    break
    
                # Update last_node_count for the next iteration
                last_node_count = self.node_count
                
                depth += 1
        except StopSearch:
            # Time limit exceeded, quickly put the best move in the queue
            if best_move is not None:
                self.queue.put(best_move)  # Make sure the best move is in the queue
            else:
                self.queue.put(random.choice(state.actions()))  # Fallback move
    
        # Ensure at least one move is returned (in case no search was completed)
        if best_move is None:
            best_move = random.choice(state.actions())
    
        self.queue.put(best_move)


    def alpha_beta_search(self, state: Isolation, depth, heuristic):
        """ Initiates the alpha-beta search process with initial alpha and beta values. """
        def max_value(state, depth, alpha, beta):
            self.node_count += 1  # Increment node counter
            
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.hybrid_heuristic(state)
    
            value = float("-inf")
    
            # Pre-sort the actions based on the heuristic evaluation (descending order)
            actions = sorted(state.actions(), key=lambda action: self.heuristic_evaluation(state, action, is_max=True), reverse=True)
    
            for action in actions:
                value = max(value, min_value(self.simulate_move(state, action, self.player_id), depth - 1, alpha, beta))
                if value >= beta:
                    self.pruned_node_count += 1  # Increment pruned node counter
                    return value  # Prune the remaining branches
                alpha = max(alpha, value)
            return value
    
        def min_value(state, depth, alpha, beta):
            self.node_count += 1  # Increment node counter
            
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.hybrid_heuristic(state)
    
            value = float("inf")
    
            # Pre-sort the actions based on the heuristic evaluation (ascending order)
            actions = sorted(state.actions(), key=lambda action: self.heuristic_evaluation(state, action, is_max=False))
    
            for action in actions:
                value = min(value, max_value(self.simulate_move(state, action, 1 - self.player_id), depth - 1, alpha, beta))
                if value <= alpha:
                    self.pruned_node_count += 1  # Increment pruned node counter
                    return value  # Prune the remaining branches
                beta = min(beta, value)
            return value
    
        # Start the search with initial alpha and beta values
        return max(state.actions(), key=lambda x: min_value(self.simulate_move(state, x, self.player_id), depth - 1, float("-inf"), float("inf")))

    def heuristic_evaluation(self, state: Isolation, action, is_max):
        """
        Heuristic function to evaluate a given action for move ordering.
    
        - `is_max`: Boolean indicating if the evaluation is for the max player (self) or the min player (opponent).
        """
        # Use simulate_move to apply the action correctly, considering the player's turn
        player_id = self.player_id if is_max else 1 - self.player_id
        new_state = self.simulate_move(state, action, player_id)  # Simulate the move for the correct player
    
        # Use the hybrid_heuristic or another suitable heuristic to evaluate the resulting state
        return self.hybrid_heuristic(new_state)

    def minimax_base(self, state, depth, heuristic):
    
        def min_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return heuristic(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1))
            return value
    
        def max_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return heuristic(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1))
            return value
    
        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))

    def simulate_move(self, state: Isolation, action, player_id):
        """ 
            Simulate the move for a specific player by temporarily adjusting ply_count.
            
            You cannot use the state object directly because every call to state.results increases the ply_count 
            variable and this means, that the active player alternates between each call. 
        """
        # Create a new state with the modified ply_count to force the correct player's move
        temp_ply_count = state.ply_count
        if player_id != state.player():  # Check if the desired player has the turn
            temp_ply_count += 1  # Temporarily adjust ply_count to give the desired player the turn
    
        # Create a copy of the state with the modified ply_count
        temp_state = Isolation(board=state.board, ply_count=temp_ply_count, locs=state.locs)
    
        # Apply the move
        return temp_state.result(action)

    def hybrid_heuristic(self, state):
        # Get the locations of both players
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
    
        self.log(f"Calculating hybrid heuristic...")
        self.log(f"Player {self.player_id} location: {own_loc}, Opponent location: {opp_loc}")
    
        # Get current mobility (available moves)
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)

        self.log(f"Own liberties: {own_liberties}")
        self.log(f"Opponent liberties: {opp_liberties}")
    
        # Proximity to the opponent (Manhattan distance between players)
        distance_to_opponent = self.manhattan_distance(own_loc, opp_loc)
        self.log(f"Distance to opponent: {distance_to_opponent}")
    
        # Valid knight actions
        valid_actions = set(Action)
    
        # Future Mobility: Only consider future moves that lead to unblocked cells
        future_own_moves = 0
        for action in own_liberties:
            if action in valid_actions and self.is_open(state, action, own_loc):
                self.log(f"Trying action {action} for player {self.player_id} at location {own_loc}")
                self.print_board(state, own_loc, action)  # Print board before applying result
                try:
                    new_state = self.simulate_move(state, action, self.player_id)  # Simulate move for own player
                    liberties = state.liberties(new_state.locs[self.player_id])
                    future_own_moves += len(liberties)
                    self.log(f"Future own moves: {liberties}")
                except RuntimeError as e:
                    self.log(f"Error in future move for own player: {e}")
    
        future_opp_moves = 0
        for action in opp_liberties:
            if action in valid_actions and self.is_open(state, action, opp_loc):
                self.log(f"Trying action {action} for opponent at location {opp_loc}")
                self.print_board(state, opp_loc, action)  # Print board before applying result
                try:
                    new_state = self.simulate_move(state, action, 1 - self.player_id)  # Simulate move for opponent
                    liberties = state.liberties(new_state.locs[1 - self.player_id])
                    future_opp_moves += len(liberties)
                    self.log(f"Future opponent moves: {liberties}")
                except RuntimeError as e:
                    self.log(f"Error in future move for opponent: {e}")
    
        # Heuristic weighting factors
        mobility_weight = 1.0
        future_mobility_weight = 0.5
        proximity_weight = -0.2
        trap_bonus = -100 if len(opp_liberties) <= 2 else 0
    
        # Final heuristic calculation
        mobility_score = mobility_weight * (len(own_liberties) - len(opp_liberties))
        future_mobility_score = future_mobility_weight * (future_own_moves - future_opp_moves)
        proximity_score = proximity_weight * distance_to_opponent

        self.log(f"Mobility score: {mobility_score}")
        self.log(f"Future mobility score: {future_mobility_score}")
        self.log(f"Proximity score: {proximity_score}")
        self.log(f"Trap bonus: {trap_bonus}")
    
        heuristic_value = mobility_score + future_mobility_score + proximity_score + trap_bonus
        self.log(f"Final heuristic value: {heuristic_value}\n")
    
        return heuristic_value


    # Print the current board configuration for debugging
    def print_board(self, state, player_location, action):
        """
        Print the current board state before the result method is applied.
        :param state: The current game state
        :param player_location: The current location of the player
        :param action: The action (move) to be applied
        """
        # Get the DebugState version of the current state
        debug_state = DebugState.from_state(state)
    
        # Print the current board with player positions
        self.log("print_board: Current Board:")
        self.log(debug_state)
    
        # Print the player's current location and the action they are about to take
        target_location = player_location + action
        self.log(f"print_board: Player at location {player_location} is about to take action {action}")
        self.log(f"print_board: Target location after action: {target_location}")
        self.log(f"print_board: Is the target location open? {self.is_open(state, action, player_location)}")
    
    def is_open(self, state, action, player_location):
        """
        Check if the target cell is open and not blocked.
        :param state: The current game state
        :param action: The action (move) to evaluate
        :param player_location: The current location of the player
        :return: True if the target cell is open, False if it is blocked
        """
        target_location = player_location + action
        return bool(state.board & (1 << target_location))  # Returns True if the target cell is open
        
    def manhattan_distance(self, loc1, loc2):
        x1, y1 = DebugState.ind2xy(loc1)
        x2, y2 = DebugState.ind2xy(loc2)
        return abs(x1 - x2) + abs(y1 - y2)    
    
    def movement_heuristic(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
