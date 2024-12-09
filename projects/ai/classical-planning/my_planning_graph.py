
from itertools import chain, combinations
from aimacode.planning import Action
from aimacode.utils import expr

from layers import BaseActionLayer, BaseLiteralLayer, makeNoOp, make_node


class ActionLayer(BaseActionLayer):

    def _inconsistent_effects(self, actionA, actionB):
        """ Return True if an effect of one action negates an effect of the other

        Hints:
            (1) `~Literal` can be used to logically negate a literal
            (2) `self.children` contains a map from actions to effects

        See Also
        --------
        layers.ActionNode
        """
        # Get effects of actionA and actionB from self.children (if this map exists)
        effectsA = self.children[actionA]
        effectsB = self.children[actionB]
    
        # Compare effects between Action A and Action B
        for effectA in effectsA:
            for effectB in effectsB:                               
                if effectA == ~effectB:  # Logical negation
                    return True  # Inconsistent effects found
    
        return False  # No inconsistency found


    def _interference(self, actionA, actionB):
        """ Return True if the effects of either action negate the preconditions of the other 

        Hints:
            (1) `~Literal` can be used to logically negate a literal
            (2) `self.parents` contains a map from actions to preconditions
        
        See Also
        --------
        layers.ActionNode
        """
        # Retrieve preconditions of actionA and actionB from self.parents
        preconditionsA = self.parents[actionA]
        preconditionsB = self.parents[actionB]
    
        # Retrieve effects of actionA and actionB
        effectsA = self.children[actionA]
        effectsB = self.children[actionB]
    
        # Check if any effect of actionA negates a precondition of actionB
        for effectA in effectsA:
            for precondB in preconditionsB:
                if effectA == ~precondB:
                    return True  # Interference found
    
        # Check if any effect of actionB negates a precondition of actionA
        for effectB in effectsB:
            for precondA in preconditionsA:
                if effectB == ~precondA:
                    return True  # Interference found
    
        # If no interference is found
        return False

    def _competing_needs(self, actionA, actionB):
        """ Return True if any preconditions of the two actions are pairwise mutex in the parent layer

        Hints:
            (1) `self.parent_layer` contains a reference to the previous literal layer
            (2) `self.parents` contains a map from actions to preconditions
        
        See Also
        --------
        layers.ActionNode
        layers.BaseLayer.parent_layer
        """
        preconditionsA = self.parents[actionA]
        preconditionsB = self.parents[actionB]
    
        # Iterate over each precondition of actionA
        for precondA in preconditionsA:
            # For each precondition of actionB, check if the pair is mutex in the parent literal layer
            # Hint here: the mutexes are already determined in the previous layers so a check with is_mutex is sufficient
            for precondB in preconditionsB:
                if self.parent_layer.is_mutex(precondA, precondB):  # Directly check if they are mutex
                    return True  # Competing needs found
    
        # If no pair of preconditions is mutex, return False
        return False


class LiteralLayer(BaseLiteralLayer):

    def _inconsistent_support(self, literalA, literalB):
        """ Return True if always to achieve both literals are pairwise mutex in the parent layer

        Hints:
            (1) `self.parent_layer` contains a reference to the previous action layer
            (2) `self.parents` contains a map from literals to actions in the parent layer

        See Also
        --------
        layers.BaseLayer.parent_layer
        """
        actionsA = self.parents[literalA]
        actionsB = self.parents[literalB]
    
        # Check if every combination of actions from actionsA and actionsB are mutex
        for actionA in actionsA:
            for actionB in actionsB:
                # If any pair of actions is not mutex, return False 
                # The previous check for the mutex actions is already stored, so no need to check it again here
                if not self.parent_layer.is_mutex(actionA, actionB):
                    return False
    
        # If all pairs of actions are mutex, return True
        return True

    def _negation(self, literalA, literalB):
        """ Return True if two literals are negations of each other """
        return literalA == ~literalB


class PlanningGraph:
    def __init__(self, problem, state, serialize=True, ignore_mutexes=False):
        """
        Parameters
        ----------
        problem : PlanningProblem
            An instance of the PlanningProblem class

        state : tuple(bool)
            An ordered sequence of True/False values indicating the literal value
            of the corresponding fluent in problem.state_map

        serialize : bool
            Flag indicating whether to serialize non-persistence actions. Actions
            should NOT be serialized for regression search (e.g., GraphPlan), and
            _should_ be serialized if the planning graph is being used to estimate
            a heuristic
        """
        self._serialize = serialize
        self._is_leveled = False
        self._ignore_mutexes = ignore_mutexes
        self.goal = set(problem.goal)

        # make no-op actions that persist every literal to the next layer
        no_ops = [make_node(n, no_op=True) for n in chain(*(makeNoOp(s) for s in problem.state_map))]
        self._actionNodes = no_ops + [make_node(a) for a in problem.actions_list]
        
        # initialize the planning graph by finding the literals that are in the
        # first layer and finding the actions they should be connected to
        literals = [s if f else ~s for f, s in zip(state, problem.state_map)]
        layer = LiteralLayer(literals, ActionLayer(), self._ignore_mutexes)
        layer.update_mutexes()
        self.literal_layers = [layer]
        self.action_layers = []

    def h_levelsum(self):
        """ Calculate the level sum heuristic for the planning graph

        The level sum is the sum of the level costs of all the goal literals
        combined. The "level cost" to achieve any single goal literal is the
        level at which the literal first appears in the planning graph. Note
        that the level cost is **NOT** the minimum number of actions to
        achieve a single goal literal.
        
        For example, if Goal_1 first appears in level 0 of the graph (i.e.,
        it is satisfied at the root of the planning graph) and Goal_2 first
        appears in level 3, then the levelsum is 0 + 3 = 3.

        Hints
        -----
          (1) See the pseudocode folder for help on a simple implementation
          (2) You can implement this function more efficiently than the
              sample pseudocode if you expand the graph one level at a time
              and accumulate the level cost of each goal rather than filling
              the whole graph at the start.

        See Also
        --------
        Russell-Norvig 10.3.1 (3rd Edition)
        """
        goal_literals = set(self.goal)  # The goal state (literals)
        checked_goals = set()  # Track checked goals
        cost = []  # Store the cost (levels) at which goals are achieved
        
        # Start checking from LiteralLayer 0 (cost 0 for goals already achieved in the initial state)
        level = 0
        
        # Expand and check levels until all goals are found
        while len(checked_goals) < len(goal_literals):
            # If it's not the first iteration, extend the graph by adding a new action and literal layer
            if level > 0: # remember we want to search also in the intial state
                self._extend()
        
            # Get the current literal layer (last added)
            literal_layer = self.literal_layers[-1]
        
            # Filter out only positive literals (ignore negated ones)
            current_literals = [literal for literal in literal_layer if not literal.op.startswith('~')]
        
            # Check if any goal literals are in this layer
            for goal in goal_literals - checked_goals:
                # Check if the goal is present in this layer (no mutex check necessary)
                if goal in current_literals:                    
                    cost.append(level)  # Record the level at which this goal is found
                    checked_goals.add(goal)  # Mark the goal as checked
    
            # Move to the next level
            level += 1
        
            # Stop if the planning graph has leveled off (no more expansion possible)
            if self._is_leveled:
                break
        
        # Return the sum of all levels where goals were first found
        return sum(cost)

    def h_maxlevel(self):
        """ Calculate the max level heuristic for the planning graph

        The max level is the largest level cost of any single goal fluent.
        The "level cost" to achieve any single goal literal is the level at
        which the literal first appears in the planning graph. Note that
        the level cost is **NOT** the minimum number of actions to achieve
        a single goal literal.

        For example, if Goal1 first appears in level 1 of the graph and
        Goal2 first appears in level 3, then the levelsum is max(1, 3) = 3.

        Hints
        -----
          (1) See the pseudocode folder for help on a simple implementation
          (2) You can implement this function more efficiently if you expand
              the graph one level at a time until the last goal is met rather
              than filling the whole graph at the start.

        See Also
        --------
        Russell-Norvig 10.3.1 (3rd Edition)

        Notes
        -----
        WARNING: you should expect long runtimes using this heuristic with A*
        """
        goal_literals = set(self.goal)  # The goal state (literals)
        checked_goals = set()  # Track checked goals
        max_value = 0
    
        level = 0  # Start checking from the initial state (LiteralLayer 0)
    
        # Expand and check levels until all goals are found or the graph levels off
        while len(checked_goals) < len(goal_literals):
            # Expand the graph, but skip extending on level 0 (initial state)
            if level > 0:
                self._extend()
    
            # Get the current literal layer (last added)
            literal_layer = self.literal_layers[-1]
    
            # Iterate over the goals that haven't been checked yet
            for goal in goal_literals - checked_goals:
                # Check if the goal is in the current literal layer
                if goal in literal_layer:
                    max_value = max(max_value, level)
                    checked_goals.add(goal)
    
                # If all goals are checked, break early
                if len(checked_goals) == len(goal_literals):
                    return max_value
    
            level += 1  # Move to the next level
    
            # Stop if the planning graph has leveled off (no more expansion possible)
            if self._is_leveled:
                break
    
        return max_value

    def h_setlevel(self):
        """ Calculate the set level heuristic for the planning graph

        The set level of a planning graph is the first level where all goals
        appear such that no pair of goal literals are mutex in the last
        layer of the planning graph.

        Hints
        -----
          (1) See the pseudocode folder for help on a simple implementation
          (2) You can implement this function more efficiently if you expand
              the graph one level at a time until you find the set level rather
              than filling the whole graph at the start.

        See Also
        --------
        Russell-Norvig 10.3.1 (3rd Edition)

        Notes
        -----
        WARNING: you should expect long runtimes using this heuristic on complex problems
        """
        goal_literals = set(self.goal)  # The goal state (literals)
        level = 0  # Start checking from the initial state (LiteralLayer 0)
    
        # Expand and check levels until all goals are found
        while True:
            # Extend the graph if we are past the initial level
            if level > 0:
                self._extend()
    
            # Get the current literal layer (last added)
            literal_layer = self.literal_layers[-1]
    
            # Check if all goals are present in this layer
            if all(goal in literal_layer for goal in goal_literals):
                # Check for mutex relations between goals
                if not any(literal_layer.is_mutex(goal1, goal2)
                           for goal1 in goal_literals for goal2 in goal_literals if goal1 != goal2):
                    return level  # Return the first level where all goals are present and not mutex
    
            # Move to the next level
            level += 1
    
            # Stop if the planning graph has leveled off (no more expansion possible)
            if self._is_leveled:
                break
    
        # If no valid set level is found, return a large number (indicating failure)
        return float('inf')

    ##############################################################################
    #                     DO NOT MODIFY CODE BELOW THIS LINE                     #
    ##############################################################################

    def fill(self, maxlevels=-1):
        """ Extend the planning graph until it is leveled, or until a specified number of
        levels have been added

        Parameters
        ----------
        maxlevels : int
            The maximum number of levels to extend before breaking the loop. (Starting with
            a negative value will never interrupt the loop.)

        Notes
        -----
        YOU SHOULD NOT THIS FUNCTION TO COMPLETE THE PROJECT, BUT IT MAY BE USEFUL FOR TESTING
        """
        while not self._is_leveled:
            if maxlevels == 0: break
            self._extend()
            maxlevels -= 1
        return self

    def _extend(self):
        """ Extend the planning graph by adding both a new action layer and a new literal layer

        The new action layer contains all actions that could be taken given the positive AND
        negative literals in the leaf nodes of the parent literal level.

        The new literal layer contains all literals that could result from taking each possible
        action in the NEW action layer. 
        """
        if self._is_leveled: return

        parent_literals = self.literal_layers[-1]
        parent_actions = parent_literals.parent_layer
        action_layer = ActionLayer(parent_actions, parent_literals, self._serialize, self._ignore_mutexes)
        literal_layer = LiteralLayer(parent_literals, action_layer, self._ignore_mutexes)

        for action in self._actionNodes:
            # actions in the parent layer are skipped because are added monotonically to planning graphs,
            # which is performed automatically in the ActionLayer and LiteralLayer constructors
            if action not in parent_actions and action.preconditions <= parent_literals:
                action_layer.add(action)
                literal_layer |= action.effects

                # add two-way edges in the graph connecting the parent layer with the new action
                parent_literals.add_outbound_edges(action, action.preconditions)
                action_layer.add_inbound_edges(action, action.preconditions)

                # # add two-way edges in the graph connecting the new literaly layer with the new action
                action_layer.add_outbound_edges(action, action.effects)
                literal_layer.add_inbound_edges(action, action.effects)

        action_layer.update_mutexes()
        literal_layer.update_mutexes()
        self.action_layers.append(action_layer)
        self.literal_layers.append(literal_layer)
        self._is_leveled = literal_layer == action_layer.parent_layer
