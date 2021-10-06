# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        MDP = self.mdp
        states = MDP.getStates()
        discount = self.discount
        # Need to do self.iterations iterations of value iteration.
        # This involves updating each of the state values once per iteration based on the previous iteration's values.
        # Will look at each of the actions, sum up the value for that action based on transition and reward model,
        # and finally set the value of the state for this iteration to the max value out of the actions.
        for i in range(self.iterations): # For as many iterations defined in the construction
            currentValues = self.values.copy() # Copy the current values (V_k)
            for s in states: # For each state
                if MDP.isTerminal(s): # Check if it's a terminal state
                    continue # If so, move on to the next state because terminal states have no actions/transitions
                actions = MDP.getPossibleActions(s) # Get the possible actions from state
                tempActionValues = [] # Temporary list to store values for actions from state
                for a in actions: # For each actions possible from state
                    T = MDP.getTransitionStatesAndProbs(s, a) # Get the transitions for that (state, action) pair
                    # actionTransitionSum = sum([t[1] * (MDP.getReward(s, a, t[0]) + discount * currentValues[t[0]]) for t in T])
                    actionTransitionSum = 0 # Sum of values for each transition possible for action
                    for t in T: # For each transition
                        nextState = t[0]
                        probability = t[1]
                        valueNextState = currentValues[nextState] # Using V_k values to calculate V_k+1
                        actionTransitionSum += probability * (MDP.getReward(s, a, nextState) + discount * valueNextState)
                    tempActionValues.append(actionTransitionSum) # Add sum of transitions for each action to list
                maxActionValue = max(tempActionValues) # Get the max of the action values
                self.values[s] = maxActionValue # Update V(state) to this max value (V_k+1)


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        MDP = self.mdp
        discount = self.discount
        currentValues = self.values.copy()
        # This function calculates the Q value of a (state, action) pair by summing the value of each possible transition
        # according to the action, state, and value of the next state in our current self.values.
        T = MDP.getTransitionStatesAndProbs(state, action)
        actionTransitionSum = 0
        for t in T:
            nextState = t[0]
            probability = t[1]
            valueNextState = currentValues[nextState]
            actionTransitionSum += probability * (MDP.getReward(state, action, nextState) + discount * valueNextState)
        return actionTransitionSum
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        MDP = self.mdp
        discount = self.discount
        currentValues = self.values.copy()
        # This function is trying to compute the best action for a certain state from the current values of the states.
        # It essentially performs one step of value iteration for a state and returns the action with the highest value.
        # Instead of storing the action values and taking the max, the action values are stored in a Counter/dict
        # with the key = action and value = actionValue where argmax is used to return the key with the highest value.
        # In the case of ties, the max action seen first is chosen.
        if MDP.isTerminal(state): return None # If it's a terminal state, there is no action (None)
        actions = MDP.getPossibleActions(state) # Get possible actions from state
        actionCounter = util.Counter() # This counter will store the action and it's action value
        for a in actions: # For each action
            T = MDP.getTransitionStatesAndProbs(state, a) # Get the possible transitions
            actionTransitionSum = 0
            for t in T: # As before, sum up the value for each transition according to T, R, and current values.
                nextState = t[0]
                probability = t[1]
                valueNextState = currentValues[nextState]
                actionTransitionSum += probability * (MDP.getReward(state, a, nextState) + discount * valueNextState)
            actionCounter[a] = actionTransitionSum # Insert each (action, actionValue) key-value pair into the counter.
        return actionCounter.argMax() # Return the argmax of this counter (the action with highest actionValue).
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        MDP = self.mdp
        states = MDP.getStates()
        discount = self.discount
        # For asynchronous value iteration, only one state is updated per iteration where the first iteration updates
        # the first state, second iteration updates the second state, etc. When all states have been updated, it loops back
        # to the first state and continues like this for self.iterations iterations.
        numStates = len(MDP.getStates()) # Getting the number of states
        for i in range(self.iterations): # For self.iterations iterations
            s = states[i % numStates] # The current state to be updated is the current iteration mod the number of states
            # EX: 2 states. 0 -> states[0%2] = states[0], 1 -> states[1%2] = states[1], 2 -> states[2%2] -> states[0]...
            if MDP.isTerminal(s): # If in terminal state, move on
                continue
            actions = MDP.getPossibleActions(s) # Nothing else too different from regular value iteration here and onwards.
            tempActionValues = []
            for a in actions:
                T = MDP.getTransitionStatesAndProbs(s, a)
                actionTransitionSum = 0
                for t in T:
                    nextState = t[0]
                    probability = t[1]
                    valueNextState = self.values[nextState]
                    # One difference is that we can directly use self.values instead of a copy because only one state is changed per iteration.
                    actionTransitionSum += probability * (MDP.getReward(s, a, nextState) + discount * valueNextState)
                tempActionValues.append(actionTransitionSum)
            maxActionValue = max(tempActionValues)
            self.values[s] = maxActionValue



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        MDP = self.mdp
        states = MDP.getStates()
        discount = self.discount
        # For prioritized sweeping value iteration, we are updating states according to a priority queue.
        # This involves finding the predecessors of all states and pushing states onto the priority queue
        # according to the difference in their current value and their value if an iteration of value iteration was
        # performed on it.
        # When a state's value is updated, all of its predecessors update their priority in the queue according
        # to the new value for the updated state (if their priority increases).
        # Note that the priority is defined as -diff because a min heap was used for the priority queue so updating
        # a priority actually means checking if the value has decreased (become more negative = higher priority).
        # Calculate predecessors of all states
        predecessors = {} # Predecessors will be a dict with key = one state, value = set of predecessors of that state
        for s in states: # Setting up the dict values for each state to empty set
            predecessors[s] = set()
        # The idea is to go through each state s, and note each state that s can lead to.
        # Once all the possible next states are known, add s to the predecessors of those next states.
        # Instead of calculating the predecessors directly, I'm finding which states that s is a possible predecessor
        # to and adding s to their predecessor set.
        for s in states: # For each state
            possibleNextStates = set()
            # I'm using sets so the same state doesn't end up in the nextStates more than once
            # which may occur if two actions can lead to the same state.
            actions = MDP.getPossibleActions(s) # Get possible actions
            for a in actions:
                T = MDP.getTransitionStatesAndProbs(s, a)
                for t in T:
                    nextState = t[0]
                    possibleNextStates.add(nextState)
            for succ in possibleNextStates:
                predecessors[succ] = predecessors[succ] | {s}

        priorityQueue = util.PriorityQueue()
        for s in states:
            if MDP.isTerminal(s):
                continue
            actions = MDP.getPossibleActions(s)
            tempActionValues = []
            for a in actions:
                T = MDP.getTransitionStatesAndProbs(s, a)
                actionTransitionSum = sum([t[1] * (MDP.getReward(s, a, t[0]) + discount * self.values[t[0]]) for t in T])
                tempActionValues.append(actionTransitionSum)
            maxActionValue = max(tempActionValues)
            diff = abs(self.values[s] - maxActionValue)
            priorityQueue.push(s, -diff)

        for i in range(self.iterations):
            if priorityQueue.isEmpty():
                break
            s = priorityQueue.pop()
            if MDP.isTerminal(s):
                continue
            actions = MDP.getPossibleActions(s)
            tempActionValues = []
            for a in actions:
                T = MDP.getTransitionStatesAndProbs(s, a)
                actionTransitionSum = sum([t[1] * (MDP.getReward(s, a, t[0]) + discount * self.values[t[0]]) for t in T])
                tempActionValues.append(actionTransitionSum)
            maxActionValue = max(tempActionValues)
            self.values[s] = maxActionValue

            preds_of_s = predecessors[s]
            for p in preds_of_s:
                if MDP.isTerminal(p):
                    continue
                actions = MDP.getPossibleActions(p)
                tempActionValues = []
                for a in actions:
                    T = MDP.getTransitionStatesAndProbs(p, a)
                    actionTransitionSum = sum([t[1] * (MDP.getReward(p, a, t[0]) + discount * self.values[t[0]]) for t in T])
                    tempActionValues.append(actionTransitionSum)
                maxActionValue = max(tempActionValues)
                diff = abs(self.values[p] - maxActionValue)
                if diff > self.theta:
                    priorityQueue.update(p, -diff)
