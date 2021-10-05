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
        for i in range(self.iterations):
            currentValues = self.values.copy()
            for s in states:
                if MDP.isTerminal(s):
                    continue
                actions = MDP.getPossibleActions(s)
                tempActionValues = []
                for a in actions:
                    T = MDP.getTransitionStatesAndProbs(s, a)
                    # actionTransitionSum = sum([t[1] * (MDP.getReward(s, a, t[0]) + discount * currentValues[t[0]]) for t in T])
                    actionTransitionSum = 0
                    for t in T:
                        nextState = t[0]
                        probability = t[1]
                        valueNextState = currentValues[nextState]
                        actionTransitionSum += probability * (MDP.getReward(s, a, nextState) + discount * valueNextState)
                    tempActionValues.append(actionTransitionSum)
                maxActionValue = max(tempActionValues)
                self.values[s] = maxActionValue


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
        if MDP.isTerminal(state): return None
        actions = MDP.getPossibleActions(state)
        actionCounter = util.Counter()
        for a in actions:
            T = MDP.getTransitionStatesAndProbs(state, a)
            actionTransitionSum = 0
            for t in T:
                nextState = t[0]
                probability = t[1]
                valueNextState = currentValues[nextState]
                actionTransitionSum += probability * (MDP.getReward(state, a, nextState) + discount * valueNextState)
            actionCounter[a] = actionTransitionSum
        return actionCounter.argMax()
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
        numStates = len(MDP.getStates())
        for i in range(self.iterations):
            s = states[i % numStates]
            if MDP.isTerminal(s):
                continue
            actions = MDP.getPossibleActions(s)
            tempActionValues = []
            for a in actions:
                T = MDP.getTransitionStatesAndProbs(s, a)
                # actionTransitionSum = sum([t[1] * (MDP.getReward(s, a, t[0]) + discount * currentValues[t[0]]) for t in T])
                actionTransitionSum = 0
                for t in T:
                    nextState = t[0]
                    probability = t[1]
                    valueNextState = self.values[nextState]
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
        # Calculate predecessors of all states
        predecessors = {}
        for s in states:
            predecessors[s] = set()
        for s in states:
            possibleNextStates = set()
            actions = MDP.getPossibleActions(s)
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
