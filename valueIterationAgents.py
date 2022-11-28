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
        for iteration in range(self.iterations):
            iteration_value = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    iteration_value[state] = 0.0
                else:
                    optimal_action = self.computeActionFromValues(state)
                    iteration_value[state] = self.computeQValueFromValues(state, optimal_action)
            self.values = iteration_value

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
        # transition_function = self.mdp.getTransitionStatesAndProbs(state, action)
        # discount = self.discount

        # return sum([i[1] * (self.mdp.getReward(state,action,i[0]) + gamma * self.values[i[0]])  for i in transition_function])
        transition_function = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0
        for successor, probability in transition_function:
            reward = self.mdp.getReward(state, action, successor)
            discount = self.discount
            utility = self.getValue(successor)
            q_value += probability * (reward + discount * utility)
        return q_value
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
        if self.mdp.isTerminal(state):
            return None;

        possible_action = self.mdp.getPossibleActions(state)
        q_values =[]
        optimal_action = {}

        for action in possible_action:
            val = self.computeQValueFromValues(state, action)
            q_values.append(val)
            optimal_action[val] = action

        return optimal_action[max(q_values)]
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
        iterations = self.iterations
        states = self.mdp.getStates()

        for iteration in range(0, iterations):

            value = self.values.copy()
            state = states[iteration % len(states)]

            if (self.mdp.isTerminal(state)):
                value[state] = 0
                continue

            optimal_value = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
            value[state] = optimal_value

            self.values = value

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
        predecessors = {}
        priority_q = util.PriorityQueue()

        for state in self.mdp.getStates():
            predecessors[state] = set()

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            else:
                possible_action = self.mdp.getPossibleActions(state)
                for action in possible_action:
                    successor = self.mdp.getTransitionStatesAndProbs(state, action)
                    for next_state, prob in successor:
                        if prob > 0:
                            predecessors[next_state].add(state)

                utility = self.values[state]
                optimal_action = self.computeActionFromValues(state)
                max_q = self.computeQValueFromValues(state, optimal_action)

                delta = abs(utility - max_q)
                priority_q.push(state, -delta)

        for iteration in range(0, self.iterations):
            if priority_q.isEmpty():
                break
            else:
                state = priority_q.pop()
                optimal_action = self.computeActionFromValues(state)
                self.values[state] = self.computeQValueFromValues(state, optimal_action)

                for predecessor in predecessors[state]:
                    utility = self.values[predecessor]
                    optimal_action = self.computeActionFromValues(predecessor)
                    max_q = self.computeQValueFromValues(predecessor, optimal_action)

                    delta = abs(utility - max_q)
                    if delta > self.theta:
                        priority_q.update(predecessor, -delta)