from pacai.agents.learning.value import ValueEstimationAgent
from pacai.util import counter


class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate=0.9, iters=100, **kwargs):
        super().__init__(index)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = counter.Counter()  # A Counter is a dict with default 0

        # Compute the values here.
        for i in range(iters):
            value = counter.Counter()
            # print("the new value is ", value)
            for state in self.mdp.getStates():
                action = self.getAction(state)
                # print("the action is ", action)
                if action is not None:
                    value[state] = self.getQValue(state, action)
                    # print("the state is ", state)
                    # print("the value is ", value)

            self.values = value

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values[state]

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        bestAction = actions[0]
        # print("the action is ", bestAction)
        maxV = self.getQValue(state, actions[0])
        # print("the value is", maxV)

        for action in actions:
            value = self.getQValue(state, action)
            if maxV <= value:
                maxV = value
                bestAction = action

        return bestAction

    def getPolicy(self, state):
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        max_value = self.getQValue(state, actions[0])
        max_action = actions[0]

        for action in actions:
            value = self.getQValue(state, action)
            if max_value <= value:
                max_value = value
                max_action = action

        return max_action

    def getQValue(self, state, action):
        score = 0
        # get a list of possible state and probability
        stateProb = self.mdp.getTransitionStatesAndProbs(state, action)
        # print(stateProb)
        for states, prob in stateProb:
            currentReward = self.mdp.getReward(state, action, states)
            value = self.getValue(states)
            score += prob * (currentReward + self.discountRate * value)

        return score
