__author__ = 'Ben'
import numpy as np


class ActionHandler:
    """
    The :class:`ActionHandler` class takes care of the interface between the action indexes returned from ALE and a
    vector of length (num actions). It also allows two different types of stochastic selection methods.
    :class:`ActionPolicy`-eGreedy where it randomly selects an action with probability e. Or
    :class:`ActionPolicy`-randVals where it adds noise to the action vector before choosing the index of the max action.

    This class supports linear annealing of both the eGreedy probability value and the randVals scalar.

    Parameters
    ----------
    action_policy : :class:`ActionPolicy`
       Specifies whether using eGreedy or adding randVals to the action value vector

    random_values : tuple
       Specifies which values to use for the action policy
       format: (Initial random value, ending random value, number of steps to anneal over)

    actions : tuple, list, array
       Default None, should be set by gameHandler.
       The legal actions from the :class:`libs.ale_python_interface.ALEInterface`
    """
    def __init__(self, action_policy, random_values, actions=None):
        self.actionPolicy = action_policy

        self.randVal = random_values[0]
        self.lowestRandVal = random_values[1]
        lin = np.linspace(random_values[0], random_values[1], random_values[2])
        self.diff = lin[0] - lin[1]
        self.countRand = 0
        self.actionCount = 0

        if actions is not None:
            self.numActions = len(actions)
            self.actions = None
            self.set_legal_actions(actions)
        else:
            self.numActions = 0
            self.actions = actions

    def get_action(self, action_values, random=True):
        """
        Get_Action takes an action_values vector from a learner of length # legal actions and will perform the
        stochastic selection policy on it.

        Parameters
        ----------
        action_values : array of length # legal actions
            Output from a learner of values for each possible action

        random : bool
            Default true. Whether to perform the stochastic action_values selection policy or just return the max value
            index.

        Returns
        -------
        action_ind : int
            Index of max action value.
        """
        if random:
            if self.actionPolicy == ActionPolicy.eGreedy:
                # egreedy policy to choose random action_values
                if np.random.uniform(0, 1) <= self.randVal:
                    e_greedy = np.random.randint(self.numActions)
                    action_values[e_greedy] = np.inf  # set this value as the max action
                    self.countRand += 1
            elif self.actionPolicy == ActionPolicy.randVals:
                action_values += np.random.randn(self.numActions) * self.randVal

        action = np.argmax(action_values)
        self.actionCount += 1

        return action

    def anneal(self):
        """
        Anneals the random value used in the stochastic action selection policy.
        """
        self.randVal -= self.diff
        if self.randVal < self.lowestRandVal:
            self.randVal = self.lowestRandVal

    def set_legal_actions(self, legal_actions):
        """
        Sets the legal actions for this handler. Sets up values need for conversion from ALE action id to learner.

        Parameters
        ----------
        legal_actions : array/list/tuple
            Legal actions in current ALE game
        """
        self.actions = np.asarray(legal_actions, dtype=int)
        assert len(self.actions.shape) == 1, "actions must be a vector"
        self.numActions = len(legal_actions)

    def game_action_to_action_ind(self, action):
        """
        Converts an action id returned from ALE to the index used in the learner.

        Parameters
        ----------
        action : int
        ALE action index

        Returns
        -------
        action_ind : int
            Action index relative to learner output vector
        """
        return np.where(action == self.actions)[0][0]

    def action_vect_to_game_action(self, action_vect, random=True):
        """
        Converts action vector output of learner to a ALE ready action id. Default performs stochastic action policy

        Parameters
        ----------
        action_vect : array
            Action vector output from learner
        random : bool
            Default True. Whether or not to use stochastic action policy

        Returns
        -------
        ale_action_ind : int
            ALE ready action id
        """
        return self.actions[self.get_action(action_vect, random)]


from enum import Enum
class ActionPolicy(Enum):
    """
    :class:`ActionPolicy` is an Enum used to determine which policy an action handler should use for
    random exploration.

    Currently supported are eGreedy and the addition of random values to the action vector (randVals)

    The idea behind adding random values can be found here:
    https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/
    """
    eGreedy = 1
    randVals = 2
