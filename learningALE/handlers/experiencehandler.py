__author__ = 'Ben'
import numpy as np


class ExperienceHandler:
    """
    Depreciated in favor of :class:`DataSet` written by spragunr. It is faster, has a lower memory footprint, and is
    battle tested.
    The :class:`ExperienceHandler` class handles storing of experience used for experience replay techniques. It keeps a
    list of states, rewards, actions, and terminal states. Design decisions: This class is relatively optimized, Lists
    were chosen because they are easy to append to and can be easily converted to numpy arrays for training.

    Parameters
    ----------
    max_len : int
       Specifies the maximum number of states to store. Number of actions/rewards will be this number / frame_skip
    frame_skip : int
        Specifies the frame skip value
    replay_start_size : int
        Default 100, number of states to wait for before returning something in :meth:`ExperienceHandler.get_random_experience`
    """
    def __init__(self, max_len, frame_skip, replay_start_size=100):
        self.states = list()
        self.rewards = list()
        self.actions = list()
        self.term_states = set()
        self.frame_skip = frame_skip
        self.size = 0
        self.max_len = max_len
        self.replay_start_size = replay_start_size

    def add_experience(self, states, action, reward):
        """
        Adds an experience (states, action, reward) to their respective lists

        Parameters
        ----------
        states : iterable
         Represents game states for this action/reward will be of len(frame skip)
        action : int
         Action_ind from learner
        reward : int
         Reward from ALE

        Returns
        -------
        size : int
            Current size of stored states
        """
        self.states.extend(states)
        self.actions.append(action)
        self.rewards.append(reward)
        self.size += self.frame_skip
        return self.size

    def add_terminal(self):
        """
        Adds a terminal state indicator at the last inserted state
        """
        self.term_states.add(self.size/self.frame_skip-1)  # add a terminal state indicator.

    def get_random_experience(self, mini_batch, dtype=np.float32):
        """
        Used to get a random experience from memory.
        Returns numpy arrays representing state, action, reward, state time+1, terminal indicators, and indexes of where
        these are stored in this class' lists. Can return None if size < replay_start_size. Otherwise returns random
        sampling (with replacement) of stored experiences in the format: (states, actions, rewards, states time+1,
        terminal, indexes).

        Parameters
        ----------
        mini_batch : int
         Number of experiences to get
        dtype : numpy data type
         Automatically converts returned states & states time+1 in this datatype

        Returns
        -------
        states : numpy.array(dtype)
         Numpy.array of dtype with shape (mini_batch,) + (state.shape)
        actions : numpy.array(int)
         Learner action_inds of the action performed
        rewards : numpy.array(int)
         Reward from ALE
        states_tp1 : numpy.array(dtype)
         Numpy.array of dtype with shape (mini_batch,) + (state.shape). Will be zeros if terminal is 1 for the
         corresponding state
        terminal : numpy.array(int)
         Whether a state is terminal or not
        indexes : numpy.array(int)
         Index of where this return is stored in lists. Mostly unused except for testing
        """
        if self.size <= self.replay_start_size:
            return None, None, None, None, None, None

        # Design decision: I know this can return duplicates, but is unlikely when size is large
        # It's also much more efficient see test/randint_vs_choice
        rp = np.random.randint(self.size - self.frame_skip - 1, size=mini_batch)
        state_shape = self.states[0].shape

        # create empty arrays to hold state data, faster than appending and doing asarray
        mb_states = np.zeros((mini_batch, self.frame_skip) + state_shape, dtype=dtype)
        mb_states_tp1 = np.zeros((mini_batch, self.frame_skip) + state_shape, dtype=dtype)

        # lists are okay for integers, and I think faster than creating an empty array and accessing by index
        mb_actions = list()
        mb_rewards = list()
        mb_terminal = list()
        mb_inds = list()

        # loop over the randomly selected state indicators to create minibatch
        for exp in range(mini_batch):
            rand_ind = rp[exp]
            frame_inds = np.arange(0, self.frame_skip) + rand_ind  # generate frame_skip number of frames in a row

            # we have to do a conversion here to get the correct index of the action for the selected state index
            # states are added as iterables of len(frame_skip) actions/rewards. So we need to take the indicator and
            # divide by frame_skip to get the action and reward. Also NOTE: we take the action/reward of the last frame
            ar_ind = frame_inds[-1] // self.frame_skip  # we need a integer ind
            mb_inds.append(ar_ind)
            mb_states[exp] = [self.states[frame] for frame in frame_inds]
            mb_actions.append(self.actions[ar_ind])
            mb_rewards.append(self.rewards[ar_ind])
            if ar_ind in self.term_states:  # term states is based on action/reward ind
                mb_terminal.append(1)
            else:
                mb_terminal.append(0)
                mb_states_tp1[exp] = [self.states[frame+1] for frame in frame_inds]

        mb_actions = np.asarray(mb_actions, dtype=int)
        mb_rewards = np.asarray(mb_rewards, dtype=int)
        mb_terminal = np.asarray(mb_terminal, dtype=int)
        mb_inds = np.asarray(mb_inds, dtype=int)
        return mb_states, mb_actions, mb_rewards, mb_states_tp1, mb_terminal, mb_inds

    def trim(self):
        """
        Trims/removes first inserted values until self.size < self.max_len. Moves terminal indicators to their new
        respective indexes. This code is not meant to be fast so it should not be called each timestep.
        """
        while len(self.states) > self.max_len:
            for itr in range(self.frame_skip):
                del self.states[0]

                self.size -= 1
            # rewards, actions and terminals are stored per each frame_skip number of frames so only delete one
            del self.rewards[0]
            del self.actions[0]
            # if we are deleting a state we need to decrement the terminal state indicators
            term_states = np.asarray(list(self.term_states))
            term_states -= 1
            term_states = term_states[term_states >= 0]
            self.term_states = set(term_states)
