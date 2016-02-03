from learningALE.learners.learner import learner
from learningALE.handlers.actionhandler import ActionHandler, ActionPolicy
from learningALE.handlers.experiencehandler import ExperienceHandler
from learningALE.handlers.dataset import DataSet
from learningALE.learners.nns import CNN
import numpy as np


class DQNLearner(learner):
    def __init__(self, skip_frame, num_actions, load=None, random_state=np.random.RandomState()):
        super().__init__()

        rand_vals = (1, 0.1, 1000000)  # starting at 1 anneal eGreedy policy to 0.1 over 1,000,000 actions
        self.action_handler = ActionHandler(ActionPolicy.eGreedy, rand_vals)

        self.minimum_replay_size = 100
        self.exp_handler = DataSet(84, 84, random_state, max_steps=1000000, phi_length=skip_frame)
        self.cnn = CNN((None, skip_frame, 84, 84), num_actions)

        self.skip_frame = skip_frame
        self.discount = .95
        self.costList = list()
        self.state_tm1 = None

        if load is not None:
            self.cnn.load(load)

    def frames_processed(self, frames, action_performed, reward):
        game_action = self.action_handler.game_action_to_action_ind(action_performed)
        if self.state_tm1 is not None:
            self.exp_handler.add_sample(self.state_tm1, game_action, reward, False)

        # generate minibatch data
        if self.exp_handler.size > self.minimum_replay_size:
            states, actions, rewards, state_tp1s, terminal = self.exp_handler.random_batch(32)
            cost = self.cnn.train(states, actions, rewards, state_tp1s, terminal)
            self.costList.append(cost)
            self.action_handler.anneal()

        self.state_tm1 = frames[-1]

    def get_action(self, processed_screens):
        return self.cnn.get_output(processed_screens)[0]

    def game_over(self):
        self.exp_handler.add_terminal()  # adds a terminal

    def get_game_action(self):
        return self.action_handler.action_vect_to_game_action(
            self.get_action(self.exp_handler.phi(self.state_tm1).reshape(1, self.skip_frame, 84, 84)))

    def set_legal_actions(self, legal_actions):
        self.action_handler.set_legal_actions(legal_actions)

    def save(self, file):
        self.cnn.save(file)

    def get_cost_list(self):
        return self.costList


class DQNTester:
    def __init__(self, skip_frame, num_actions, load, rand_val=0.05):
        rand_vals = (rand_val, rand_val, 2)
        self.action_handler = ActionHandler(ActionPolicy.eGreedy, rand_vals)
        self.cnn = CNN((None, skip_frame, 84, 84), num_actions, 1)
        self.cnn.load(load)
        self.q_vals = list()
        self.skip_frame = skip_frame
        self.exp_handler = DataSet(84, 84, np.random.RandomState(), phi_length=skip_frame)
        self.skip_frame = skip_frame
        self.state_tm1 = np.zeros((84, 84), dtype=np.uint8)

    def get_game_action(self):
        q_vals = self.cnn.get_output(self.exp_handler.phi(self.state_tm1).reshape(1, self.skip_frame, 84, 84))[0]
        self.q_vals.append(q_vals)
        return self.action_handler.action_vect_to_game_action(q_vals)

    def frames_processed(self, frames, action_performed, reward):
        game_action = self.action_handler.game_action_to_action_ind(action_performed)
        self.exp_handler.add_sample(self.state_tm1, game_action, reward, False)
        self.state_tm1 = frames[-1]

    def set_legal_actions(self, legal_actions):
        self.action_handler.set_legal_actions(legal_actions)


