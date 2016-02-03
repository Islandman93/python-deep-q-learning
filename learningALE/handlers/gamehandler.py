import numpy as np
from scipy.misc import imresize
from learningALE.libs.ale_python_interface import ALEInterface
__author__ = 'Ben'


class GameHandler:
    """
    The :class:`GameHandler` class takes care of the interface between the ALE and the learner.

    Currently supported is the ability to display the ALE screen when running, skip x number of frames by repeating the
    last action, and configuring the dtype to convert the gamescreen to (default is float16 for space).

    Parameters
    ----------
    rom : byte string
        Specifies the directory to load the rom from. Must be a byte string: b'dir_for_rom/rom.bin'
    show_rom : boolean
        Whether or not to show the game being played or not. True takes longer to run but can be fun to watch
    skip_frame : int
        Number of frames to skip using the last action chosen
    learner : :class:`learners.learner`
        Default None. The learner, on construction GameHandler will call set_legal_actions. If none then set_legal_actions
        needs to be called
    """
    def __init__(self, rom, show_rom, skip_frame, learner=None):
        # set up emulator
        self.ale = ALEInterface(show_rom)
        self.ale.loadROM(rom)
        (self.screen_width, self.screen_height) = self.ale.getScreenDims()
        legal_actions = self.ale.getMinimalActionSet()

        # set up vars
        self.skipFrame = skip_frame

        if learner:
            learner.set_legal_actions(legal_actions)

        self.total_frame_count = 0

    def run_one_game(self, learner, neg_reward=False, early_return=False, clip=True, max_episode_frame=np.inf):
        """
        Runs a game until ale.game_over()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvis true. Currently does not support stopping at a specific frame count during
        an episode

        Parameters
        ----------
        learner : :class:`learners.learner`
            Will call get_game_action and frames_processed.
            get_game_action must return a valid ALE action ind. frames_processed can be a pass.

        neg_reward : bool
            Default False. Whether or not to use negative rewards, recieved when agent looses a life.

        early_return : bool
            Default False. If set to true and neg_rewards is set then will return on first loss of life

        clip : bool
            Default True. Whether or not to clip positive rewards to 1

        max_episode_frame : int
            Default np.inf. The maximum number of frames to run per episode

        Returns
        -------
        int
            Total reward from game. Can be negative if neg_reward is true.
        """
        total_reward = 0.0
        gamescreen = None
        self.ale.reset_game()
        cur_lives = self.ale.lives()
        action_to_perform = 0  # initially set at zero because we start the game before asking the learner
        while not self.ale.game_over() and self.ale.getEpisodeFrameNumber() < max_episode_frame:
            # get frames
            frames = list()
            reward = 0

            # loop over skip frame
            for frame in range(self.skipFrame):
                gamescreen = self.ale.getScreenGrayscale(gamescreen)

                # convert ALE gamescreen into usable image, scaled between 0 and 1
                processedImg = imresize(gamescreen[33:-16, :, 0], 0.525, interp='nearest')
                frames.append(processedImg)

                # act on the action to perform, should be ALE compatible action ind
                rew = self.ale.act(action_to_perform)

                # clip positive rewards to 1
                if rew > 0 and clip:
                    reward += 1
                else:
                    reward += rew

                # if allowing negative rewards, see if lives has decreased
                if neg_reward:
                    new_lives = self.ale.lives()
                    if new_lives < cur_lives:
                        reward -= 1  # losing a life is a negative 1 reward
                        cur_lives = new_lives

            # end frame skip loop

            total_reward += reward
            frames = np.asarray(frames)

            # frames_processed must be here before action_to_perform gets overwritten.
            learner.frames_processed(frames, action_to_perform, reward)

            action_to_perform = learner.get_game_action()

            self.total_frame_count += 1 * self.skipFrame

            # if doing early return, end game on first loss of life
            if reward < 0 and early_return:
                return total_reward

        # end of game
        return total_reward

    def set_legal_actions(self, learner):
        learner.set_legal_actions(self.ale.getMinimalActionSet())