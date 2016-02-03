import time
import matplotlib.pyplot as plt
import numpy as np
from learningALE.handlers.gamehandler import GameHandler
from learningALE.learners.DQN import DQNLearner


# setup vars
rom = b'D:\\_code\\breakout.bin'
gamename = 'breakout'
skip_frame = 4
num_actions = 4
learner = DQNLearner(skip_frame, num_actions)
epochs = 100

ep_count = 0
game_handler = GameHandler(rom, False, skip_frame, learner)
scoreList = list()
bestTotReward = -np.inf
plt.ion()
st = time.time()
while game_handler.total_frame_count/skip_frame < epochs*50000:
    total_reward = game_handler.run_one_game(learner)
    scoreList.append(total_reward)

    learner.game_over()

    # if this is the best score save it as such
    if total_reward >= bestTotReward:
        learner.save('dqnbest{0}.pkl'.format(total_reward))
        bestTotReward = total_reward

    # save params every 25000 updates (half an epoch)
    if (game_handler.total_frame_count/skip_frame) >= ep_count * 50000:
        ep_count += 0.5
        # plot cost and score
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(learner.get_cost_list(), '.')
        plt.subplot(1, 2, 2)
        sl = np.asarray(scoreList)
        plt.plot(sl, '.')
        plt.pause(0.0001)
        learner.save('dqn{0}.pkl'.format(game_handler.total_frame_count / skip_frame))

    et = time.time()
    print("Episode ended with score: " + str(total_reward))
    print('Total Time:', et - st, 'Updates:', game_handler.total_frame_count / skip_frame,
          'UPS:', (game_handler.total_frame_count / skip_frame) / (et - st),
          'Frame Count:', game_handler.total_frame_count, 'FPS:', game_handler.total_frame_count / (et - st))

plt.ioff()
plt.show()

# final save
learner.save('dqn{0}.pkl'.format(game_handler.total_frame_count / skip_frame))
