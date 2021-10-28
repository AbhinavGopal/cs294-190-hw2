"""
Implementation of TAMER (Knox + Stone, 2009)
When training, use 'W' and 'A' keys for positive and negative rewards
"""

import asyncio
import gym

from tamer.agent import Tamer
import argparse


async def main(epsilon, gammas, epsilon_ts):
    env = gym.make('MountainCar-v0')

    # hyperparameters
    discount_factor = 1
    epsilon = 0  # vanilla Q learning actually works well with no random exploration
    min_eps = 0
    num_episodes = 2
    tame = True  # set to false for vanilla Q learning

    # set a timestep for training TAMER
    # the more time per step, the easier for the human
    # but the longer it takes to train (in real time)
    # 0.2 seconds is fast but doable
    tamer_training_timestep = 0.3
    gammas = [float(gamma) for gamma in gammas]

    agent = Tamer(env, num_episodes, discount_factor, min_eps, tame,
                  tamer_training_timestep, model_file_to_load=None, epsilon=epsilon, 
                  gammas=gammas, epsilon_ts=epsilon_ts)

    await agent.train(model_file_to_save='autosave')
    agent.play(n_episodes=1, render=True)
    agent.evaluate(n_episodes=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, default=0)
    # parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--gammas', nargs='+', help='gamma values for sgdfnapproximator', default=[5.0,2.0,1.0,0.5])
    parser.add_argument('--epsilon_ts', action='store_true', default=False)
    args = parser.parse_args()
    asyncio.run(main(args.epsilon, args.gammas, args.epsilon_ts))

    # New stuff for Offline RL
    args = parser.parse_args()




