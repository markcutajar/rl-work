import gym
import numpy as np
from tqdm import tqdm
from algorithms.utils import plot_learning_curve


def epsilon(ep,
            max_episodes,
            max_epsilon=0.3,
            min_epsilon=0.01):

    ratio = ((max_epsilon-min_epsilon) / (
        1 + np.exp((-max_episodes/2.0 + ep)*(10/max_episodes))
    ))

    calculated = min_epsilon + ratio
    return calculated


def choose_action(state, episode, max_episodes):
    if np.random.uniform(0, 1) < epsilon(episode, max_episodes):
        return env.action_space.sample()
    return np.argmax(Q[state, :])


def learn(s, sn, r, a, gamma=0.96, lr_rate=0.7):
    """
    Q Learning
    Q(s,a) <- Q(s,a) + lr(r + gamma * max(Q[sn, :])) - Q(s, a)
    """
    predict = Q[s, a]
    target = r + gamma * np.max(Q[sn, :])
    Q[s, a] = Q[s, a] + lr_rate * (target - predict)


def shape_reward(s, d, r, goal=100, hole=-100):
    if d and r < 1.0:
        return hole
    elif d and r > 0.0:
        return goal
    return -1


if __name__ == '__main__':

    env = gym.make('FrozenLake8x8-v0', is_slippery=True)
    total_episodes = 200000
    max_steps = 400

    # positive_rewards = np.linspace(100, 1000, num=3)
    # negative_rewards = np.linspace(-100, -1000, num=3)
    # rewards_combinations = []
    #
    # for p in positive_rewards:
    #     for n in negative_rewards:
    #         rewards_combinations.append((p, n))

    # for (pos, neg) in tqdm([(100, -100)]):

    positive_reward = 100
    negative_reward = -100
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    total_steps = []

    for episode in tqdm(range(total_episodes)):

        steps_taken = 1
        observation = env.reset()

        while steps_taken <= max_steps:

            action = choose_action(observation, episode, total_episodes)
            observation_, reward, done, info = env.step(action)
            shaped_reward = shape_reward(steps_taken, done, reward, goal=positive_reward, hole=negative_reward)
            learn(observation, observation_, shaped_reward, action)

            observation = observation_

            if done or steps_taken == max_steps:
                total_steps.append(reward)
                break

            steps_taken += 1

    plot_learning_curve(
        [i+1 for i in range(len(total_steps))],
        total_steps,
        'plots/frozen_lake_slippery_QL_LUT.png',
        name=f'Pos: {positive_reward} Neg: {negative_reward}'
    )

    actions = np.argmax(Q, axis=1)
    values = np.max(Q, axis=1)

    def create_action(a_):
        if a_ == 0:
            return 'LEFT'
        elif a_ == 1:
            return 'DOWN'
        elif a_ == 2:
            return 'RIGHT'
        elif a_ == 3:
            return 'UP'
        else:
            raise ValueError('Not recognized action')

    value_actions = []
    for v, a in zip(values, actions):
        value_actions.append(create_action(a))
    value_actions = np.array(value_actions)
    value_actions = np.reshape(value_actions, (8, 8))

    for i in value_actions:
        for j in i:
            print(j, end='\t')
        print('\n')

    print(f'Average reward: {np.mean(total_steps[-100])}')

    import pdb; pdb.set_trace()
    print('here')


