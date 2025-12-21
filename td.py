import numpy as np
from env import GridWorld

def random_argmax(arr):
    """随机打破平均的 argmax"""
    return np.random.choice(np.flatnonzero(arr == np.max(arr)))

def sarsa(env: GridWorld, episodes=30000, max_episode_length=50, alpha=0.05, epsilon=0.05):
    env.reset()
    for ep in range(episodes):
        s = (np.random.randint(0, env.size), np.random.randint(0, env.size))
        # ε-greedy 选择动作
        action_probs = env.pi[s]
        a = np.random.choice(np.arange(env.action_number), p=action_probs)

        for t in range(max_episode_length):
            s2, r = env.take_action(s, a)

            # ε-greedy 选择下一个动作
            action_probs = env.pi[s2]
            a2 = np.random.choice(np.arange(env.action_number), p=action_probs)

            # 更新 action-value
            env.action_value[s][a] = env.action_value[s][a] + alpha * (r + env.gamma * env.action_value[s2][a2] - env.action_value[s][a])
            # 更新策略
            best_a = random_argmax(env.action_value[s])
            env.pi[s] = np.ones(env.action_number) * (epsilon / env.action_number)
            env.pi[s][best_a] += 1 - epsilon

            s, a = s2, a2

    for s in [(i, j) for i in range(env.size) for j in range(env.size)]:
        env.pi_star[s] = np.argmax(env.action_value[s])

    print("Pi after SARSA:")
    print(env.pi_star)

def q_learning(env: GridWorld, episodes=30000, max_episode_length=50, alpha=0.05, epsilon=0.05):
    env.reset()
    for ep in range(episodes):
        s = (np.random.randint(0, env.size), np.random.randint(0, env.size))

        for t in range(max_episode_length):

            # ε-greedy 选择动作
            action_probs = env.pi[s]
            a = np.random.choice(np.arange(env.action_number), p=action_probs)

            s2, r = env.take_action(s, a)

            # Q-learning 更新 action-value
            best_a2 = random_argmax(env.action_value[s2])
            env.action_value[s][a] = env.action_value[s][a] + alpha * (r + env.gamma * env.action_value[s2][best_a2] - env.action_value[s][a])

            # 更新策略
            best_a = random_argmax(env.action_value[s])
            env.pi[s] = np.ones(env.action_number) * (epsilon / env.action_number)
            env.pi[s][best_a] += 1 - epsilon

            s = s2

    for s in [(i, j) for i in range(env.size) for j in range(env.size)]:
        env.pi_star[s] = np.argmax(env.action_value[s])

    print("Pi after Q-Learning:")
    print(env.pi_star)


if __name__ == "__main__":
    size = 5
    forbidden_states = [(1,1),(1, 2),(2,2),(3,1),(3,3),(4,1)]
    target_state = (3,2)
    gamma = 0.9
    forbidden_penalty = -10
    boundary_penalty = -1
    target_reward = 1
    env = GridWorld(
        size=size,
        forbidden_states=forbidden_states,
        target_state=target_state,
        gamma=gamma,
        forbidden_penalty=forbidden_penalty,
        boundary_penalty=boundary_penalty,
        target_reward=target_reward
    )
    sarsa(env)
    q_learning(env)

    print("pi_star_GT:")
    GT = [
        [1,1,1,1,2],
        [0,0,1,2,2],
        [0,3,2,1,2],
        [0,1,4,3,2],
        [0,1,0,3,3],
    ]
    GT = np.array(GT)
    print(GT)
    