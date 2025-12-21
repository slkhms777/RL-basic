import numpy as np
from env import GridWorld

def random_argmax(arr):
    """随机打破平均的 argmax"""
    return np.random.choice(np.flatnonzero(arr == np.max(arr)))

def mc_explore_start(env: GridWorld, episodes=10000, max_episode_length=500):
    env.reset()
    # 为每个状态-动作对维护一个回报列表
    returns = np.empty((env.size, env.size, env.action_number), dtype=object)
    for i in range(env.size):
        for j in range(env.size):
            for a in range(env.action_number):
                returns[i, j, a] = []
    for ep in range(episodes):
        s0 = (np.random.randint(0, env.size), np.random.randint(0, env.size))
        a0 = np.random.randint(0, env.action_number)
        trajectory = [(s0, a0)]

        s_next, r = env.take_action(s0, a0)
        rewards = [r]

        while True:
            action_probs = env.pi[s_next]
            a_next = np.random.choice(np.arange(env.action_number), p=action_probs)

            trajectory.append((s_next, a_next))

            s, a = s_next, a_next
            s_next, r = env.take_action(s, a)
            rewards.append(r)

            if len(trajectory) > max_episode_length:
                break

        T = len(trajectory)
        G = 0

        first_visit_dict = {}
        for t in range(T):
            s_t, a_t = trajectory[t]
            if (s_t, a_t) not in first_visit_dict:
                first_visit_dict[(s_t, a_t)] = t

        for t in range(T-1, -1, -1):
            G = env.gamma * G + rewards[t]
            s_t, a_t = trajectory[t]

            if (first_visit_dict[(s_t, a_t)] == t):
                returns[s_t][a_t].append(G)
                env.action_value[s_t][a_t] = np.mean(returns[s_t][a_t])
        
        # 更新策略
        for s in [(i, j) for i in range(env.size) for j in range(env.size)]:
            best_a = random_argmax(env.action_value[s])
            # 更新 pi 为确定性策略 (one-hot)
            env.pi[s] = np.eye(env.action_number)[best_a]

    for s in [(i, j) for i in range(env.size) for j in range(env.size)]:
        env.pi_star[s] = random_argmax(env.action_value[s])
    
    print("Pi after MC with Explore Start:")
    print(env.pi_star)


def mc_epsilon_greedy(env: GridWorld, episodes=10000, max_episode_length=500, epsilon=0.1):
    env.reset()
    # 为每个状态-动作对维护一个回报列表
    returns = np.empty((env.size, env.size, env.action_number), dtype=object)
    for i in range(env.size):
        for j in range(env.size):
            for a in range(env.action_number):
                returns[i, j, a] = []
    for ep in range(episodes):
        s0 = (np.random.randint(0, env.size), np.random.randint(0, env.size))
        a0 = np.random.randint(0, env.action_number)
        trajectory = [(s0, a0)]

        s_next, r = env.take_action(s0, a0)
        rewards = [r]

        while True:
            action_probs = env.pi[s_next]
            a_next = np.random.choice(np.arange(env.action_number), p=action_probs)

            trajectory.append((s_next, a_next))

            s, a = s_next, a_next
            s_next, r = env.take_action(s, a)
            rewards.append(r)

            if len(trajectory) > max_episode_length:
                break

        T = len(trajectory)
        G = 0
        
        first_visit_dict = {}
        for t in range(T):
            s_t, a_t = trajectory[t]
            if (s_t, a_t) not in first_visit_dict:
                first_visit_dict[(s_t, a_t)] = t

        for t in range(T-1, -1, -1):
            G = env.gamma * G + rewards[t]
            s_t, a_t = trajectory[t]

            if first_visit_dict[(s_t, a_t)] == t:
                returns[s_t][a_t].append(G)
                env.action_value[s_t][a_t] = np.mean(returns[s_t][a_t])
        
        # 更新策略
        for s in [(i, j) for i in range(env.size) for j in range(env.size)]:
            best_a = random_argmax(env.action_value[s])
            for a in range(env.action_number):
                if a == best_a:
                    env.pi[s][a] = 1 - epsilon + (epsilon / env.action_number)
                else:
                    env.pi[s][a] = epsilon / env.action_number

    for s in [(i, j) for i in range(env.size) for j in range(env.size)]:
        env.pi_star[s] = random_argmax(env.action_value[s])

    print("Pi after MC with Epsilon-Greedy:")
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
    mc_explore_start(env)
    mc_epsilon_greedy(env)
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
    