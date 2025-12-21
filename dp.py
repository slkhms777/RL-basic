from env import GridWorld
import numpy as np

def value_iteration(env: GridWorld, max_iter=20):
    env.reset()
    # initialize policy
    env.pi = np.ones((env.size, env.size, env.action_number), dtype=np.float32) / env.action_number
    
    i = 0
    while True:
        tmp_pi = np.zeros((env.size, env.size, env.action_number), dtype=np.float32)
        tmp_state_value = np.zeros((env.size, env.size), dtype=np.float32)
        for s in [(i, j) for i in range(env.size) for j in range(env.size)]:
            best_a = 0
            best_q = -float('inf')
            for a in range(env.action_number):
                s2, r = env.take_action(s, a)
                q_value = r + env.gamma * env.state_value[s2]
                if q_value >= best_q:
                    best_q = q_value
                    best_a = a
            # policy update
            tmp_pi[s] = np.eye(env.action_number)[best_a]
            # value update
            tmp_state_value[s] = best_q

        delta = np.max(np.abs(env.state_value - tmp_state_value))

        env.pi = tmp_pi
        env.state_value = tmp_state_value

        if delta < 1e-4 or i >= max_iter:
            break
        else:
            i += 1
    print("Final Results after Value Iteration")
    print(f"state_value :\n {env.state_value}")
            
def policy_iteration(env: GridWorld, max_iter=250):
    env.reset()
    # initialize policy
    env.pi = np.ones((env.size, env.size, env.action_number), dtype=np.float32) / env.action_number

    i = 0
    while True:
        # policy evaluation
        while True:
            # print(f"  Evaluation Iteration: {j}")
            max_delta = 0
            tmp_state_value = env.state_value.copy()
            for s in [(i, j) for i in range(env.size) for j in range(env.size)]:
                v =0
                for a in range(env.action_number):
                    s2, r = env.take_action(s, a)
                    v += env.pi[s][a] * (r + env.gamma * env.state_value[s2])
                delta = np.abs(tmp_state_value[s] - v)
                if delta > max_delta:
                    max_delta = delta
                tmp_state_value[s] = v
            if max_delta < 10: # max_delta 越小，收敛越慢
                env.state_value = tmp_state_value
                break
        # policy improvement
        for s in [(i, j) for i in range(env.size) for j in range(env.size)]:
            best_a = 0
            best_q = -float('inf')
            for a in range(env.action_number):
                s2, r = env.take_action(s, a)
                q_value = r + env.gamma * env.state_value[s2]
                if q_value > best_q:
                    best_q = q_value
                    best_a = a
            # policy update
            env.pi[s] = np.eye(env.action_number)[best_a]
        if i >= max_iter:
            break
        else:
            i += 1
    print("Final Results after Policy Iteration")
    print(f"state_value :\n {env.state_value}")
        
def truncated_policy_iteration(env: GridWorld, max_iter=250, max_eval_iter=10):
    # initialize policy
    env.pi = np.ones((env.size, env.size, env.action_number), dtype=np.float32) / env.action_number
    
    i = 0
    while True:
        # policy evaluation
        j = 0
        while True:
            # print(f"  Evaluation Iteration: {j}")
            max_delta = 0
            tmp_state_value = env.state_value.copy()
            for s in [(i, j) for i in range(env.size) for j in range(env.size)]:
                v =0
                for a in range(env.action_number):
                    s2, r = env.take_action(s, a)
                    v += env.pi[s][a] * (r + env.gamma * env.state_value[s2])
                delta = np.abs(tmp_state_value[s] - v)
                if delta > max_delta:
                    max_delta = delta
                tmp_state_value[s] = v
            if max_delta < 1e-4 or j >= max_eval_iter:
                env.state_value = tmp_state_value
                break
            else:
                j += 1
        # policy improvement
        for s in [(i, j) for i in range(env.size) for j in range(env.size)]:
            best_a = 0
            best_q = -float('inf')
            for a in range(env.action_number):
                s2, r = env.take_action(s, a)
                q_value = r + env.gamma * env.state_value[s2]
                if q_value > best_q:
                    best_q = q_value
                    best_a = a
            # policy update
            env.pi[s] = np.eye(env.action_number)[best_a]
        if i >= max_iter:
            break
        else:
            i += 1
    print("Final Results after Truncated Policy Iteration")
    print(f"state_value :\n {env.state_value}")

if __name__ == "__main__":
    size = 2
    forbidden_states = [(0, 1)]
    target_state = (1, 1)
    gamma = 0.9
    forbidden_penalty = -1
    boundary_penalty = -1
    target_reward = 1
    env1 = GridWorld(
        size=size,
        forbidden_states=forbidden_states,
        target_state=target_state,
        gamma=gamma,
        forbidden_penalty=forbidden_penalty,
        boundary_penalty=boundary_penalty,
        target_reward=target_reward
    )
    value_iteration(env1)

    size = 5
    forbidden_states = [(1,1),(1, 2),(2,2),(3,1),(3,3),(4,1)]
    target_state = (3,2)
    gamma = 0.9
    forbidden_penalty = -10
    boundary_penalty = -1
    target_reward = 1
    env2 = GridWorld(
        size=size,
        forbidden_states=forbidden_states,
        target_state=target_state,
        gamma=gamma,
        forbidden_penalty=forbidden_penalty,
        boundary_penalty=boundary_penalty,
        target_reward=target_reward
    )
    policy_iteration(env2)
    truncated_policy_iteration(env2)