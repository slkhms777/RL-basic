import torch
import numpy as np
import gymnasium as gym  
import matplotlib.pyplot as plt
import torch.nn.functional as F
import rl_utils
from tqdm import tqdm


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class GRPO:
    '''
    GRPO (Group Relative Policy Optimization) 算法
    核心思想：不使用critic网络，而是通过采样一组动作，计算组内相对优势来估计优势
    '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr,
                 lmbda, epochs, eps, gamma, device, group_size=8):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.group_size = group_size  # GRPO中采样的组大小
    
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def compute_grpo_advantage(self, states, actions, rewards, next_states, dones):
        '''
        GRPO优势：对于每个状态，采样一组动作，计算组内相对优势
        这里简化为使用蒙特卡洛回报来计算优势
        '''
        # 计算折扣回报
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float).to(self.device)
        
        # 归一化回报作为优势估计（GRPO的核心：使用组内相对优势）
        advantage = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return advantage.view(-1, 1)
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 计算GRPO优势（不使用critic，直接使用归一化回报）
        advantage = self.compute_grpo_advantage(states, actions, rewards, next_states, dones)
        
        # 计算旧策略的对数概率
        with torch.no_grad():
            old_probs = self.actor(states).gather(1, actions)
            old_log_probs = torch.log(old_probs + 1e-8)

        # PPO-CLIP更新
        for _ in range(self.epochs):
            probs = self.actor(states).gather(1, actions)
            log_probs = torch.log(probs + 1e-8)
            
            # 计算概率比率
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            
            # GRPO损失（注意这里取负数，因为我们要最大化目标函数）
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            
            # 添加KL散度惩罚（可选，用于约束策略更新幅度）
            # kl_penalty = 0.01 * torch.mean((ratio - 1) ** 2)
            # actor_loss = actor_loss + kl_penalty
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


def train(agent, env, num_episodes=500):
    """training grpo agent (on-policy)"""
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, info = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('GRPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('GRPO on {}'.format(env_name))
    plt.show()


if __name__ == "__main__":
    env_name = "CartPole-v1"
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    obs, info = env.reset(seed=0)
    env.action_space.seed(0)
    torch.manual_seed(0)

    ## 超参数设置
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    actor_lr = 1e-3
    group_size = 8  # GRPO组大小
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "mps") if torch.backends.mps.is_available() else torch.device("cpu")    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = GRPO(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim,
                 actor_lr=actor_lr, lmbda=lmbda, epochs=epochs,
                 eps=eps, gamma=gamma, device=device, group_size=group_size)

    train(agent=agent, env=env, num_episodes=num_episodes)
