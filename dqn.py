import torch 
from torch import nn
from env import GridWorld
import collections
import random
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class Q_Net(nn.Module):
    # main network 和 target network 的网络结构相同
    def __init__(self, state_dim=2, hidden_dim=64, action_dim=5):
        super(Q_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        # state: (batch_size, 2) -> 输出 (batch_size, 5)
        h = self.relu1(self.fc1(x))
        h = self.relu2(self.fc2(h))
        out = self.fc3(h)
        return out # q_value


class DQN:
    def __init__(self, env, state_dim=2, hidden_dim=256, action_dim=5,
                 batch_size=128, gamma=0.9, lr=0.0001,
                 target_update_freq=1000, replay_buffer_size=20000, min_replay_size=2000,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 device=None):
        """
        DQN算法类
        
        Args:
            env: 环境对象
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
            action_dim: 动作维度
            batch_size: 批次大小
            gamma: 折扣因子
            lr: 学习率
            target_update_freq: 目标网络更新频率
            replay_buffer_size: 经验回放池大小
            min_replay_size: 开始训练的最小经验池大小
            epsilon_start: epsilon初始值
            epsilon_end: epsilon最终值
            epsilon_decay: epsilon衰减率
            device: 计算设备
        """
        self.env = env
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.target_update_freq = target_update_freq
        self.replay_buffer_size = replay_buffer_size
        self.min_replay_size = min_replay_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 选择device，优先选择cuda，其次是mps，最后是cpu
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() 
                                      else "mps" if torch.backends.mps.is_available() 
                                      else "cpu")
        else:
            self.device = device
        
        # 主网络和目标网络
        self.main_net = Q_Net(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_net = Q_Net(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=lr)
        
        # replay_buffer
        self.replay_buffer = collections.deque(maxlen=replay_buffer_size)
        
        # 训练状态
        self.epsilon = epsilon_start
        self.step_counter = 0
        
        # 记录训练过程
        self.loss_list = []
        self.diff_list = []
        
    def select_action(self, state, epsilon=None):
        """
        使用ε-greedy策略选择动作
        
        Args:
            state: 当前状态
            epsilon: epsilon值，如果为None则使用self.epsilon
            
        Returns:
            选择的动作
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if random.random() < epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
                q_values = self.main_net(state_tensor)
                action = q_values.argmax().item()
        return action
    
    def store_transition(self, state, action, reward, next_state):
        """存储经验到回放池"""
        self.replay_buffer.append((state, action, reward, next_state))
    
    def update(self):
        """
        从经验池采样并更新网络
        
        Returns:
            损失值，如果经验池不足则返回None
        """
        if len(self.replay_buffer) < self.min_replay_size:
            return None
        
        # 随机采样一个batch
        batch_data = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch_data)
        
        # 张量化
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        
        # 计算当前 Q 
        current_q_values = self.main_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标 Q
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_states).max(1)
            target_q_values = rewards + self.gamma * next_q_values
        
        # 计算损失
        loss_func = nn.MSELoss()
        loss = loss_func(current_q_values, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新计步器，更新网络参数
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
        
        return loss.item()
    
    def train(self, max_episodes=20, max_steps=50000, GT_q=None, GT_pi_star=None, save_dir="vis/dqn"):
        """
        训练DQN
        
        Args:
            max_episodes: 最大训练轮数
            max_steps: 每轮最大步数
            GT_q: 真实Q值（用于评估）
            GT_pi_star: 真实最优策略（用于评估）
            save_dir: 保存目录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 将GT_q转换为tensor
        if GT_q is not None:
            GT_q = torch.tensor(GT_q, dtype=torch.float32).to(self.device)
        
        for episode in tqdm(range(max_episodes), desc="Training"):
            state = (random.randint(0, self.env.size - 1), random.randint(0, self.env.size - 1))
            total_reward = 0
            
            for step in range(max_steps):
                # 选择动作
                action = self.select_action(state)
                
                # 执行动作，获得下一步状态和奖励
                next_state, reward = self.env.take_action(state, action)
                total_reward += reward
                
                # 存入经验池
                self.store_transition(state, action, reward, next_state)
                
                # 更新状态
                state = next_state
                
                # 训练
                loss = self.update()
                if loss is not None:
                    self.loss_list.append(loss)
            
            # 衰减 epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # 评估（如果提供了GT_q）
            if GT_q is not None:
                with torch.no_grad():
                    state_grid = torch.tensor([[(i, j) for j in range(self.env.size)] 
                                               for i in range(self.env.size)], 
                                              dtype=torch.float32).to(self.device)
                    q_grid = self.main_net(state_grid.view(-1, 2)).view(self.env.size, self.env.size, -1)
                    v_grid, _ = q_grid.max(dim=2)
                    diff = torch.abs(v_grid - GT_q).mean().item()
                    self.diff_list.append(diff)
                    print(f"\nEpisode {episode}, Mean V value difference from GT: {diff:.4f}")
        
        # 绘制曲线
        self.plot_training_curves(save_dir)
        
        # 获取学到的策略
        pi_star = self.get_policy()
        print("Learned optimal policy (action indices):")
        print(pi_star)
        if GT_pi_star is not None:
            print("Ground truth optimal policy:")
            print(GT_pi_star)
        
        return pi_star
    
    def get_policy(self):
        """
        获取当前学到的策略
        
        Returns:
            策略矩阵
        """
        pi_star = np.zeros((self.env.size, self.env.size), dtype=int)
        for i in range(self.env.size):
            for j in range(self.env.size):
                state_tensor = torch.tensor([[i, j]], dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    q_vals = self.main_net(state_tensor)
                    pi_star[i, j] = q_vals.argmax().item()
        return pi_star
    
    def plot_training_curves(self, save_dir="vis/dqn"):
        """
        绘制训练曲线
        
        Args:
            save_dir: 保存目录
        """
        # 绘制loss曲线
        if self.loss_list:
            plt.figure()
            plt.plot(self.loss_list)
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("DQN Loss Curve")
            plt.savefig(f"{save_dir}/loss_curve.png")
            plt.close()
        
        # 绘制GT difference曲线
        if self.diff_list:
            plt.figure()
            plt.plot(self.diff_list)
            plt.xlabel("Episode")
            plt.ylabel("Mean V Value Difference from GT")
            plt.title("DQN V Value Difference Curve")
            plt.savefig(f"{save_dir}/gt_diff_curve.png")
            plt.close()
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'main_net': self.main_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.main_net.load_state_dict(checkpoint['main_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


if __name__ == "__main__":
    # 初始化环境
    env = GridWorld(
        size=5,
        forbidden_states=[(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)],
        target_state=(3, 2),
        gamma=0.9,
        forbidden_penalty=-10,
        boundary_penalty=-1,
        target_reward=1,
    )
    
    # Ground truth Q值和策略
    GT_q = [
        [3.5, 3.9, 4.3, 4.8, 5.3],
        [3.1, 3.5, 4.8, 5.3, 5.9],
        [2.8, 2.5, 10., 5.9, 6.6],
        [2.5, 10.0, 10.0, 10.0, 7.3],
        [2.3, 9.0, 10.0, 9.0, 8.1]
    ]
    
    GT_pi_star = [
        [1, 1, 1, 1, 2],
        [0, 0, 1, 2, 2],
        [0, 3, 2, 1, 2],
        [0, 1, 4, 3, 2],
        [0, 1, 0, 3, 3],
    ]
    GT_pi_star = np.array(GT_pi_star)
    
    # 创建DQN智能体
    dqn_agent = DQN(
        env=env,
        state_dim=2,
        hidden_dim=256,
        action_dim=env.action_number,
        batch_size=128,
        gamma=0.9,
        lr=0.0001,
        target_update_freq=1000,
        replay_buffer_size=20000,
        min_replay_size=2000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995
    )
    
    # 训练
    learned_policy = dqn_agent.train(
        max_episodes=20,
        max_steps=50000,
        GT_q=GT_q,
        GT_pi_star=GT_pi_star,
        save_dir="vis/dqn"
    )
    
    # 保存模型
    dqn_agent.save_model("vis/dqn/model.pth")