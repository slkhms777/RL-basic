import torch 
from torch import nn
from env import GridWorld
import collections
import random
import numpy as np
from tqdm import tqdm
import random
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


# 初始化环境
env = GridWorld(
    size = 5,
    forbidden_states = [(1,1),(1, 2),(2,2),(3,1),(3,3),(4,1)],
    target_state = (3,2),
    gamma = 0.9,
    forbidden_penalty = -10,
    boundary_penalty = -1,
    target_reward = 1,
)

# DQN参数
batch_size = 128
gamma = 0.9
lr = 0.0001
state_dim = 2  # 状态维度
action_dim = env.action_number  # 动作维度
hidden_dim = 256

target_update_freq = 1000
replay_buffer_size = 20000
min_replay_size = 2000

max_episodes = 20
max_steps = 50000

# 选择device，优先选择cuda，其次是mps，最后是cpu
device =torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 主网络和目标网络
main_net = Q_Net(state_dim, hidden_dim, action_dim).to(device)            
target_net = Q_Net(state_dim, hidden_dim, action_dim).to(device)             

# 优化器
optimizer = torch.optim.Adam(main_net.parameters(), lr=lr)
# replay_buffer
replay_buffer = collections.deque(maxlen=replay_buffer_size)

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995

epsilon = 0.1
step_counter = 0

GT_q = [
    [ 3.5, 3.9, 4.3, 4.8, 5.3],
    [ 3.1, 3.5, 4.8, 5.3, 5.9],
    [ 2.8, 2.5, 10., 5.9, 6.6],
    [ 2.5, 10.0, 10.0, 10.0, 7.3],
    [ 2.3, 9.0, 10.0, 9.0, 8.1]
]
GT_q = torch.tensor(GT_q, dtype=torch.float32).to(device)
GT_pi_star = [
    [1,1,1,1,2],
    [0,0,1,2,2],
    [0,3,2,1,2],
    [0,1,4,3,2],
    [0,1,0,3,3],
]
GT_pi_star = np.array(GT_pi_star)
loss_list = []
diff_list = []

# 创建保存目录
os.makedirs("vis/dqn", exist_ok=True)

# DQN步骤
for episode in tqdm(range(max_episodes), desc="Training"):
    state = (random.randint(0, env.size - 1), random.randint(0, env.size - 1))
    total_reward = 0
    epsilon = EPSILON_START

    for step in range(max_steps):
        # ε-greedy
        if random.random() < epsilon:
            action = random.randint(0, 4)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor([state], dtype=torch.float32).to(device)
                q_values = main_net(state_tensor)
                action = q_values.argmax().item()

        # 执行动作，获得下一步状态和奖励
        next_state, reward = env.take_action(state, action)
        total_reward += reward

        # 存入经验池
        replay_buffer.append((state, action, reward, next_state))

        # 更新状态
        state = next_state

        # 训练(buffer达到一定大小再开始)
        if len(replay_buffer) >= min_replay_size:
            # 随机采样一个batch
            batch_data = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states = zip(*batch_data)

            # 张量化
            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.int64).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(device)

            # 计算当前 Q 
            current_q_values = main_net(states).gather(1, actions.unsqueeze(1)).squeeze(1) # [batch_size, ]

            # 计算目标 Q
            with torch.no_grad():
                next_q_values, _ = target_net(next_states).max(1)
                target_q_values = rewards + gamma * next_q_values
            
            # 计算损失
            loss_func = nn.MSELoss()
            loss = loss_func(current_q_values, target_q_values)
            # 记录loss
            loss_list.append(loss.item())

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新计步器，更新网络参数
            step_counter += 1
            if step_counter % target_update_freq == 0:
                target_net.load_state_dict(main_net.state_dict())

    # 衰减 epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    with torch.no_grad():
        state_grid = torch.tensor([[(i, j) for j in range(env.size)] for i in range(env.size)], dtype=torch.float32).to(device)
        q_grid = main_net(state_grid.view(-1, 2)).view(env.size, env.size, -1)
        v_grid, _ = q_grid.max(dim=2)
        diff = torch.abs(v_grid - GT_q).mean().item()
        diff_list.append(diff)
        print(f"\nEpisode {episode}, Mean V value difference from GT: {diff:.4f}")

# 绘制loss曲线
plt.figure()
plt.plot(loss_list)
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("DQN Loss Curve")
plt.savefig("vis/dqn/loss_curve.png")
plt.close()

# 绘制GT difference曲线
plt.figure()
plt.plot(diff_list)
plt.xlabel("Episode")
plt.ylabel("Mean V Value Difference from GT")
plt.title("DQN V Value Difference Curve")
plt.savefig("vis/dqn/gt_diff_curve.png")
plt.close()

pi_star = np.zeros((env.size, env.size), dtype=int)
for i in range(env.size):
    for j in range(env.size):
        state_tensor = torch.tensor([[i, j]], dtype=torch.float32).to(device)
        with torch.no_grad():
            q_vals = main_net(state_tensor)
            pi_star[i, j] = q_vals.argmax().item()

print("Learned optimal policy (action indices):")
print(pi_star)
print(GT_pi_star)


class DQN:
    def __init__(self):
        pass

    def train(self):
        pass

if __name__ == "__main__":
    pass