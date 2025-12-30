# 强化学习基础算法复现

本项目复现了强化学习中的经典基础算法

## 目录结构

- `env.py`：环境定义（格子世界 GridWorld）
- `dp.py`：动态规划（值迭代、策略迭代等）
- `mc.py`：蒙特卡洛方法
- `td.py`：时序差分方法（SARSA、Q-Learning 等）
- `dqn.py`：深度 Q-learning 网络（DQN）
- `reinforce.py`：策略梯度方法
- `actor_critic.py`：演员-评论家方法


## 环境说明

前5个算法使用《强化学习中的数学原理》中的网格世界环境, 后2个算法使用 OpenAI Gym 中的 CartPole 环境


## 快速开始

```bash
pip install -r requirement.txt
```

以动态规划算法为例：

```bash
python dp.py
```

## 参考资料
- [《强化学习中的数学原理》](https://github.com/MathFoundationRL/Book-Mathmatical-Foundation-of-Reinforcement-Learning)
- [《动手学强化学习》](https://hrl.boyuai.com)
- [OpenAI Gym](https://gym.openai.com/)


## 致谢
- 感谢 [@haukzero](https://github.com/haukzero) 提供的改进建议。
