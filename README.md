# 强化学习基础算法复现

本项目复现了强化学习中的经典基础算法

## 目录结构

- `env.py`：环境定义（格子世界 GridWorld）
- `dp.py`：动态规划（值迭代、策略迭代等）
- `mc.py`：蒙特卡洛方法
- `td.py`：时序差分方法（SARSA、Q-Learning 等）
- `requirement.txt`：依赖包列表

## 环境说明

本项目以格子世界（GridWorld）为例，支持自定义障碍、目标状态、奖励与惩罚等。可用于对比不同 RL 算法的收敛效果和策略表现。

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
- Sutton & Barto, Reinforcement Learning: An Introduction

