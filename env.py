import numpy as np
#########
# |--------> y
# |
# |
# |
# v
# x
#########
class GridWorld():
    def __init__(self, size=5, forbidden_states=None, target_state=None,
                  gamma=0.9, boundary_penalty=-1, forbidden_penalty=-1, target_reward=1):
        self.size = size
        self.grid = np.zeros((self.size, self.size))

        if forbidden_states is not None:
            self.forbidden_states = forbidden_states
        else:
            print("No forbidden states set!")
        
        if target_state is not None:
            self.target_state = target_state
        else:
            print("No target state set!")
            
        self.gamma = gamma

        self.boundary_penalty = boundary_penalty
        self.forbidden_penalty = forbidden_penalty
        self.target_reward = target_reward

        self.action_number = 5

        self.action_value = np.zeros((self.size, self.size, self.action_number), dtype=np.float64)
        self.state_value = np.zeros((self.size, self.size), dtype=np.float64)
        # 策略
        self.pi = np.zeros((self.size, self.size, self.action_number), dtype=np.float64)
        # 最优策略
        self.pi_star = np.zeros((self.size, self.size), dtype=np.float64)

        self.reset()

    def reset(self):
        self.action_value = np.zeros((self.size, self.size, self.action_number), dtype=np.float64)
        self.state_value = np.zeros((self.size, self.size), dtype=np.float64)
        self.pi = np.ones((self.size, self.size, self.action_number), dtype=np.float64) / self.action_number
        self.pi_star = np.zeros((self.size, self.size), dtype=np.float64)

    def action_space(self):
        return ['up', 'right', 'down', 'left', 'stay']

    def not_in_boundary(self, s):
        return 0 <= s[0] < self.size and 0 <= s[1] < self.size

    def take_action(self, s1, a):
        r = 0

        if a == 0:  # up
            s2 = (s1[0] - 1, s1[1])
        if a == 1:  # right 
            s2 = (s1[0], s1[1] + 1)
        if a == 2:  # down
            s2 = (s1[0] + 1, s1[1])
        if a == 3:  # left
            s2 = (s1[0], s1[1] - 1)
        if a == 4:  # stay
            s2 = s1

        # 边界检查
        if not self.not_in_boundary(s2):
            s2 = s1
            r += self.boundary_penalty
        else:
            # 障碍检查
            if s2 in self.forbidden_states:
                r += self.forbidden_penalty

            if s2 == self.target_state:
                r += self.target_reward
        
        return s2, r