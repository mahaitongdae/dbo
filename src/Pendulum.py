from envs.classic_pendulum import SafeClassicPendulum
import numpy as np

class InvertedPendulum:
    def __init__(self):
        self.env = SafeClassicPendulum(init_state=[0.3, -0.9], threshold=0.1)
        self.obs = self.env.reset()
        self.domain = np.array([[-10, 10], [-10, 10]])

        self.min = 0
        self.arg_min = np.array([[0, 0]])

    def function(self, k):
        done = False
        objective_function = 0
        steps = 0
        while steps <= 200:
            self.obs, rew, done, info = self.env.step([k[0]*self.obs[0] + k[1]*self.obs[1]])
            objective_function += rew
            steps += 1
        return objective_function

if __name__ == '__main__':
    task = InvertedPendulum()
    print(task.function([-10, -10]))