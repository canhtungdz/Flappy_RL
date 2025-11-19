# agent/random_agent.py
import numpy as np

class RandomAgent:
    def __init__(self, flap_prob=0.12):
        """
        flap_prob: xác suất nhảy mỗi frame (0.12 = 12%)
        """
        self.flap_prob = flap_prob

    def select_action(self, state):
        """
        state: np.array([dx, dy, v])
        return: 0 (không nhảy) hoặc 1 (nhảy)
        """
        # Ở bước test này, mình chưa dùng state, chỉ random để xem agent đã điều khiển được chưa
        if np.random.rand() < self.flap_prob:
            return 1
        return 0
