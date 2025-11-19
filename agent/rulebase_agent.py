from typing import Dict, Any
import numpy as np


class RuleBaseAgent:
    """
    Agent đơn giản sử dụng rule-based để chơi Flappy Bird.
    
    State:
    - dx: khoảng cách ngang đến tâm lỗ (dương: chưa đến, âm: đã qua)
    - dy: khoảng cách dọc đến tâm lỗ (dương: bird trên tâm, âm: bird dưới tâm)
    - v: vận tốc bird (dương: rơi xuống, âm: bay lên), từ -10 đến 10
    
    Quy tắc:
    - Nếu bird ở dưới tâm lỗ (dy < 0) -> flap
    - Nếu bird đang rơi quá nhanh -> flap
    """
    
    def __init__(self, dy_threshold: float = -120, velocity_threshold: float = 7):
        """
        Args:
            dy_threshold: Ngưỡng dy để quyết định flap (bird dưới tâm bao nhiêu thì flap)
            velocity_threshold: Ngưỡng vận tốc rơi để flap
        """
        self.dy_threshold = dy_threshold
        self.velocity_threshold = velocity_threshold
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Quyết định có nên flap hay không dựa trên state.
        
        Args:
            state: np.array([dx, dy, v])
                - dx: khoảng cách ngang đến tâm lỗ
                - dy: khoảng cách dọc đến tâm lỗ
                - v: vận tốc bird
        
        Returns:
            1 nếu nên flap, 0 nếu không
        """
        dx, dy, v = state
        
        # Quy tắc 1: Bird ở dưới tâm lỗ -> flap
        if dy < self.dy_threshold:
            return 1
        
        # Quy tắc 2: Bird đang rơi quá nhanh -> flap
        if v > self.velocity_threshold:
            return 1
        
        return 0


class AdvancedRuleBaseAgent:
    """
    Agent rule-based nâng cao với nhiều quy tắc phức tạp hơn.
    """
    
    def __init__(self, 
                 dy_threshold: float = -60,
                 velocity_threshold: float = 4,
                 predict_steps: int = 3):
        """
        Args:
            dy_threshold: Ngưỡng dy để quyết định flap
            velocity_threshold: Ngưỡng vận tốc
            predict_steps: Số bước dự đoán vị trí tương lai
        """
        self.dy_threshold = dy_threshold
        self.velocity_threshold = velocity_threshold
        self.predict_steps = predict_steps
        
    def select_action(self, state: np.ndarray) -> int:
        """
        Quyết định có nên flap dựa trên nhiều yếu tố.
        
        Args:
            state: np.array([dx, dy, v])
        
        Returns:
            1 nếu nên flap, 0 nếu không
        """
        dx, dy, v = state
        
        # Dự đoán vị trí dy trong tương lai (đơn giản hóa: dy_future = dy - v * steps)
        # Vì v dương là rơi xuống, bird sẽ xa tâm lỗ hơn (dy giảm nếu v > 0)
        predicted_dy = dy - v * self.predict_steps
        
        # Quy tắc 1: Nếu sắp đến pipe (dx gần 0 hoặc đã qua một chút)
        if -50 < dx < 100:
            # Dự đoán bird sẽ ở dưới tâm quá nhiều -> flap
            if predicted_dy < self.dy_threshold:
                return 1
            
            # Bird đang ở quá dưới -> flap ngay
            if dy < self.dy_threshold - 10:
                return 1
        
        # Quy tắc 2: Khi xa pipe, giữ bird ở vị trí hợp lý
        else:
            # Giữ bird không rơi quá thấp
            if dy < -30:
                return 1
            
            # Ngăn bird rơi quá nhanh
            if v > self.velocity_threshold:
                return 1
        
        # Quy tắc 3: Không flap khi bird đang bay quá cao
        if dy > 50:
            return 0
            
        return 0


class ConservativeRuleBaseAgent:
    """
    Agent rule-based bảo thủ - luôn cố gắng giữ bird gần tâm lỗ.
    """
    
    def __init__(self, target_dy: float = 0, tolerance: float = 15):
        """
        Args:
            target_dy: Vị trí mục tiêu so với tâm lỗ (0 = đúng tâm)
            tolerance: Độ chênh lệch cho phép
        """
        self.target_dy = target_dy
        self.tolerance = tolerance
        
    def select_action(self, state: np.ndarray) -> int:
        """
        Luôn cố gắng giữ bird ở vị trí target_dy so với tâm lỗ.
        """
        dx, dy, v = state
        
        # Tính sai số hiện tại
        error = dy - self.target_dy
        
        # Nếu bird dưới mục tiêu (error < 0) và đang rơi -> flap
        if error < -self.tolerance and v > 0:
            return 1
        
        # Nếu bird dưới mục tiêu và đang rơi nhanh -> flap chắc chắn
        if error < -self.tolerance and v > 3:
            return 1
        
        # Nếu bird đang ở đúng vị trí nhưng rơi nhanh -> flap nhẹ
        if abs(error) < self.tolerance and v > 5:
            return 1
            
        return 0
