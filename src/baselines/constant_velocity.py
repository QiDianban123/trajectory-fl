"""
常速度基线模型：假设未来轨迹 = 历史末帧位置 + 末帧速度 × 时间步。
"""

import torch
import torch.nn as nn


class ConstantVelocityBaseline(nn.Module):
    """
    最简单的轨迹预测基线，无训练参数。
    
    Args:
        future_steps: 预测未来多少帧（默认 50，与 LSTM 一致）。
    """

    def __init__(self, future_steps: int = 50):
        super().__init__()
        self.future_steps = future_steps

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: [B, T_h, 2]，历史轨迹（x, y）。
        Returns:
            future: [B, T_f, 2]，预测轨迹。
        """
        # 计算末帧速度：最后两帧的位移
        velocity = history[:, -1, :] - history[:, -2, :]  # [B, 2]

        # 时间步 [1, 2, ..., T_f]
        device = history.device
        dtype = history.dtype
        time_steps = torch.arange(
            1, self.future_steps + 1, dtype=dtype, device=device
        )
        time_steps = time_steps.view(1, self.future_steps, 1)  # [1, T_f, 1]

        # 末帧位置
        last_pos = history[:, -1:, :]  # [B, 1, 2]

        # 外推：future = last_pos + velocity * time
        future = last_pos + velocity.unsqueeze(1) * time_steps  # [B, T_f, 2]
        return future