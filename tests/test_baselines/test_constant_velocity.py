import torch
from src.baselines import ConstantVelocityBaseline


def test_output_shape():
    """验证输出 shape 正确。"""
    model = ConstantVelocityBaseline(future_steps=50)
    history = torch.randn(4, 30, 2)  # B=4, T_h=30
    future = model(history)
    assert future.shape == (4, 50, 2)


def test_no_trainable_parameters():
    """常速度基线没有可训练参数。"""
    model = ConstantVelocityBaseline()
    params = list(model.parameters())
    assert len(params) == 0


def test_perfect_prediction_on_constant_velocity():
    """对真正的匀速直线运动，预测应完全准确。"""
    model = ConstantVelocityBaseline(future_steps=10)
    
    # 构造匀速直线运动：每帧 x 方向 +1，y 方向 +2
    history = torch.zeros(1, 5, 2)
    for t in range(1, 5):
        history[0, t, 0] = history[0, t - 1, 0] + 1.0  # x +1
        history[0, t, 1] = history[0, t - 1, 1] + 2.0  # y +2
    
    future = model(history)
    
    # 验证：未来第 1 帧 = 末帧 + 速度
    expected_pos = history[0, -1, :] + (history[0, -1, :] - history[0, -2, :])
    assert torch.allclose(future[0, 0, :], expected_pos, atol=1e-6)