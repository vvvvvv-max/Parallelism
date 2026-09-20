"""
parallelism_demo.py — 数据并行最小示例

说明：
- 本示例演示数据并行的核心逻辑：将 batch 切分到多卡，各自前向+反向，
  梯度 all-reduce 后更新参数。
- 如果机器上没有多卡，代码会自动降级为单卡模式（仅演示流程）。

依赖：pip install torch
运行：python parallelism_demo.py
"""
import torch
import torch.nn as nn
import torch.optim as optim


def make_data(n=64, dim=10):
    x = torch.randn(n, dim)
    y = (x.sum(dim=1, keepdim=True) > 0).float()
    return x, y


class SimpleModel(nn.Module):
    def __init__(self, in_dim=10, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def data_parallel_demo():
    x, y = make_data()
    model = SimpleModel()

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel")
        model = nn.DataParallel(model)
        device = torch.device("cuda")
    else:
        print("未检测到多卡，使用单卡/CPU 演示数据并行流程")
        device = torch.device("cpu")

    model = model.to(device)
    x, y = x.to(device), y.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    for epoch in range(3):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch + 1}, loss = {loss.item():.4f}")

    print("\n数据并行关键点：")
    print("1. batch 被切分到多卡，每卡独立前向+反向")
    print("2. 梯度通过 all-reduce 求平均")
    print("3. 每卡用相同梯度更新，保证参数一致")


if __name__ == "__main__":
    data_parallel_demo()