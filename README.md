# MiniGrid 强化学习实验

本项目研究 MiniGrid DoorKey 稀疏奖励导航任务，提供可复现的实验代码、固定评估场景及原始结果。

当前公开内容为 **PPO baseline v1，训练随机种子0、1、2**，仅发布三个final和三个best权重，同时保留完整轻量实验记录。训练完成后的 final 策略作为主结果，按验证集选择的 best checkpoint 仅作补充。所有模型采用相同的最终测试场景；8×8测试不进行额外训练。

## 文档导航

- [Baseline：方法、复现步骤、完整结果与分析](docs/baseline.md)
- [GitHub 发布操作说明](PUBLISH_GITHUB.md)

## 目录结构

```text
MiniGrid/
├── README.md                  项目总览与导航
├── docs/baseline.md            baseline的详细说明和分析
├── PUBLISH_GITHUB.md           发布步骤
├── requirements.txt           软件依赖
├── configs/                   训练配置、固定场景列表、best选择规则
├── minigrid_baseline/          环境、CNN、PPO预算控制和评估模块
├── train_minigrid.py           训练入口
├── baseline.py                评估、选择best、汇总、绘图及核验入口
├── results/                   三个seed的公开模型和实验记录
└── ARTIFACTS.json              公开实验产物的校验清单
```

`runs/`用于新训练，`reports/`用于对公开结果重新生成汇总和图表，两者均已加入Git忽略列表。

## 快速开始

建议使用Python 3.11和独立环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python baseline.py verify
```

以上核验不训练或执行策略。PyTorch应选择适合自身平台的构建；完整训练、重新评估和短流程检查命令见[baseline文档](docs/baseline.md)。

## 公开与复现原则

公开内容使用通用路径和运行接口，不依赖特定计算平台。模型、固定场景、逐Episode结果和软件版本用于支持结果核验；跨平台重新训练不保证逐位一致。请保留原始结果，将新实验写入独立目录。

本项目基于MiniGrid和Stable-Baselines3，具体参考来源与实验边界见[baseline文档](docs/baseline.md)。代码所有者应决定仓库许可证，并保留第三方归属及适用的AI辅助使用说明。
