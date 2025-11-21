# 🐦 Flappy Bird Reinforcement Learning

Train AI chơi Flappy Bird bằng Deep Q-Network (DQN).

## 🚀 Quick Start

### Cài đặt

```bash
git clone https://github.com/canhtungdz/Flappy_RL.git
cd flappy_rl
pip install pygame torch numpy
```

### Chơi với agent đã train

```bash
python3 scripts/run_agent.py
```
Trong scripts/run_agent.py mặc định để chạy model best_model_10000.pth
có thể thay đổi model bằng cách thay đường dẫn đến model (file .pth)
Trong đó cũng chứa đoạn code để test với Rulebase agent.
## 🎓 Training

### Train mới

```bash
python3 scripts/train_dqn.py --episodes 10000
```
model sẽ được lưu ở thư mục checkpoints
### Train với config tùy chỉnh

```bash
python3 scripts/train_dqn.py \
    --episodes 10000 \
    --lr 0.001 \
    --epsilon-decay 300000
```

### Resume training

```bash
# Xem checkpoints có sẵn
python3 scripts/list_checkpoints.py

# Resume từ checkpoint
python3 scripts/train_dqn.py \
    --resume checkpoints/dqn_episode_5000.pth \
    --episodes 10000
```

## 📊 Evaluation

```bash
# Test model
python3 scripts/evaluate_dqn.py \
    --checkpoint checkpoints/best_model.pth \
    --episodes 10

# So sánh các agents
python3 scripts/compare_agents.py
```

## ⚙️ Arguments

| Argument | Default | Mô tả |
|----------|---------|-------|
| `--episodes` | 10000 | Số episodes train |
| `--lr` | 0.0001 | Learning rate |
| `--epsilon-decay` | 100000 | Epsilon decay steps |
| `--batch-size` | 64 | Batch size |
| `--hidden-dim` | 128 | Hidden layer size |
| `--resume` | None | Path để resume training |

## 📁 Cấu trúc

```
flappy_rl/
├── agent/              # AI agents (Random, Rule-based, DQN)
├── training/           # Training code
├── scripts/            # Scripts để train/evaluate
├── FlapPyBird/         # Game engine
└── checkpoints/        # Saved models
└── saved_model/        # các model đã train sẵn
```

## 🎮 State & Action

**State:** `[dx, dy, v]`
- `dx`: khoảng cách ngang đến pipe
- `dy`: khoảng cách dọc đến tâm lỗ  
- `v`: vận tốc bird

**Action:** `[0, 1]` (không flap / flap)

## 🛠️ Troubleshooting

### Model không học (score = 0)

```bash
# Tăng learning rate
python3 scripts/train_dqn.py --lr 0.001

# Tăng epsilon decay (explore lâu hơn)
python3 scripts/train_dqn.py --epsilon-decay 300000
```

## 📝 Requirements

```
Python 3.8+
pygame 2.4.0+
torch 2.0.0+
numpy 1.24.0+
```