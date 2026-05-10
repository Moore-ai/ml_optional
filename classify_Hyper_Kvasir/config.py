from pathlib import Path

# 路径
DATA_ROOT = Path("Hyper-Kvasir/labeled-images")
CLASSIFY_DIR = Path("classify_Hyper_Kvasir")
OUTPUT_DIR = CLASSIFY_DIR / "output"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# 数据
IMG_SIZE = 384
BATCH_SIZE = 32
NUM_WORKERS = 4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_CLASSES = 23

# 模型
DROPOUT = 0.3
DROPOUT_HEAD = 0.2

# 训练
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 1e-4
T_MAX = 30
ETA_MIN = 1e-6
EARLY_STOP_PATIENCE = 10
FREEZE_EPOCHS = 5
LABEL_SMOOTHING = 0.1

# 混合精度
AMP_DTYPE = "bfloat16"

# 可视化
FIGURE_DPI = 150
