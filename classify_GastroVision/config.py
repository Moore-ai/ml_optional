from pathlib import Path

# 路径
DATA_ROOT = Path("GastroVision/osfstorage-archive/Gastrovision/Gastrovision/Gastrovision")
CLASSIFY_DIR = Path("classify_GastroVision")
OUTPUT_BASE = CLASSIFY_DIR / "output"
LOG_DIR = OUTPUT_BASE / "logs"
FIGURE_DIR = OUTPUT_BASE / "figures"
CHECKPOINT_DIR = OUTPUT_BASE / "checkpoints"

# 数据
IMG_SIZE = 384
BATCH_SIZE = 32
NUM_WORKERS = 4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_CLASSES = 27

# 模型
DROPOUT = 0.3
DROPOUT_HEAD = 0.2

# 训练
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 1e-4
T_MAX = 50
ETA_MIN = 1e-6
MIN_SAMPLES = 100
EARLY_STOP_PATIENCE = 10
FREEZE_EPOCHS = 5
LABEL_SMOOTHING = 0.1

# 混合精度
AMP_DTYPE = "bfloat16"

# 可视化
FIGURE_DPI = 150

# === 精度改进开关 ===

# Focal Loss（True 时替代 CrossEntropyLoss）
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 0.5

# RandAugment（True 时替代手动增强）
USE_RAND_AUGMENT = True
RA_NUM_OPS = 2
RA_MAGNITUDE = 5

# MixUp / CutMix
USE_MIXUP = True
MIXUP_ALPHA = 1.0
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 0.5

# Test-Time Augmentation
USE_TTA = True

# CosineAnnealingWarmRestarts（True 时替代 CosineAnnealingLR）
USE_WARM_RESTARTS = False
WR_T0 = 10
WR_T_MULT = 2
