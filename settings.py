import torch

# General ----------------------------------------------------------------------
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
TRAINING_DATA_PATH = "./data/commonsense_170k.json"
OUTPUT_DIR = "./denselora_weights"
ADAPTER_PATH  = "./denselora_weights/checkpoint-10652/denselora_adapters.pt"
LOG_DIR = "./logs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_TYPE = torch.bfloat16

# Training Settings ------------------------------------------------------------
EPOCHS = 2
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
LR = 3e-4
MAX_SEQ_LENGTH = 512

RANK = 32
ALPHA = 32
DROPOUT = 0.05
WARMUP_STEPS = 100
LOGGING_STEPS = 500

# Inference Settings -----------------------------------------------------------
MAX_NEW_TOKENS = 128
NUM_BEAMS = 4
TEMPERATURE = 0.1
TOP_P = 0.75
TOP_K = 40