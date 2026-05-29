import torch

# General ----------------------------------------------------------------------
# MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MODEL_ID = "google/gemma-4-E4B-it"

TRAINING_DATA_PATH = "./data/commonsense_170k.json"

BOOLQ_DATA_PATH = "./data/boolq/test.json"
PIQA_DATA_PATH = "./data/piqa/test.json"
SIQA_DATA_PATH = "./data/social_i_qa/test.json"
HELLAS_DATA_PATH = "./data/hellaswag/test.json"
WINOG_DATA_PATH = "./data/winogrande/test.json"
ARCE_DATA_PATH = "./data/ARC-Easy/test.json"
ARCC_DATA_PATH = "./data/ARC-Challenge/test.json"
OBQA_DATA_PATH = "./data/openbookqa/test.json"

DENSE_LORA_OUTPUT_DIR = "./denselora_weights/gemma/ver1"
LORA_OUTPUT_DIR = "./lora_weights/gemma/ver1"

DENSE_LORA_ADAPTER_PATH  = "./denselora_weights/gemma/ver1/checkpoint-21304/denselora_adapters.pt"
LORA_ADAPTER_PATH  = "./lora_weights/gemma/ver1/checkpoint-21304"  # Since PEFT expects the folder path with weights and config

DENSE_LORA_RESULT_PATH = "./result/denselora/gemma/result_v1.txt"
LORA_RESULT_PATH = "./result/lora/gemma/result_v1.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_TYPE = torch.bfloat16

WANDB_RUN_NAME = "run-gemma-denselora-1"

# Training Settings ------------------------------------------------------------
EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LR = 3e-4
MAX_SEQ_LENGTH = 512

RANK = 32
ALPHA = 32
DROPOUT = 0.05
WARMUP_STEPS = 100
LOGGING_STEPS = 50

# Inference Settings -----------------------------------------------------------
MAX_NEW_TOKENS = 128
NUM_BEAMS = 4
TEMPERATURE = 0.1
TOP_P = 0.75
TOP_K = 40
