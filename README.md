# DenseLoRA: Dense Low-Rank Adaptation of Large Language Models

IMPORTANT: This is the unofficial implementation of the ACL 2025 accepted paper [DenseLoRA: Dense Low-Rank Adaptation of Large Language Models](https://aclanthology.org/2025.acl-long.503.pdf), we selected this paper as our 2026 spring Natural Language Processing final project at Department of Computer science, NCCU

[[Final Presentation Slide](docs/Presentation.pdf)] [[Final Report](docs/Final_Report.pdf)]

## Table of Content

- [Introduction](#introduction)
- [Take a Glance at the Result](#take-a-glance-at-the-result)
- [Instruction Steps](#instruction-steps)
- [File Structure](#file-structure)

## Introduction

In this section, we introduce you what we have done on this project briefly:
1. We understood the core concept behind the paper and started implementing it via `PyTorch`
2. We gathered the required datasets and finetune lora on llama-3-8b and gemma-4-e4b as baseline
3. We conducted the experiment by switching lora to denselora on the exact same modules of the 2 models
4. We analyzed the result
6. We made the final slides and report to demonstrate our results, findings, and conclusions. The following section provides a quick view of it

for detailed results, see `/results` and `prelude/result_part2` folders

## Take a Glance at the Result

For both lora and denselora, we use the same hyper params and target {q, k, v, up, down} modules:

| Epochs | Optimizer | LR | Weight Decay | Beta | Batch Size | Warm Up Steps | Max Seq Length |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | AdamW | 0.0003 | 0.01 (default) | (0.9, 0.999) | 16 | 100 | 512 |

| Rank | Alpha | Dropout |
| --- | --- | --- |
| 32 | 32 | 0.05 |

The following are the finetune results

![loss](/docs/loss.png)

![grad](/docs/grad_norm.png)

We can see no matter it's loss or grad norm, denselora performs better than lora. Now, lets take a look at the validation result

| Dataset | LLaMA-3-8B (LoRA) | LLaMA-3-8B (DenseLoRa) | Gemma-4-E4B (LoRA) | Gemma-4-E4B (DenseLoRa) |
| --- | --- | --- | --- | --- |
| BoolQ | 71.65 | **74.80** | 71.22 | **71.19** |
| PIQA | 84.71 | **90.75** | 89.66 | **90.21** |
| SIQA | 79.53 | **82.55** | 81.68 | **83.16** |
| HellaSwag | 93.10 | **96.52** | 94.24 | **95.41** |
| WinoGrande | 83.74 | **88.79** | 87.45 | **88.63** |
| ARC-e | 86.45 | **93.43** | 94.61 | **96.89** |
| ARC-c | 75.09 | **82.85** | 85.84 | **90.36** |
| OBQA | 80.80 | **90.40** | 89.60 | **91.60** |
| **Average** | 81.88 | **87.51** | 86.79 | **88.43** |

Denselora is significantly better! Finally, let's see the param efficiency

| Model | Method | Trainable Params | % of Total |
| --- | --- | --- | --- |
| LLaMA-3-8B | LoRA | 56,623,104 | 0.7002% |
|  | DenseLoRa | 1,769,472 | 0.0220% |
| Gemma-4-E4B | LoRA | 45,907,968 | 0.5748% |
|  | DenseLoRa | 1,648,640 | 0.0208% |

## Instruction Steps

### Data Preperation

1. `cd` to project root and prepare a `data/` folder
2. `git clone https://github.com/AGI-Edgerunners/LLM-Adapters.git`
3. `mv LLM-Adapters/dataset/* data/`
4. (Optional) To make the `data/` folder cleaner: `rm -r data/AddSub/ data/AQuA/ data/gsm8k/ data/mathqa/ data/mawps/ data/MultiArith/ data/SVAMP/ data/SingleEq/`
5. After we have all the validation data gathered, it's time to get training data. `mv LLM-Adapters/ft-training_set/commonsense_170k.json data/`
6. Finished, if hit `tree data/`, it will look like:
```
data/
├── ARC-Challenge
│   ├── test.json
│   └── train.json
├── ARC-Easy
│   ├── test.json
│   └── train.json
├── boolq
│   ├── test.json
│   └── train.json
├── hellaswag
│   ├── test.json
│   └── train.json
├── openbookqa
│   ├── test.json
│   └── train.json
├── piqa
│   ├── test.json
│   └── train.json
├── social_i_qa
│   ├── test.json
│   └── train.json
├── winogrande
│   ├── test.json
│   └── train.json
└── commonsense_170k.json
```

### Finetuning

1. Adjust the hyper-params in `settings.py`
2. simply `py finetune_lora.py` and `py finetune_denselora.py`

### Inferencing

1. Adjust the hyper-params in `settings.py`
2. simply `py inference_lora.py` and `py inference_denselora.py`

## File Structure

For the src code:
```
.
├── finetune_denselora.py
├── finetune_lora.py
├── inference_denselora.py
├── inference_lora.py
├── prelude (Just to showcase the importance of finetuning)
├── README.md
├── requirements.txt
├── settings.py
└── src
    ├── denselora.py
    └── utils.py
```

Follow the instruction [here](#instruction-steps) to get the data, and start training, you will have:

<details>
<summary>Full File Structure</summary>

```
.
├── data
│   ├── ARC-Challenge
│   │   ├── test.json
│   │   └── train.json
│   ├── ARC-Easy
│   │   ├── test.json
│   │   └── train.json
│   ├── boolq
│   │   ├── test.json
│   │   └── train.json
│   ├── commonsense_170k.json
│   ├── hellaswag
│   │   ├── test.json
│   │   └── train.json
│   ├── openbookqa
│   │   ├── test.json
│   │   └── train.json
│   ├── piqa
│   │   ├── test.json
│   │   └── train.json
│   ├── social_i_qa
│   │   ├── test.json
│   │   └── train.json
│   └── winogrande
│       ├── test.json
│       └── train.json
├── denselora_weights
│   ├── gemma
│   │   └── ver1
│   │       └── checkpoint-21304
│   │           ├── denselora_adapters.pt
│   │           ├── optimizer.pt
│   │           ├── rng_state.pth
│   │           ├── scheduler.pt
│   │           └── trainer_state.json
│   └── llama
│       └── ver1
│           └── checkpoint-21304
│               ├── denselora_adapters.pt
│               ├── optimizer.pt
│               ├── rng_state.pth
│               ├── scheduler.pt
│               └── trainer_state.json
├── finetune_denselora.py
├── finetune_lora.py
├── inference_denselora.py
├── inference_lora.py
├── lora_weights
│   ├── gemma
│   │   └── ver1
│   │       ├── adapter_config.json
│   │       ├── adapter_model.safetensors
│   │       ├── chat_template.jinja
│   │       ├── checkpoint-21304
│   │       │   ├── adapter_config.json
│   │       │   ├── adapter_model.safetensors
│   │       │   ├── chat_template.jinja
│   │       │   ├── optimizer.pt
│   │       │   ├── README.md
│   │       │   ├── rng_state.pth
│   │       │   ├── scheduler.pt
│   │       │   ├── tokenizer_config.json
│   │       │   ├── tokenizer.json
│   │       │   ├── trainer_state.json
│   │       │   └── training_args.bin
│   │       ├── README.md
│   │       ├── tokenizer_config.json
│   │       ├── tokenizer.json
│   │       └── training_args.bin
│   └── llama
│       └── ver1
│           ├── adapter_config.json
│           ├── adapter_model.safetensors
│           ├── checkpoint-21304
│           │   ├── adapter_config.json
│           │   ├── adapter_model.safetensors
│           │   ├── optimizer.pt
│           │   ├── README.md
│           │   ├── rng_state.pth
│           │   ├── scheduler.pt
│           │   ├── tokenizer_config.json
│           │   ├── tokenizer.json
│           │   ├── trainer_state.json
│           │   └── training_args.bin
│           ├── README.md
│           ├── tokenizer_config.json
│           ├── tokenizer.json
│           └── training_args.bin
├── prelude
│   ├── chatgpt_part2.py
│   ├── chatgpt_test.py
│   ├── claude_part2.py
│   ├── claude_test.py
│   ├── gemini_part2.py
│   ├── gemini_test.py
│   ├── llama_instruct_test.py
│   ├── llama_test.py
│   ├── result
│   │   ├── chatgpt_test.txt
│   │   ├── claude_test.txt
│   │   ├── gemini_test.txt
│   │   ├── llama_instruct_test.txt
│   │   └── llama_test.txt
│   └── result_part2
│       ├── chatgpt_test.txt
│       ├── claude_test.txt
│       └── gemini_test.txt
├── README.md
├── requirements.txt
├── result
│   ├── denselora
│   │   ├── gemma
│   │   │   └── result_v1.txt
│   │   └── llama
│   │       └── result_v1.txt
│   └── lora
│       ├── gemma
│       │   └── result_v1.txt
│       └── llama
│           └── result_v1.txt
├── settings.py
└── src
    ├── denselora.py
    └── utils.py
```

</details>