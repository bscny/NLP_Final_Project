# NLP_Final_Project

## Table of Content

- [Instruction Steps](#instruction-steps)
- [Result Overview](#result-overview)
- [File Structure](#file-structure)

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

### Inferencing

## Result Overview

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