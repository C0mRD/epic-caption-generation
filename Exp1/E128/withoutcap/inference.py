import torch
import os
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "model128_bbox_iembd",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# Image: 001297.jpg
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Generate a detailed and accurate caption for the given image within 50 words.", # instruction
        "label_boundingbox:{chair: {'xyxy': {'Left': 1, 'Top': 87, 'Right': 358, 'Bottom': 500}},sofa: {'xyxy': {'Left': 0, 'Top': 79, 'Right': 352, 'Bottom': 500}},person: {'xyxy': {'Left': 1, 'Top': 83, 'Right': 347, 'Bottom': 419}}}, \n\
        image_embedding:[0.19, 0.63, 0.1, -0.01, 0.17, 0.86, 0.02, 0.01, 0.78, 1.19, -1.14, -0.03, 0.04, -0.47, 0.13, -0.2, -0.96, -0.85, -1.11, -1.46, -0.63, -1.17, -0.55, -0.69, 0.12, -0.63, 0.94, 0.59, 0.07, -0.26, 0.68, 0.23, -0.19, -0.17, -1.06, 0.25, -0.02, 0.04, 0.57, -0.03, 0.22, 0.56, 1.41, -0.61, -0.15, 1.1, -0.13, 0.67, -0.07, 0.48, -0.0, -0.39, 0.03, 0.06, 1.24, 0.42, -0.61, 0.47, 0.21, -0.47, -0.24, -0.58, -0.64, 0.19, 0.25, -0.85, -0.5, -0.23, -1.89, 2.54, -0.06, -0.05, 0.57, -0.51, 0.79, -0.6, -1.18, -0.06, 0.24, 0.23, -1.13, 0.56, 1.1, -0.16, 0.47, -0.23, 0.69, 0.16, -0.08, -0.95, 0.63, -0.14, -0.73, -1.07, -1.37, 0.44, 0.22, 0.43, 0.38, 0.11]", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
print(f"001297.jpg:\n{tokenizer.batch_decode(outputs)}\n")


# Image: 001033.jpg
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Generate a detailed and accurate caption for the given image within 50 words.", # instruction
        "label_boundingbox:{bicycle: {'xyxy': {'Left': 151, 'Top': 232, 'Right': 314, 'Bottom': 474}},person: {'xyxy': {'Left': 144, 'Top': 96, 'Right': 311, 'Bottom': 389}}}, \n\
        image_embedding:[0.06, -0.8, -0.55, 0.11, -0.58, 0.73, -1.13, -0.01, 0.18, 1.13, -0.19, 0.32, -0.28, -0.25, 0.29, -0.66, -0.48, -1.46, 0.2, -1.13, 0.1, -0.57, 0.51, 0.02, -0.43, -0.64, 0.21, -0.13, 0.05, 0.56, 1.62, -0.29, -1.07, 0.25, 1.09, -0.24, -0.01, -1.19, 0.26, 0.3, -0.13, 0.15, 0.15, 0.73, -0.09, 0.92, 0.17, -0.58, -0.47, -1.37, -0.74, -0.87, 0.44, -0.0, 1.43, 0.25, 1.75, 0.79, 0.9, 0.77, -0.32, -0.21, 0.26, 0.12, 0.98, 0.16, 0.05, -0.59, 0.18, 1.5, 0.15, -0.03, 0.05, -0.48, 0.35, -0.01, 0.24, 0.1, 0.24, -0.1, 0.99, 0.36, 0.23, 0.01, 0.23, -0.08, 0.05, 0.43, -0.94, 1.34, 0.21, -0.97, -0.28, -0.47, 0.3, -0.02, -0.4, -0.05, 0.43, -0.23]", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
print(f"001033.jpg:\n{tokenizer.batch_decode(outputs)}\n")


# Image: 000465.jpg
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Generate a detailed and accurate caption for the given image within 50 words.", # instruction
        "label_boundingbox:{car: {'xyxy': {'Left': 0, 'Top': 130, 'Right': 500, 'Bottom': 318}}}, \n\
        image_embedding:[1.0, 0.2, -0.49, 0.39, 0.2, -0.21, -0.37, 0.55, 0.04, -1.54, 0.02, 0.06, -0.32, -0.05, -0.25, -0.49, 0.41, -0.23, -1.41, -1.11, -0.11, 0.09, -0.03, -0.5, 0.51, -0.21, -1.01, 0.37, -0.21, -0.55, -0.24, 0.01, 0.88, 0.02, -1.07, 0.29, 0.11, -0.71, -0.72, -0.09, -0.62, -0.13, 0.07, 0.24, 0.25, -0.33, -0.46, -0.45, 0.48, -0.37, -0.22, 0.6, 0.77, -0.97, -0.56, -0.24, 0.87, -0.75, 0.86, 0.55, 0.15, -0.67, -0.34, -0.2, -0.75, -0.68, -0.62, -1.01, 0.74, 1.03, -0.2, 0.23, 0.61, 0.66, 0.77, -0.58, -0.46, 0.26, 0.53, 0.29, 0.27, -0.59, 0.75, 0.36, 0.16, -0.26, 0.11, 1.05, -0.04, -0.63, 0.5, 0.0, 0.49, -0.1, -2.01, 0.06, 0.19, 0.41, -0.28, -0.0]", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
print(f"000465.jpg:\n{tokenizer.batch_decode(outputs)}\n")
