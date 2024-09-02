import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from collections import Counter
import numpy as np
import torch
import os
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

model_path = str(input("Enter the path to the model: "))
dataset_path = str(input("Enter the path to the dataset: "))
output_path = str(input("Enter the path to the output file to save score: "))

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Function to format prompts
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["description"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Function to calculate distinct-1 and distinct-2
def calculate_distinctness(predictions):
    unigrams = Counter()
    bigrams = Counter()
    for pred in predictions:
        tokens = pred.split()
        unigrams.update(tokens)
        bigrams.update(zip(tokens, tokens[1:]))

    distinct1 = len(unigrams) / sum(unigrams.values()) if unigrams else 0
    distinct2 = len(bigrams) / sum(bigrams.values()) if bigrams else 0
    return distinct1, distinct2

# Function to calculate repetition rate
def calculate_repetition_rate(predictions):
    total_tokens = 0
    repeated_tokens = 0
    for pred in predictions:
        tokens = pred.split()
        total_tokens += len(tokens)
        repeated_tokens += len(tokens) - len(set(tokens))

    return repeated_tokens / total_tokens if total_tokens else 0

# Function to calculate average length ratio
def calculate_length_ratio(predictions, references):
    ratios = [len(pred.split()) / len(ref.split()) if len(ref.split()) > 0 else 0 for pred, ref in zip(predictions, references)]
    return sum(ratios) / len(ratios)

# Function to calculate BLEU, ROUGE, BERTScore, Distinctness, Repetition Rate, and Length Ratio
def calculate_metrics(predictions, references):
    # BLEU and ROUGE
    bleu_scores = []
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for pred, ref in zip(predictions, references):
        # BLEU score
        bleu_score = sentence_bleu([ref.split()], pred.split())
        bleu_scores.append(bleu_score)

        # ROUGE scores
        scores = scorer.score(ref, pred)
        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge1 = sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"])
    avg_rouge2 = sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"])
    avg_rougeL = sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"])

    # BERTScore
    P, R, F1 = bert_score(predictions, references, model_type="bert-base-uncased", lang="en", rescale_with_baseline=False)

    # Distinctness
    distinct1, distinct2 = calculate_distinctness(predictions)

    # Repetition Rate
    repetition_rate = calculate_repetition_rate(predictions)

    # Length Ratio
    length_ratio = calculate_length_ratio(predictions, references)

    return avg_bleu, avg_rouge1, avg_rouge2, avg_rougeL, P.mean().item(), R.mean().item(), F1.mean().item(), distinct1, distinct2, repetition_rate, length_ratio

# Function to load dataset, perform inference, and calculate metrics
def evaluate_model_on_dataset(model_path, dataset_path, tokenizer, score_file_path):
    # Load dataset
    dataset = load_dataset('csv', data_files={'train': dataset_path})
    dataset = dataset['train']
    small_dataset = dataset.train_test_split(test_size=0.9)['train']  # 10% of the data

    # Format prompts
    formatted_dataset = small_dataset.map(formatting_prompts_func, batched=True)

    predictions = []
    ground_truths = []

    FastLanguageModel.for_inference(model)

    # Perform inference one by one
    for example in formatted_dataset['text']:
        inputs = tokenizer(
            example,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to('cuda')

        output = model.generate(**inputs, max_new_tokens=64)
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append(prediction)

        ground_truth = example.split("### Response:")[1].strip()
        ground_truths.append(ground_truth)

    # Calculate metrics
    scores = calculate_metrics(predictions, ground_truths)

    # Save scores to file
    with open(score_file_path, "w") as f:
        f.write(f"Average BLEU Score: {scores[0]}\n")
        f.write(f"Average ROUGE-1 Score: {scores[1]}\n")
        f.write(f"Average ROUGE-2 Score: {scores[2]}\n")
        f.write(f"Average ROUGE-L Score: {scores[3]}\n")
        f.write(f"BERTScore Precision: {scores[4]}\n")
        f.write(f"BERTScore Recall: {scores[5]}\n")
        f.write(f"BERTScore F1: {scores[6]}\n")
        f.write(f"Distinct-1: {scores[7]}\n")
        f.write(f"Distinct-2: {scores[8]}\n")
        f.write(f"Repetition Rate: {scores[9]}\n")
        f.write(f"Length Ratio: {scores[10]}\n")

    return scores


# Evaluate the model on the dataset
scores = evaluate_model_on_dataset(model_path, dataset_path, tokenizer, output_path)

# Print scores
print(f"Average BLEU Score: {scores[0]}")
print(f"Average ROUGE-1 Score: {scores[1]}")
print(f"Average ROUGE-2 Score: {scores[2]}")
print(f"Average ROUGE-L Score: {scores[3]}")
print(f"BERTScore Precision: {scores[4]}")
print(f"BERTScore Recall: {scores[5]}")
print(f"BERTScore F1: {scores[6]}")
print(f"Distinct-1: {scores[7]}")
print(f"Distinct-2: {scores[8]}")
print(f"Repetition Rate: {scores[9]}")
print(f"Length Ratio: {scores[10]}")
