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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

model_path = str(input("Enter the model path: "))
dataset_path = str(input("Enter the dataset path: "))
output_path = "scores/scores.csv"  # Output path modified to a CSV file

max_seq_length = 2048
dtype = torch.float16
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    cache_dir="../llama3.1"
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)

EOS_TOKEN = tokenizer.eos_token

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = ""
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

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

def calculate_repetition_rate(predictions):
    total_tokens = 0
    repeated_tokens = 0
    for pred in predictions:
        tokens = pred.split()
        total_tokens += len(tokens)
        repeated_tokens += len(tokens) - len(set(tokens))

    return repeated_tokens / total_tokens if total_tokens else 0

def calculate_length_ratio(predictions, references):
    ratios = [len(pred.split()) / len(ref.split()) if len(ref.split()) > 0 else 0 for pred, ref in zip(predictions, references)]
    return sum(ratios) / len(ratios)

def calculate_metrics(predictions, references):
    bleu_scores = []
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for pred, ref in zip(predictions, references):
        bleu_score = sentence_bleu([ref.split()], pred.split())
        bleu_scores.append(bleu_score)

        scores = scorer.score(ref, pred)
        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge1 = sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"])
    avg_rouge2 = sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"])
    avg_rougeL = sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"])

    P, R, F1 = bert_score(predictions, references, model_type="bert-base-uncased", lang="en", rescale_with_baseline=False)

    distinct1, distinct2 = calculate_distinctness(predictions)
    repetition_rate = calculate_repetition_rate(predictions)
    length_ratio = calculate_length_ratio(predictions, references)

    return avg_bleu, avg_rouge1, avg_rouge2, avg_rougeL, P.mean().item(), R.mean().item(), F1.mean().item(), distinct1, distinct2, repetition_rate, length_ratio

def extract_response(text):
    parts = text.split("###")
    response_part = parts[-1]
    response = response_part[12:]
    response = response[:-17]
    return response

def trim_to_last_sentence(text):
    end_punctuation = ['.', '!', '?']
    text = text.rstrip()
    last_positions = [text.rfind(punct) for punct in end_punctuation]
    last_sentence_end = max(last_positions)
    if last_sentence_end != -1:
        return text[:last_sentence_end + 1]
    else:
        return text

def save_scores_to_csv(output_path, row_data):
    # Check if the file exists and if not, create it with headers
    file_exists = os.path.isfile(output_path)
    headers = [
        "Model_Name", "Prediction", "Ground_Truth", "Extracted_Response",
        "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L",
        "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1",
        "Distinct-1", "Distinct-2", "Repetition_Rate", "Length_Ratio"
    ]

    df = pd.DataFrame([row_data], columns=headers)

    if file_exists:
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, mode='w', header=True, index=False)

def evaluate_model_on_dataset(model_path, dataset_path, tokenizer, output_path):
    dataset = load_dataset('csv', data_files={'test': dataset_path})

    test_data = dataset['test']

    formatted_dataset = test_data.map(formatting_prompts_func, batched=True)

    predictions = []
    ground_truths = []

    FastLanguageModel.for_inference(model)

    for idx, example in enumerate(formatted_dataset['text']):
        inputs = tokenizer(
            example,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048
        ).to('cuda')

        outputs = model.generate(**inputs, max_new_tokens=2000)
        generated_outputs = extract_response(str(tokenizer.decode(outputs[0], skip_special_tokens=True)))
        prediction = trim_to_last_sentence(generated_outputs)
        print("\nprediction: ===== \n")
        print(prediction)
        print("\nend of prediction: ===== \n")
        predictions.append(prediction)

        ground_truth = example.split("### Response:")[1].strip()
        print("\nGT: ===== \n")
        print(ground_truth)
        print("\nend of GT: ===== \n")
        ground_truths.append(ground_truth)

        # Calculate metrics for the current iteration
        iteration_scores = calculate_metrics([prediction], [ground_truth])

        # Save the prediction, extracted response, ground truth, and scores
        row_data = [
            model_path, prediction, ground_truth, generated_outputs,
            iteration_scores[0], iteration_scores[1], iteration_scores[2], iteration_scores[3],
            iteration_scores[4], iteration_scores[5], iteration_scores[6],
            iteration_scores[7], iteration_scores[8], iteration_scores[9], iteration_scores[10]
        ]
        save_scores_to_csv(output_path, row_data)

    # Calculate overall metrics for all predictions
    overall_scores = calculate_metrics(predictions, ground_truths)

    return overall_scores

# Evaluate the model on the dataset
scores = evaluate_model_on_dataset(model_path, dataset_path, tokenizer, output_path)

# Print aggregated scores
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
