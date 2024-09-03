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
from tqdm import tqdm
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

model_path = str(input("Enter the model path: "))
dataset_path = str(input("Enter the dataset path: "))
output_path = "scores/scores.txt"  # Output path modified to a text file

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

# Alpaca prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Function to calculate distinctness scores
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

# Function to calculate length ratio
def calculate_length_ratio(predictions, references):
    ratios = [len(pred.split()) / len(ref.split()) if len(ref.split()) > 0 else 0 for pred, ref in zip(predictions, references)]
    return sum(ratios) / len(ratios)

# Function to calculate various metrics
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

# Function to extract response from generated text
def extract_response(text):
    parts = text.split("###")
    response_part = parts[-1]
    response = response_part[12:]
    response = response[:-17]
    return response

# Function to trim generated text to the last complete sentence
def trim_to_last_sentence(text):
    end_punctuation = ['.', '!', '?']
    text = text.rstrip()
    last_positions = [text.rfind(punct) for punct in end_punctuation]
    last_sentence_end = max(last_positions)
    if last_sentence_end != -1:
        return text[:last_sentence_end + 1]
    else:
        return text

# Function to save evaluation scores to a text file
def save_scores_to_txt(output_path, image_path, prediction, ground_truth, scores):
    with open(output_path, 'a') as file:
        file.write(f"Image Path: {image_path}\n")
        file.write(f"Prediction: {prediction}\n")
        file.write(f"Ground Truth: {ground_truth}\n")
        file.write(f"BLEU: {scores[0]:.4f}\n")
        file.write(f"ROUGE-1: {scores[1]:.4f}\n")
        file.write(f"ROUGE-2: {scores[2]:.4f}\n")
        file.write(f"ROUGE-L: {scores[3]:.4f}\n")
        file.write(f"BERTScore Precision: {scores[4]:.4f}\n")
        file.write(f"BERTScore Recall: {scores[5]:.4f}\n")
        file.write(f"BERTScore F1: {scores[6]:.4f}\n")
        file.write(f"Distinct-1: {scores[7]:.4f}\n")
        file.write(f"Distinct-2: {scores[8]:.4f}\n")
        file.write(f"Repetition Rate: {scores[9]:.4f}\n")
        file.write(f"Length Ratio: {scores[10]:.4f}\n")
        file.write("\n" + "="*80 + "\n\n")

# Load the dataset
df = pd.read_csv(dataset_path)

# Initialize score lists
perplexities = []
bleu_scores = []
rouge_scores = []
bertscore_p = []
bertscore_r = []
bertscore_f1 = []
distinct1_scores = []
distinct2_scores = []
moverscore_scores = []
repetition_rates = []
length_ratios = []

FastLanguageModel.for_inference(model)

# Iterate over each row in the dataset
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    instruction = row['instruction']
    input_text = row['input']
    target_output = row['description']
    image_path = row.get('image_path', 'N/A')  # Fetch image path if available

    # Prepare the input for the model
    prompt = alpaca_prompt.format(instruction, input_text, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda:0")

    # Generate the model output
    outputs = model.generate(**inputs, max_new_tokens=1000, use_cache=True)
    generated_output = extract_response(str(tokenizer.batch_decode(outputs)))
    generated_output = trim_to_last_sentence(generated_output)

    print("=============================Output=================")
    print(generated_output)
    print("=============================Target=================")
    print(target_output)
    print("====================================================")

    # Calculate scores for the current example
    iteration_scores = calculate_metrics([generated_output], [target_output])

    # Save the prediction, ground truth, and scores to a text file
    save_scores_to_txt(output_path, image_path, generated_output, target_output, iteration_scores)

    # Append individual scores to the respective lists for averaging later
    bleu_scores.append(iteration_scores[0])
    rouge_scores.append(iteration_scores[1:4])
    bertscore_p.append(iteration_scores[4])
    bertscore_r.append(iteration_scores[5])
    bertscore_f1.append(iteration_scores[6])
    distinct1_scores.append(iteration_scores[7])
    distinct2_scores.append(iteration_scores[8])
    repetition_rates.append(iteration_scores[9])
    length_ratios.append(iteration_scores[10])

    torch.cuda.empty_cache()

# Calculate average scores
avg_bleu = sum(bleu_scores) / len(bleu_scores)
avg_rouge_1 = sum([score[0] for score in rouge_scores]) / len(rouge_scores)
avg_rouge_2 = sum([score[1] for score in rouge_scores]) / len(rouge_scores)
avg_rouge_l = sum([score[2] for score in rouge_scores]) / len(rouge_scores)
avg_bertscore_p = sum(bertscore_p) / len(bertscore_p)
avg_bertscore_r = sum(bertscore_r) / len(bertscore_r)
avg_bertscore_f1 = sum(bertscore_f1) / len(bertscore_f1)
avg_distinct1 = sum(distinct1_scores) / len(distinct1_scores)
avg_distinct2 = sum(distinct2_scores) / len(distinct2_scores)
avg_repetition_rate = sum(repetition_rates) / len(repetition_rates)
avg_length_ratio = sum(length_ratios) / len(length_ratios)

# Write the average scores to the output file
with open(output_path, 'a') as file:
    file.write("\n\n====== Average Scores ======\n")
    file.write(f"Average BLEU: {avg_bleu:.4f}\n")
    file.write(f"Average ROUGE-1: {avg_rouge_1:.4f}\n")
    file.write(f"Average ROUGE-2: {avg_rouge_2:.4f}\n")
    file.write(f"Average ROUGE-L: {avg_rouge_l:.4f}\n")
    file.write(f"Average BERTScore Precision: {avg_bertscore_p:.4f}\n")
    file.write(f"Average BERTScore Recall: {avg_bertscore_r:.4f}\n")
    file.write(f"Average BERTScore F1: {avg_bertscore_f1:.4f}\n")
    file.write(f"Average Distinct-1: {avg_distinct1:.4f}\n")
    file.write(f"Average Distinct-2: {avg_distinct2:.4f}\n")
    file.write(f"Average Repetition Rate: {avg_repetition_rate:.4f}\n")
    file.write(f"Average Length Ratio: {avg_length_ratio:.4f}\n")

print("Evaluation complete. Results written to", output_path)
