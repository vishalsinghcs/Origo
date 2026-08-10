import json
import argparse
from pathlib import Path
from tqdm import tqdm
from unsloth import FastLanguageModel
import pandas as pd

def load_test_data(filepath):
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def extract_predicted_json(text):
    # Try to find JSON block in the output
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
    except:
        pass
    return None

def evaluate_layer1(model, tokenizer, test_data):
    print("Evaluating Layer 1 (Router)...")
    y_true = []
    y_pred = []
    
    for sample in tqdm(test_data):
        prompt = tokenizer.apply_chat_template(sample["messages"][:-1], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)
        response = tokenizer.batch_decode(outputs)[0]
        
        # Extract assistant response
        assistant_resp = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        
        pred_json = extract_predicted_json(assistant_resp)
        pred_label = pred_json.get("label", "ERROR") if pred_json else "ERROR"
        
        true_json = extract_predicted_json(sample["messages"][-1]["content"])
        true_label = true_json.get("label", "ERROR") if true_json else "ERROR"
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nLayer 1 Classification Report:")
    print(classification_report(y_true, y_pred, labels=["SAFE", "SUSPICIOUS", "UNSAFE"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=["SAFE", "SUSPICIOUS", "UNSAFE"]))


def evaluate_layer2(model, tokenizer, test_data):
    print("Evaluating Layer 2 (Specialist)...")
    y_true_action = []
    y_pred_action = []
    
    for sample in tqdm(test_data):
        prompt = tokenizer.apply_chat_template(sample["messages"][:-1], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens=500, use_cache=True)
        response = tokenizer.batch_decode(outputs)[0]
        
        assistant_resp = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        
        pred_json = extract_predicted_json(assistant_resp)
        pred_action = pred_json.get("action", "ERROR") if pred_json else "ERROR"
        
        true_json = extract_predicted_json(sample["messages"][-1]["content"])
        true_action = true_json.get("action", "ERROR") if true_json else "ERROR"
        
        y_true_action.append(true_action)
        y_pred_action.append(pred_action)
        
    from sklearn.metrics import classification_report
    print("\nLayer 2 Action Determination Report:")
    print(classification_report(y_true_action, y_pred_action, labels=["REDACT", "REWRITE", "BLOCK", "ALLOW"]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=["1", "2"], required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (e.g., checkpoints/final)")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test.jsonl")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    print(f"Loading test data from {args.test_data}...")
    test_data = load_test_data(args.test_data)
    
    if args.layer == "1":
        evaluate_layer1(model, tokenizer, test_data)
    else:
        evaluate_layer2(model, tokenizer, test_data)

if __name__ == "__main__":
    main()
