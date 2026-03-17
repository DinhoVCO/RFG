from beir import util
from beir.retrieval.evaluation import EvaluateRetrieval
from kdir_src.dataset.beir import get_beir_dataset
import os
import json

def evaluate_results(qrels, results):
    retriever = EvaluateRetrieval()
    metrics = retriever.evaluate(qrels, results, k_values=[1, 3, 5, 10])
    return metrics

#../results/experimento/dataset/
def evaluate_all_results(results_path):
    normalized_path1 = results_path.strip(os.sep)
    parts = results_path.split(os.sep)
    experiment = parts[-2]
    dataset = parts[-1]
    print(dataset)
    corpus, queries, qrels = get_beir_dataset(dataset)
    output_base_dir = os.path.join("..", "metrics", experiment, dataset)
    os.makedirs(output_base_dir, exist_ok=True)
    for filename in os.listdir(results_path):
        file_path = os.path.join(results_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".json"):
            print(f"Calculating metrics for {filename}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            metrics = evaluate_results(qrels, data)
            output_file_path = os.path.join(output_base_dir, filename)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)
            print(f"  Metrics saved in: {output_file_path}")

def evaluate_all_results_v2(path_save_metrics,results_path):
    normalized_path = results_path.rstrip(os.sep)
    parts = [p for p in normalized_path.split(os.sep) if p]
    experiment = parts[-2]
    dataset = parts[-1]
    print(dataset)
    corpus, queries, qrels = get_beir_dataset(dataset)
    output_base_dir = os.path.join("..", "metrics",path_save_metrics, experiment, dataset)
    os.makedirs(output_base_dir, exist_ok=True)
    for filename in os.listdir(normalized_path):
        file_path = os.path.join(normalized_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".json"):
            print(f"Calculating metrics for {filename}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            metrics = evaluate_results(qrels, data)
            output_file_path = os.path.join(output_base_dir, filename)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)
            print(f"  Metrics saved in: {output_file_path}")