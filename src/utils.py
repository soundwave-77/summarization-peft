import json
import os

from datasets import load_dataset


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def ensure_unique_dir(path: str) -> str:
    if os.path.isdir(path):
        raise FileExistsError(f"Directory '{path}' already exists.")
    os.makedirs(path, exist_ok=True)
    return path

def load_datasets(train_path: str, val_path: str, test_path: str):
    return load_dataset(
        "json",
        data_files={"train": train_path, "validation": val_path, "test": test_path}
    )

def save_metrics(metrics: dict, metrics_dir: str, filename: str):
    file_path = os.path.join(metrics_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)