import json
import os


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def check_path_existence(path: str) -> None:
    if os.path.exists(path):
        raise FileExistsError(f"Directory `{path}` already exists.")
    os.makedirs(path)

def save_metrics(metrics: dict, metrics_dir: str, filename: str):
    file_path = os.path.join(metrics_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)