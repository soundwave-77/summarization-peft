import json
import os


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def check_path_existence(path: str) -> None:
    if os.path.exists(path):
        raise FileExistsError(f"Directory `{path}` already exists.")
    os.makedirs(path)

def save_dict_to_json(metrics: dict, metrics_dir: str, filename: str) -> str:
    file_path = os.path.join(metrics_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    
    return file_path

def print_number_of_trainable_model_parameters(model) -> str:
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\
            all model parameters: {all_model_params}\
            percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"