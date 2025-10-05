import logging
import os

import evaluate
import numpy as np

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.adapters.factory import AdapterFactory
from src.utils import load_config, save_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    load_dotenv("../.env")

    logger.info("### LOAD CONFIGS...")
    
    train_config = load_config("src/configs/train_config.json")
    model_name = train_config["base_model"]
    adapter_name = train_config["adapter_name"]
    data_config = train_config["data"]
    train_params = train_config["training_params"]
    artifacts_dir = train_config["artifacts_dir"]
    
    logger.info("### CONFIGS SUCCESSFULLY LOADED!")

    logger.info("### CREATE ADAPTER...")
    
    adapter_factory = AdapterFactory()
    adapter = adapter_factory.create_adapter(adapter_name)
    experiment_name = f"{model_name}_{adapter_factory.experiment_name()}"
    experiment_dir = os.path.join(artifacts_dir, experiment_name)
    
    if os.path.exists(experiment_dir):
        raise FileExistsError(f"Duplicated experiment name {experiment_dir}")
    os.makedirs(experiment_dir)
    
    logger.info("### ADAPTER CREATED!")

    logger.info("### LOAD METRICS...")
    
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")
    
    logger.info("### METRICS LOADED!")

    logger.info("### LOAD DATASETS...")

    data = load_dataset(
        "json",
        data_files={
            "train": data_config["train"], 
            "validation": data_config["val"], 
            "test": data_config["test"]
        }
    )

    logger.info("### DATASETS LOADED!")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple) or preds.ndim == 2:
            preds = np.argmax(preds, axis=-1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu = bleu_metric.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])
        rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
        bertscore_metric = float(
            bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="ru")["f1"].mean()
        )
        return {
            "bleu": bleu["bleu"],
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "meteor": meteor["meteor"],
            "bertscore_f1": bertscore_metric
        }

    outputs_dir = os.path.join(experiment_dir, "outputs")
    sft_config = SFTConfig(
        run_name=experiment_name,
        output_dir=outputs_dir,
        **train_params
    )

    logger.info("### INITIALIZE TRAINER...")
    
    trainer = SFTTrainer(
        model=model_name,
        peft_config=adapter,
        args=sft_config,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        compute_metrics=compute_metrics
    )
    
    logger.info("### TRAINER INITIALIZED!")

    logger.info("### START TRAINING...")
    trainer.train()
    logger.info("### TRAINING COMPLETED!")

    logger.info("### EVALUATE ON VALIDATION DATASET...")
    val_metrics = trainer.evaluate()
    logger.info(f"Validation metrics: {val_metrics}")

    logger.info("### EVALUATE ON TEST DATASET...")
    test_metrics = trainer.predict(data["test"]).metrics
    logger.info(f"Test metrics: {test_metrics}")

    logger.info("### SAVE METRICS...")
    
    save_metrics(val_metrics, experiment_dir, "validation_metrics.json")
    save_metrics(test_metrics, experiment_dir, "test_metrics.json")
    
    logger.info("### METRICS SAVED SUCCESSFULLY!")
