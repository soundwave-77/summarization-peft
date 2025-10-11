import logging
import os

import evaluate
import numpy as np
import torch

from clearml import Task
from datasets import load_dataset
from peft import get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.adapters.factory import AdapterFactory
from src.utils import check_path_existence, load_config, save_dict_to_json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    logger.info("### LOAD CONFIGS...")
    train_config = load_config("src/configs/train_config.json")
    model_path = train_config["base_model"]
    model_name = model_path.split("/")[-1]
    adapter_name = train_config["adapter_name"]
    train_params = train_config["training_params"]
    artifacts_dir = train_config["artifacts_dir"]
    logger.info("### CONFIGS SUCCESSFULLY LOADED!")

    logger.info("### CREATE ADAPTER...")
    adapter_factory = AdapterFactory()
    adapter = adapter_factory.create_adapter(adapter_name)
    experiment_name = f"{model_name}_{adapter_factory.experiment_name()}"
    experiment_dir = os.path.join(artifacts_dir, experiment_name)
    check_path_existence(experiment_dir)
    logger.info("### ADAPTER CREATED!")

    logger.info("### CREATE CLEARML TASK...")
    task = Task.init(
        project_name="summarization-peft",
        task_name=experiment_name,
        task_type=Task.TaskTypes.training,
    )
    logger.info("### CLEARML TASK CREATED!")

    logger.info("### LOAD METRICS...")
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")
    logger.info("### METRICS LOADED!")

    logger.info("### LOAD DATASETS...")
    data = load_dataset("rcp-meetings/rudialogsum_v2").rename_column("dialog", "text")
    train_valid = data["train"].train_test_split(test_size=0.2, seed=42, shuffle=True)
    data["train"] = train_valid["train"]
    data["validation"] = train_valid["test"]
    logger.info("### DATASETS LOADED!")

    logger.info("### LOAD TOKENIZER AND MODEL...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model = get_peft_model(model, adapter)
    model.print_trainable_parameters()
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    logger.info("### MODEL AND TOKENIZER LOADED!")

    prefix = "Суммаризируй следующий диалог и дай краткий ответ на русском языке:\n\n {dialog}"

    def preprocess_function(examples):
        inputs = [prefix.format(dialog=dialog) for dialog in examples["text"]]
        tokenized_inputs = tokenizer(
            inputs,
            truncation=True,
            max_length=512,
        )

        labels = tokenizer(
            examples["summary"],
            truncation=True,
            max_length=512,
        )["input_ids"]

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_data = data.map(
        preprocess_function,
        batched=True,
        remove_columns=data["train"].column_names,
    )

    logger.info("### DATA TOKENIZED!")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu_score = bleu.compute(
            predictions=decoded_preds,
            references=[[ref] for ref in decoded_labels],
        )
        rouge_score = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        meteor_score = meteor.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )
        bert = bertscore.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            lang="ru",
        )

        return {
            "bleu": bleu_score["bleu"],
            "rouge1": rouge_score["rouge1"],
            "rouge2": rouge_score["rouge2"],
            "rougeL": rouge_score["rougeL"],
            "meteor": meteor_score["meteor"],
            "bertscore_f1": float(np.mean(bert["f1"])),
        }

    outputs_dir = os.path.join(experiment_dir, "outputs")

    num_training_examples = len(data["train"])
    per_device_train_batch_size = train_params["per_device_train_batch_size"]
    num_train_epochs = train_params["num_train_epochs"]
    steps_per_epoch = num_training_examples // per_device_train_batch_size
    total_steps = steps_per_epoch * num_train_epochs
    eval_steps = int(total_steps * 0.05)

    train_params["eval_steps"] = eval_steps
    train_params["save_steps"] = eval_steps
    train_params["logging_steps"] = eval_steps
    train_params["output_dir"] = outputs_dir

    train_params_path = save_dict_to_json(train_params, experiment_dir, "train_params.json")
    task.upload_artifact("train_params", train_params_path)

    training_args = Seq2SeqTrainingArguments(**train_params)

    logger.info("### INITIALIZE TRAINER...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    logger.info("### TRAINER INITIALIZED!")

    logger.info("### START TRAINING...")
    trainer.train()
    logger.info("### TRAINING COMPLETED!")

    logger.info("### EVALUATE ON VALIDATION DATASET...")
    val_metrics = trainer.evaluate()
    logger.info(f"Validation metrics: {val_metrics}")

    for k, v in val_metrics.items():
        task.get_logger().report_scalar("Validation Metrics", k, v)

    logger.info("### EVALUATE ON TEST DATASET...")
    test_metrics = trainer.predict(tokenized_data["test"]).metrics
    logger.info(f"Test metrics: {test_metrics}")

    for k, v in test_metrics.items():
        task.get_logger().report_scalar("Test Metrics", k, v)

    save_dict_to_json(val_metrics, experiment_dir, "validation_metrics.json")
    save_dict_to_json(test_metrics, experiment_dir, "test_metrics.json")

    logger.info("### METRICS SAVED AND UPLOADED TO CLEARML SUCCESSFULLY!")
