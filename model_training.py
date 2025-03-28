from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    get_scheduler
)
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, DatasetDict
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path='medical_qa.csv'):
    """Load and preprocess data with robust error handling"""
    try:
        # Load with pandas for better control
        df = pd.read_csv(file_path)
        
        # Convert to HF dataset
        dataset = DatasetDict({
            'train': load_dataset('csv', data_files=file_path)['train']
        })
        return dataset
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise

def tokenize_function(examples, tokenizer):
    """Improved tokenization for T5 with error handling"""
    try:
        inputs = tokenizer(
            [f"question: {q} context: {a}" for q, a in zip(examples["question"], examples["answer"])],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["answer"],
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            
        inputs["labels"] = labels
        return inputs
    except Exception as e:
        logger.error(f"Tokenization error: {e}")
        raise

def compute_metrics(eval_pred):
    """Simplified metrics for generation tasks"""
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions.argmax(axis=-1), skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Simple exact match accuracy
    acc = np.mean([1 if p == l else 0 for p, l in zip(decoded_preds, decoded_labels)])
    return {"accuracy": acc}

def train_model():
    try:
        # Initialize model and tokenizer
        model_name = "t5-small"  # Using smaller model for efficiency
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        device = torch.device("cpu")
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        
        # LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"]
        )
        model = get_peft_model(model, peft_config)
        
        # Load and preprocess data
        dataset = load_and_preprocess_data()
        tokenized_datasets = dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        # Train/val split
        split_datasets = tokenized_datasets["train"].train_test_split(test_size=0.2)
        
        # Training setup
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir="./logs",
            report_to="none",
            fp16=False,
            no_cuda=True,
            gradient_accumulation_steps=4
        )
        #fp16=torch.cuda.is_available(),
        optimizer = AdamW(model.parameters(), lr=3e-4)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split_datasets["train"],
            eval_dataset=split_datasets["test"],
            optimizers=(optimizer, get_scheduler(
                "linear",
                optimizer,
                num_warmup_steps=100,
                num_training_steps=len(split_datasets["train"]) * training_args.num_train_epochs
            )),
            compute_metrics=compute_metrics
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save
        model.save_pretrained("./medical_qa_model")
        tokenizer.save_pretrained("./medical_qa_model")
        logger.info("Training completed successfully")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    model, tokenizer = train_model()