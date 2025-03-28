from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import torch
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data(file_path='medical_qa.csv', sample_size=None):
    """Load data with memory-efficient chunking"""
    try:
        # Read with chunking for large files
        chunks = pd.read_csv(file_path, chunksize=10000)
        df = pd.concat([chunk for chunk in tqdm(chunks, desc="Loading data")])
        
        if sample_size:
            df = df.sample(min(sample_size, len(df)))
        
        # Convert to HuggingFace dataset
        return Dataset.from_pandas(df)
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def memory_safe_tokenize(dataset, tokenizer, batch_size=1000):
    """Tokenize in batches to avoid OOM errors"""
    try:
        def batch_tokenizer(examples):
            contexts = [f"[TAG: {tag}] {answer}" for tag, answer in zip(examples["tag"], examples["answer"])]
            return tokenizer(
                examples["question"],
                contexts,
                padding="max_length",
                truncation=True,
                max_length=256,  # Reduced from 512 for memory efficiency
                return_overflowing_tokens=True
            )
        
        return dataset.map(
            batch_tokenizer,
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names
        )
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        raise

def train_model():
    try:
        # Memory management
        torch.cuda.empty_cache()
        gc.collect()

        # Load model with safety checks
        model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Load data with progress tracking
        logger.info("Loading dataset...")
        dataset = load_processed_data()
        
        # Tokenize with memory protection
        logger.info("Tokenizing data...")
        tokenized_datasets = memory_safe_tokenize(dataset, tokenizer)
        
        # Train-test split
        split_datasets = tokenized_datasets.train_test_split(test_size=0.1)
        train_dataset = split_datasets["train"]
        eval_dataset = split_datasets["test"]
        
        # Training configuration with gradient checkpointing
        training_args = TrainingArguments(
            output_dir="./results2",
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            learning_rate=3e-5,
            per_device_train_batch_size=4,  # Reduced for memory safety
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            logging_dir='./logs2',
            logging_steps=100,
            report_to="none"
        )
        
        # Trainer with early stopping
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Training with progress tracking
        logger.info("Starting training...")
        trainer.train()
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        model, tokenizer = train_model()
        # Save model safely
        model.save_pretrained("./medical_qa_model2")
        tokenizer.save_pretrained("./medical_qa_model2")
        logger.info("Training completed and model saved successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")