import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_medical_dataset(file_path, sample_size=None):
    """
    Load and preprocess medical QA dataset

    Args:
        file_path (str): Path to the CSV file
        sample_size (int, optional): Number of samples to use

    Returns:
        Dataset: Processed HuggingFace dataset
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Optional sampling
        if sample_size:
            df = df.sample(min(sample_size, len(df)))

        # Convert to HuggingFace dataset
        return Dataset.from_pandas(df)

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def prepare_features(examples, tokenizer, max_length=384):
    """
    Prepare features for question answering

    Args:
        examples (dict): Batch of examples
        tokenizer (PreTrainedTokenizer): Tokenizer to use
        max_length (int): Maximum sequence length

    Returns:
        dict: Tokenized features
    """
    # Combine context with tag
    contexts = [f"[TAG: {tag}] {answer}" for tag, answer in zip(examples["tag"], examples["answer"])]

    # Tokenize questions and contexts
    tokenized_examples = tokenizer(
        examples["question"],
        contexts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

    return tokenized_examples

def train_medical_qa_model(dataset_path, output_dir='./medical_qa_model'):
    """
    Train a medical question-answering model

    Args:
        dataset_path (str): Path to the medical QA dataset
        output_dir (str): Directory to save the trained model

    Returns:
        tuple: Trained model and tokenizer
    """
    try:
        # Clear GPU cache
        torch.cuda.empty_cache()

        # Load pretrained model and tokenizer
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        # Load and preprocess dataset
        logger.info("Loading dataset...")
        dataset = load_medical_dataset(dataset_path)

        # Split dataset
        dataset = dataset.train_test_split(test_size=0.2)
        train_dataset = dataset['train']
        eval_dataset = dataset['test']

        # Tokenize datasets
        train_dataset = train_dataset.map(
            lambda examples: prepare_features(examples, tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names
        )

        eval_dataset = eval_dataset.map(
            lambda examples: prepare_features(examples, tokenizer),
            batched=True,
            remove_columns=eval_dataset.column_names
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # Train the model
        logger.info("Starting training...")
        trainer.train()

        # Save model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"Model saved to {output_dir}")

        return model, tokenizer

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def main():
    try:
        # Path to your medical QA dataset
        dataset_path = 'medical_qa.csv'

        # Train the model
        model, tokenizer = train_medical_qa_model(dataset_path)

        # Optional: Inference example
        sample_question = "What are the symptoms of diabetes?"
        inputs = tokenizer(sample_question, "[TAG: Diabetes] Long-term symptoms include...", return_tensors="pt")

        logger.info("Model training completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()