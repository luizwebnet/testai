# Medical Assistant Bot

A question-answering system for medical diseases.

## Approach

1. **Data Preprocessing**: 
   - Cleaned the dataset
   - Extracted disease tags from questions
   - Formatted data for QA training

2. **Model Training**:
   - Fine-tuned the pre-trained model on our dataset
   - Used standard QA training approach with Hugging Face Transformers

## Assumptions

- Limited dataset size, so focused on fine-tuning rather than full training
- Questions follow a consistent pattern (e.g., "What is...", "How to...")
- Answers are contained within the provided context

## Performance

- Strengths: Good at extracting relevant information
- Weaknesses: Limited by original dataset size and diversity

## Improvements

- Need more training - "epoch"
- Add more diverse medical questions
- Add confidence scoring for answers
- Deploy as an API service"# testai" 
