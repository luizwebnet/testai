import pandas as pd
import re

#Extend panda class. Called custom pandas accessor
@pd.api.extensions.register_series_accessor("clean")
class TextCleanerAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    
    def do(self):
        """
        Custom method similar to prototype in JavaScript
        Can be called as df['column'].clean.do()
        - Deletes non-ASCII, removes leading/trailing whitespace, breakline, tab and quotes)
        """
        return (
            self._obj
            .str.strip()
            .str.replace(r'[^\x00-\x7F]', '', regex=True)
            .str.replace(r'[\n\r\t\s]+', ' ', regex=True)            
            .str.replace(r'\s+([?.!,])', r'\1', regex=True)
            .str.replace(r'^["\'\u201C\u201D\u2018\u2019]|["\'\u201C\u201D\u2018\u2019]$', '', regex=True)
            .fillna('')
        )

def extract_illness_name(question):
    # Updated regex to handle multiple question formats
    patterns = [
        r'\(are\)\s(.*?)\s?\?',  # "(are) High Blood Pressure ?"
        r'What is\s(.*?)\?',     # "What is High Blood Pressure?"
        r'How to prevent\s(.*?)\?',  # "How to prevent High Blood Pressure?"
        r'How to diagnose\s(.*?)\?',  # "How to diagnose High Blood Pressure?"
        r'What are the treatments for\s(.*?)\?',  # "What are the treatments for High Blood Pressure?"
        r'Who is at risk for\s(.*?)\?',  # "Who is at risk for Urinary Tract Infections?"
        r'Where to find support for people with\s(.*?)\?',  # "Where to find support for people with Alcohol Use and Older Adults?"
        r'What causes\s(.*?)\?',  # "What causes Diabetes?"
        r'How many people are affected by\s(.*?)\?'  # "How many people are affected by Anxiety Disorders?"
        r'What I need to know about Living with\s(.*?)\?'  # "What I need to know about Living with Kidney Failure"
        r'What I need to know about\s(.*?)\?'  # "What I need to know about Diarrhea"
        r'Your Guide to\s(.*?)\?'  # "Your Guide to Diabetes: Type 1 and Type 2"
        r'the outlook for\s(.*?)\?'  # "the outlook for Kidney Dysplasia"
        r'What to do for Treatment Methods for\s(.*?)\?'  # "What to do for Treatment Methods for Kidney Failure: Peritoneal Dialysis?"
        r'How to diagnose\s(.*?)\?'  # "How to diagnose Primary Hyperparathyroidism?"
        r'What are the treatments for Primary\s(.*?)\?'  # "What are the treatments for Primary Hyperparathyroidism?"
        r'What to do for Primary\s(.*?)\?'  # "What to do for Primary Hyperparathyroidism?"
        r'What to do for\s(.*?)\?'  # "What to do for Primary Hyperparathyroidism?"                
        r'Do you have information about\s(.*?)\?'  # "Do you have information about Radiation Therapy"                
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return ""


def preprocess_data(input_file='mle_screening_dataset.csv', output_file='medical_qa.csv'):
    # Load data
    df = pd.read_csv(input_file)
    
    # Clean data
    df = (
        df
        .dropna()  # Remove rows with missing values
        .drop_duplicates()  # Remove duplicate rows
        .assign( # Sanitize content with pandas accessor
            question=lambda x: x['question'].clean.do(),
            answer=lambda x: x['answer'].clean.do(),
        )
    )

    #Add Feature TAG    
    df['tag'] = df['question'].apply(extract_illness_name)  
    
    # Save processed data
    df.to_csv(output_file, index=False)
    return df

if __name__ == "__main__":
    preprocess_data()