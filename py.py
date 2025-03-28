import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Data Preprocessing
class GlaucomaQASystem:
    def __init__(self, data_path):
        # Load and clean the dataset
        self.df = pd.read_csv(data_path)
        
        # Data Cleaning
        self.df.dropna(inplace=True)  # Remove rows with missing values
        self.df.drop_duplicates(inplace=True)  # Remove duplicate entries
        
        # Preprocessing
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.label_encoder = LabelEncoder()
        
        # Prepare features and target
        self.X = self.df['question']
        self.y = self.label_encoder.fit_transform(self.df['answer'])
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
    
    def train_model(self):
        # Create a pipeline with TF-IDF vectorization and Naive Bayes classifier
        self.model = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', MultinomialNB())
        ])
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate_model(self):
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Print evaluation metrics
        print("Model Evaluation:")
        print(classification_report(self.y_test, y_pred, 
                                    target_names=self.label_encoder.classes_))
    
    def answer_question(self, question):
        # Predict the most likely answer
        prediction = self.model.predict([question])
        answer_index = prediction[0]
        return self.label_encoder.inverse_transform([answer_index])[0]

# Example Usage
def main():
    # Initialize the QA System
    qa_system = GlaucomaQASystem('glaucoma_data.csv')
    
    # Train the model
    qa_system.train_model()
    
    # Evaluate the model
    qa_system.evaluate_model()
    
    # Example Interactions
    example_questions = [
        "What is Glaucoma?",
        "Who is at risk for Glaucoma?",
        "How to prevent Glaucoma?"
    ]
    
    print("\nExample Interactions:")
    for question in example_questions:
        answer = qa_system.answer_question(question)
        print(f"Q: {question}\nA: {answer}\n")

if __name__ == "__main__":
    main()