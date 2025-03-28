# Using BERT for medical question answering
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
#import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Dropout, Concatenate
from transformers import AutoTokenizer, TFAutoModel


# Load the dataset
df = pd.read_csv('medical_qa.csv')

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    df['question'], df[['answer', 'tag']], test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Load pre-trained BERT model and tokenizer
#The model choosed is better for Question answering tasks (especially SQuAD-like tasks) as it's specifically fine-tuned for this purpose
#tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
#bert_model = TFBertModel.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
bert_model = TFBertModel.from_pretrained("distilbert-base-uncased-distilled-squad")


# Prepare data for BERT
def encode_sentences(sentences, max_length=128):
    input_ids = []
    attention_masks = []

    for sentence in sentences:
        encoded = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )

        input_ids.append(encoded['input_ids'][0])
        attention_masks.append(encoded['attention_mask'][0])

    return tf.stack(input_ids), tf.stack(attention_masks)

# Encode the questions
train_input_ids, train_attention_masks = encode_sentences(X_train.tolist())
val_input_ids, val_attention_masks = encode_sentences(X_val.tolist())
test_input_ids, test_attention_masks = encode_sentences(X_test.tolist())

# Prepare tag labels (for classification)
# Get unique tags and create a mapping
unique_tags = df['tag'].unique()
tag_to_id = {tag: idx for idx, tag in enumerate(unique_tags)}
id_to_tag = {idx: tag for idx, tag in enumerate(unique_tags)}

# Convert tags to numerical values
train_tags = np.array([tag_to_id[tag] for tag in y_train['tag']])
val_tags = np.array([tag_to_id[tag] for tag in y_val['tag']])
test_tags = np.array([tag_to_id[tag] for tag in y_test['tag']])

# Convert to one-hot encoding
train_tags_one_hot = tf.keras.utils.to_categorical(train_tags, num_classes=len(unique_tags))
val_tags_one_hot = tf.keras.utils.to_categorical(val_tags, num_classes=len(unique_tags))
test_tags_one_hot = tf.keras.utils.to_categorical(test_tags, num_classes=len(unique_tags))

# Build the model
# Build the model
def build_model(bert_model, num_classes):
    # Create the complete model
    input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')

    # Create a custom layer to handle the BERT model call
    class BertLayer(tf.keras.layers.Layer):
        def __init__(self, bert):
            super(BertLayer, self).__init__()
            self.bert = bert

        def call(self, inputs):
            input_ids, attention_mask = inputs
            # Convert Keras tensors to TF tensors if needed
            return self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]

    # Use the custom layer
    sequence_output = BertLayer(bert_model)([input_ids, attention_mask])
    cls_token = sequence_output[:, 0, :]

    # Classification layers
    x = Dropout(0.2)(cls_token)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[input_ids, attention_mask], outputs=outputs)

    # Freeze BERT layers
    bert_model.trainable = False

    return model

# Create the model
model = build_model(bert_model, len(unique_tags))

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train the model
history = model.fit(
    [train_input_ids, train_attention_masks],
    train_tags_one_hot,
    validation_data=([val_input_ids, val_attention_masks], val_tags_one_hot),
    epochs=3,
    batch_size=8
)

# Save the model
model.save('medical_qa_model.keras')

# Create a mapping from question to answer
question_to_answer = dict(zip(df['question'], df['answer']))