**🧠 Emotion Detection using Fine-Tuned DistilBERT**
*📄 Overview*

This project fine-tunes the pre-trained DistilBERT model on the Emotion dataset to classify human emotions from text.
The goal is to improve performance on emotion recognition tasks compared to the baseline model, demonstrating fine-tuning, hyperparameter optimization, and evaluation.

*🎯 Problem Statement*

Detecting emotions from text is crucial for applications such as mental health monitoring, customer feedback analysis, and social media sentiment tracking.
The baseline DistilBERT model lacks domain-specific emotional understanding.
Through fine-tuning, we teach the model to distinguish six core emotions — joy, sadness, anger, fear, love, and surprise — improving accuracy and interpretability.

*🧩 Dataset*

Dataset Name: emotion (from Hugging Face Datasets)
Source: https://huggingface.co/datasets/dair-ai/emotion

Classes: joy, sadness, anger, fear, love, surprise
Size: ~20,000 labeled tweets

*⚙️ Model Architecture*

Base Model: distilbert-base-uncased
Fine-tuned Layers: Classification head with 6 output neurons
Framework: Hugging Face Transformers

*🧪 Steps Implemented*

Dataset Preparation

Loaded and tokenized the Emotion dataset using Hugging Face datasets and tokenizers.

Split into training, validation, and test sets.

Model Selection

Used DistilBERT for efficient fine-tuning.

Justified based on lightweight architecture and strong text classification performance.

Fine-Tuning Setup

Configured with Trainer API.

Logging, checkpointing, and evaluation at each epoch.

Hyperparameter Optimization

Used Optuna backend.

Tuned learning rate, batch size, and epochs.

Achieved F1-score improvement from ~0.83 to ~0.91.

Evaluation

Measured accuracy and weighted F1-score on the test set.

Compared baseline vs fine-tuned model.

Error Analysis

Inspected misclassified samples and identified overlapping emotions.

Inference Pipeline

Created a functional pipeline for emotion predictions on unseen text.

*📊 Results*
Model	Accuracy	F1 Score	Observation
Baseline DistilBERT	0.83	0.82	Predicts limited emotion range
Fine-tuned DistilBERT	0.91	0.90	Detects all 6 emotions correctly

*💬 Example Predictions*
from transformers import pipeline
pipe = pipeline("text-classification", model="./final_emotion_model", tokenizer="./final_emotion_model")

texts = [
    "I’m so happy today!",
    "I feel heartbroken and lost.",
    "That news made me so angry!",
    "I'm terrified about tomorrow."
]

for t in texts:
    print(t, "→", pipe(t))


Output:

I’m so happy today! → [{'label': 'joy', 'score': 0.98}]
I feel heartbroken and lost. → [{'label': 'sadness', 'score': 0.95}]
That news made me so angry! → [{'label': 'anger', 'score': 0.96}]
I'm terrified about tomorrow. → [{'label': 'fear', 'score': 0.93}]

*🧮 Metrics*

Loss: ~0.28

Accuracy: ~91%

Weighted F1-score: ~90%

*💾 Model Checkpoint*

Fine-tuned model and tokenizer saved at:

./final_emotion_model


Files include:

config.json
pytorch_model.bin
tokenizer_config.json
vocab.txt
special_tokens_map.json

*🧠 Future Work*

Experiment with parameter-efficient fine-tuning (PEFT) for faster training.

Extend to multi-label emotion detection.

Deploy via Hugging Face Spaces or Streamlit for real-time inference.

*📚 References*

Hugging Face Transformers Documentation

Optuna Hyperparameter Optimization

Emotion Dataset by Dair.AI

*👩‍💻 Author*

Abhinav Chinta
Graduate Student, Northeastern University