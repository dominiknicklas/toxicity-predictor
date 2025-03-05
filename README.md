# Toxic Comment Classification API

This project uses a neural network to classify toxic comments into six different categories. The API is built with FastAPI and can provide predictions for text inputs.

## Features
Classifies comments into: Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate

Uses a Bidirectional LSTM model for text analysis.

Provides a REST API for predictions.

## Model Training and Testing

**Dataset:** The model is trained on the Jigsaw Toxic Comment Classification Challenge Dataset, which contains user-generated comments labeled for different types of toxicity.

**Training Process:**
- The text data is vectorized using TextVectorization from TensorFlow.
- The dataset is split into 70% training, 20% validation, and 10% testing.
- A Bidirectional LSTM model with multiple dense layers is trained for 10 epochs using the Adam optimizer and binary cross-entropy loss.
- The trained model is saved as toxicity.keras, and the vectorizer is stored as vectorizer.pkl.

**Evaluation**

The model was tested on a held-out test set, and the following metrics were calculated:
- Precision: 0.9225
- Recall: 0.9156
- AUC: 0.9986

## Installation

1. Clone the Repository

```
git clone https://github.com/dominik-nicklas/toxicity-predictor.git
cd toxicity-predictor
```

2. Build Docker Image (automaticlly installs everything) and start the container

```
docker build -t toxic-api .
docker run -p 9000:9000 toxic-api
```
