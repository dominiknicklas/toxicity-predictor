from fastapi import FastAPI, Body
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import numpy as np
import json
import pandas as pd

# Lade das Modell
model = tf.keras.models.load_model('toxicity.keras')

# Laden des vectorizers
with open('text_vectorizer_config.json', 'r') as f:
    vectorizer_config = json.load(f)

# Erstelle den TextVectorizer mit der geladenen Konfiguration
vectorizer = TextVectorization.from_config(vectorizer_config)

# Lade die Gewichtung des Vektorisierers (optional)
weights = np.load('vectorizer_weights.npy', allow_pickle=True)
vectorizer.set_weights(weights)

df = pd.read_csv('train.csv')
x = df['comment_text']
vectorizer.adapt(x.values)

app = FastAPI()


# @app.on_event("startup")
# async def startup_event():
#     # Hier könnte der Vektorisierer mit den gleichen Daten angepasst werden,
#     # falls du neue Daten hast oder der Vektorisierer nicht wie erwartet funktioniert.
#     # Angenommen, du hast immer noch die Daten `X` (Trainingsdaten):
#     df = pd.read_csv('train.csv')
#     x = df['comment_text']
#     vectorizer.adapt(x.values)


@app.post("/predict/")
async def predict(input_data: str = Body(...)):
    # Die Eingabedaten vorbereiten
    input_str = vectorizer(input_data)
    # Vorhersage mit dem Modell machen
    res = model.predict(np.expand_dims(input_str, 0))
    return {"toxic": res.tolist()[0][0],
            "severe_toxic": res.tolist()[0][1],
            "obscene": res.tolist()[0][2],
            "threat": res.tolist()[0][3],
            "insult": res.tolist()[0][4],
            "identity_hate": res.tolist()[0][5],
            }
