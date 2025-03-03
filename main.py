from fastapi import FastAPI, Body
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd

# Lade das Modell
model = tf.keras.models.load_model('toxicity.keras')

# Laden des vectorizers
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

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
