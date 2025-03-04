from fastapi import FastAPI, Body
import tensorflow as tf
import numpy as np
import pickle
import uvicorn
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)

# Lade das Modell
model = tf.keras.models.load_model('models/toxicity.keras')

# Laden des vectorizers
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


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

if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=9000)