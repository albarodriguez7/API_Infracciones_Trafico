from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
import numpy as np



os.chdir(os.path.dirname(__file__))

app = Flask(__name__)


# Cargar el modelo

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


# Enruta la landing page (endpoint /)

@app.route('/', methods = ["GET"])
def hello(): 
    return "Bienvenido a mi API del modelo advertising"



# Enruta la funcion al endpoint /api/v1/predict

@app.route('/api/v1/predict', methods=["GET"])
def predict():
    args = {k.lower(): v for k, v in request.args.items()}

    
    sexo = args.get('sexo')
    novel = args.get('novel')
    edad = args.get('edad')
    num_infracciones = args.get('num_infracciones')  

   
    def to_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan

    sexo = to_float(sexo)
    novel = to_float(novel)
    edad = to_float(edad)
    num_infracciones = to_float(num_infracciones)

    # Comprobaciones
    errors = []

    if not np.isnan(sexo) and sexo not in (0.0, 1.0):
        errors.append("SEXO debe ser 0 o 1")

    if not np.isnan(novel) and novel not in (0.0, 1.0):
        errors.append("NOVEL debe ser 0 o 1")

    if not np.isnan(edad) and edad not in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        errors.append("EDAD debe ser un número entre 1 y 6")

    if errors:
        return jsonify({'error': errors}), 400

    # 🔹 Detectar missing correctamente
    missing = [
        name for name, val in [
            ('SEXO', sexo),
            ('NOVEL', novel),
            ('EDAD', edad),
            ('NUM_INFRACCIONES', num_infracciones)
        ]
        if np.isnan(val)
    ]

    if missing:
        return jsonify({
            "error": "Faltan variables obligatorias",  # No puede predecir si hay missing
            "missing": missing
        }), 400


    # Crear DataFrame
    input_data = pd.DataFrame({
        'SEXO': [sexo],
        'NOVEL': [novel],
        'EDAD': [edad],
        'NUM_INFRACCIONES': [num_infracciones]
    })

    prediction = model.predict(input_data)

    # Convertir a tipo Python (MUY IMPORTANTE)
    response = {
        "prediction": int(prediction[0]),
        "missing": missing
    }

    return jsonify(response)



if __name__ == '__main__':
    app.run(debug=True)
