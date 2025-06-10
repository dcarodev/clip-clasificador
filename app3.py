from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from deep_translator import GoogleTranslator
import json
import random
import os

app = Flask(__name__)

# Cargamos modelo CLIP (una sola vez)
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Cargamos las palabras en español
with open("palabras.json", "r", encoding="utf-8") as f:
    palabras_es = json.load(f)

@app.route("/check", methods=["POST"])
def check():
    palabra_objetivo = request.form.get("palabra", "").strip().lower()
    imagen_file = request.files.get("foto")

    if not palabra_objetivo or not imagen_file:
        return jsonify({"error": "Faltan parámetros 'palabra' o 'foto'."}), 400

    if palabra_objetivo not in palabras_es:
        return jsonify({"error": f"La palabra '{palabra_objetivo}' no está en la lista."}), 400

    try:
        # Seleccionamos un subconjunto aleatorio (máximo 10 palabras)
        max_palabras = 10
        palabras_sel_es = random.sample(palabras_es, min(len(palabras_es), max_palabras))

        # Agregamos palabra objetivo si no está en el grupo seleccionado
        if palabra_objetivo not in palabras_sel_es:
            palabras_sel_es[0] = palabra_objetivo

        # Traducimos las palabras seleccionadas (a demanda)
        palabras_sel_en = [GoogleTranslator(source='es', target='en').translate(p) for p in palabras_sel_es]

        # Procesamos imagen
        imagen = Image.open(imagen_file).convert("RGB")
        inputs = processor(text=palabras_sel_en, images=imagen, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).squeeze().tolist()

        # Encontramos la palabra detectada con mayor probabilidad
        idx_max = probs.index(max(probs))
        palabra_detectada_es = palabras_sel_es[idx_max]
        probabilidad = probs[idx_max]
        match = palabra_detectada_es == palabra_objetivo

        return jsonify({
            "match": match,
            "confidence": round(probabilidad, 4),
            "palabra_esperada": palabra_objetivo,
            "palabra_detectada": palabra_detectada_es,
            "opciones_comparadas": palabras_sel_es
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Para entorno de producción en Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

