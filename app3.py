from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import json
import random
import os

app = Flask(__name__)

# Cargar modelo CLIP
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Cargar palabras desde JSON con traducciones prehechas
with open("palabras.json", "r", encoding="utf-8") as f:
    palabras_data = json.load(f)
    palabras_es = [p["es"] for p in palabras_data]
    palabras_en = [p["en"] for p in palabras_data]

print("Calculando embeddings de texto...")
with torch.no_grad():
    text_inputs = processor(text=palabras_en, return_tensors="pt", padding=True)
    text_embeddings = model.get_text_features(**text_inputs)
    text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)

@app.route("/check", methods=["POST"])
def check():
    palabra_objetivo = request.form.get("palabra", "").strip().lower()
    imagen_file = request.files.get("foto")

    if not palabra_objetivo or not imagen_file:
        return jsonify({"error": "Faltan parámetros 'palabra' o 'foto'."}), 400

    if palabra_objetivo not in palabras_es:
        return jsonify({"error": f"La palabra '{palabra_objetivo}' no está en la lista."}), 400

    try:
        # Generar conjunto de prueba con palabra correcta incluida
        idx_correcto = palabras_es.index(palabra_objetivo)
        otros_indices = list(range(len(palabras_es)))
        otros_indices.remove(idx_correcto)
        random_indices = random.sample(otros_indices, k=9)  # 9 al azar + 1 correcta = 10
        indices = [idx_correcto] + random_indices
        random.shuffle(indices)

        palabras_es_sel = [palabras_es[i] for i in indices]
        embeddings_sel = text_embeddings[indices]

        imagen = Image.open(imagen_file).convert("RGB")
        image_inputs = processor(images=imagen, return_tensors="pt")

        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
            sims = (image_features @ embeddings_sel.T).squeeze(0)
            probs = torch.nn.functional.softmax(sims, dim=0)

        idx_max = probs.argmax().item()
        palabra_detectada_es = palabras_es_sel[idx_max]
        probabilidad = probs[idx_max].item()
        match = palabra_detectada_es == palabra_objetivo

        return jsonify({
            "match": match,
            "confidence": round(probabilidad, 4),
            "palabra_esperada": palabra_objetivo,
            "palabra_detectada": palabra_detectada_es,
            "opciones_comparadas": palabras_es_sel
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
