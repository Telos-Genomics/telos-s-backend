import os
import sys
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import torch.nn.functional as F
import json

def predecir_mutaciones(spike_path):
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)

    with open(spike_path, "r") as f:
        sequence = list(f.read().strip().replace('-', ''))

    # Buscamos mutaciones probables en sitios clave (ej: RBM 439-506)
    # Por ahora, analicemos posiciones que el comparador marcó como 'importantes'
    target_positions = [452, 484, 501, 681] 
    
    print(f"\n🔮 TELOS PROPHET: Predicciones de Evolución Probable")

    resultados_json = []

    for pos in target_positions:
        # CORRECCIÓN: Creamos una copia nueva para cada posición
        # sequence_copy es una lista independiente de la original
        sequence_copy = list(sequence)

        # La posicion original en la proteina (E - Glutamato en 484)
        original_amino = sequence_copy[pos - 1]
        print(f"\nAnalizando posicion {pos}. Original: {original_amino}")

        # Paso clave: Enmascaramos la posicion para que la IA "adivine"
        sequence_copy[pos - 1] = tokenizer.mask_token
        masked_seq = "".join(sequence_copy)

        inputs = tokenizer(masked_seq, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        # Extraemos las probabilidades para el token enmascarado
        mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        logits_at_mask = logits[0, mask_token_index]
        probabilities = torch.softmax(logits_at_mask, dim=-1)

        # Obtenemos los aminoácidos más probables
        top_probs, top_indices = torch.topk(probabilities, 5, dim=-1)

        print("Prediciones de estabilidad:")
        
        predicciones_pos = []
        for i in range(5):
            token = tokenizer.decode(top_indices[0][i])
            prob = top_probs[0][i].item() * 100
            predicciones_pos.append({"amino": token, "confidence": prob})
            # Marcamos si la predicción de la IA es distinta al original
            alerta = "CAMBIO" if token != original_amino else "ORIGINAL"
            print(f"  {i+1}: {token} ({prob:.2f}%) - {alerta}")

        resultados_json.append({
            "position": pos,
            "original": original_amino,
            "predictions": predicciones_pos
        })

    # Guardar JSON al final del script
    folder_path = "output/prophet"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/mutation_predictions_{spike_path.replace('.txt', '').replace('./output/s/spike/','')}.json", "w") as f:
        json.dump(resultados_json, f, indent=4)

if __name__ == "__main__":
    # Posiciones 484 es famosa por la mutacion E484K (variantes Beta y Gamma)
    if len(sys.argv) < 2:
        print("Uso: python3 oraculo_mutaciones.py spike.txt")
    else:
        predecir_mutaciones(sys.argv[1])