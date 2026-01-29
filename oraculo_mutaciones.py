import os
import sys
import re
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import torch.nn.functional as F
import json

def encontrar_posicion_dinamica(secuencia, motivo_regex, offset_en_motivo):
    """
    Busca un motivo (ej: 'GFNCY') en la secuencia y devuelve el índice 
    ajustado según el offset.
    """
    secuencia_str = "".join(secuencia)
    # re.search permite usar patrones como 'G.NCY' donde el punto es cualquier AA
    match = re.search(motivo_regex, secuencia_str)
    if not match:
        return None
    # match.start() nos da el inicio de la cadena encontrada
    return match.start() + offset_en_motivo + 1

def predecir_mutaciones(spike_path):
    # Cargamos modelo y tokenizer (igual que antes)
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)

    # 1. LEER LA SECUENCIA MANTENIENDO LOS GAPS (Alineada)
    with open(spike_path, "r") as f:
        # Importante: No quitamos los '-' todavía
        full_seq_with_gaps = f.read().strip()

    # 2. POSICIONES ABSOLUTAS DE REFERENCIA (Wuhan)
    # Como el archivo está alineado, la posición 501 SIEMPRE es la 501
    targets = {
        "Sitio_RBM_452": 452,
        "Sitio_RBM_484": 484,
        "Sitio_RBM_501": 501,
        "Sitio_Furina_681": 681
    }
    
    print(f"\n🔮 TELOS PROPHET: Análisis por Coordenadas Sincronizadas")
    resultados_json = []

    for nombre, pos_wuhan in targets.items():
        idx = pos_wuhan - 1 # Índice 0 de Python
        
        # Verificamos si la muestra tiene un aminoácido o un gap en esa posición
        if full_seq_with_gaps[idx] == "-":
            print(f"⚠️ {nombre} es una deleción en esta muestra. Saltando...")
            continue

        # 3. TRADUCCIÓN DE COORDENADAS PARA LA IA
        # La IA no entiende guiones. Necesitamos saber qué índice tiene 
        # nuestro aminoácido en la versión "limpia" de la proteína.
        
        # El índice en la secuencia limpia es igual al número de 
        # aminoácidos (no gaps) que hay antes de la posición actual.
        seq_antes = full_seq_with_gaps[:idx]
        idx_limpio = len(seq_antes.replace("-", ""))
        
        # Creamos la secuencia limpia para la IA
        seq_limpia_lista = list(full_seq_with_gaps.replace("-", ""))
        original_amino = seq_limpia_lista[idx_limpio]

        print(f"\nAnalizando {nombre} (Aminoacido Actual: {original_amino})")

        # 4. ENMASCARAMIENTO Y PREDICCIÓN
        seq_limpia_lista[idx_limpio] = tokenizer.mask_token
        masked_seq = "".join(seq_limpia_lista)
        
        inputs = tokenizer(masked_seq, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        # Extraer probabilidades
        mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        logits_at_mask = logits[0, mask_token_index]
        probabilities = torch.softmax(logits_at_mask, dim=-1)
        
        top_probs, top_indices = torch.topk(probabilities, 5, dim=-1)

        predicciones_pos = []
        for i in range(5):
            token = tokenizer.decode(top_indices[0][i])
            prob = top_probs[0][i].item() * 100
            predicciones_pos.append({"amino": token, "confidence": prob})
            print(f"  {i+1}: {token} ({prob:.2f}%)")

        resultados_json.append({
            "target": nombre,
            "detected_position": pos_wuhan, # Ahora reportamos la posición real de Wuhan
            "original": original_amino,
            "predictions": predicciones_pos
        })
        

    # Guardar JSON al final del script
    folder_path = "output/prophet"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/mutation_predictions_{spike_path.replace('.txt', '').replace('./output/s/spike_aligned/','')}.json", "w") as f:
        json.dump(resultados_json, f, indent=4)

if __name__ == "__main__":
    # Posiciones 484 es famosa por la mutacion E484K (variantes Beta y Gamma)
    if len(sys.argv) < 2:
        print("Uso: python3 oraculo_mutaciones.py spike.txt")
    else:
        predecir_mutaciones(sys.argv[1])