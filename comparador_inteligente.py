import os
import sys
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import torch.nn.functional as F
import csv

def analizar_contexto(posicion):
    # Logica de zonas calientes
    if 437 <= posicion <= 508:
        return "CRITICA (RBM - Contacto Directo)", 3 # Multiplicador de riesgo
    elif 319 <= posicion <= 541:
        return "ALTA (RBD - Dominio de Unión)", 2
    elif 681 <= posicion <= 685:
        return "Media (Sitio de Furina)", 1.5
    else:
        return "Normal (Cuerpo Estructural)", 1

def comparar_con_inteligencia(ref_path, var_path):
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    resultados_acumulados = []

    with open(ref_path, "r") as f:
        ref_seq = f.read().strip()
    with open(var_path, "r") as f:
        var_seq = f.read().strip()

    print("--- INFORME DE VIGILANCIA GENOMICA ---")

    for i in range(len(ref_seq)):
        if ref_seq[i] != var_seq[i]:
            pos = i + 1
            orig, mut = ref_seq[i], var_seq[i]

            # 1. Analisis de contexto
            zona, multiplicador = analizar_contexto(pos)

            # 2. Analisis de IA (LLR)
            seq_list = list(ref_seq)
            seq_list[i] = tokenizer.mask_token
            inputs = tokenizer("".join(seq_list), return_tensors="pt")

            with torch.no_grad():
                logits = model(**inputs).logits

            mask_idx = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            probs = F.softmax(logits[0, mask_idx], dim=-1)

            p_orig = probs[0, tokenizer.convert_tokens_to_ids(orig)].item()
            p_mut = probs[0, tokenizer.convert_tokens_to_ids(mut)].item()
            llr = torch.log(torch.tensor(p_mut / p_orig)).item()

            top_prob, top_idx = torch.topk(probs, 1, dim=-1)
            sugerencia_ia = tokenizer.decode(top_idx[0][0])
            prob_sugerencia = top_prob[0][0].item()

            # 3. Calculo de riesgo combinado
            # Un LLR cercano a 0 con un multiplicador alto es señal de peligro
            score_final = (1 - abs(llr)) * multiplicador
            score_riesgo = (multiplicador * 20) + (llr * 10)

            estado_txt = "🔴 AMENAZA" if (score_final > 1.5 and llr > -0.5) else "⚪ OBSERVACION"

            # 3.5 Guardar los datos en el diccionario
            resultados_acumulados.append({
                "Mutacion": f"{orig}{pos}{mut}",
                "Zona": zona,
                "LLR": round(llr, 4),
                "Estado": estado_txt,
                "Score": round(score_riesgo, 1),
                "Sugerencia_IA": f"{sugerencia_ia} ({prob_sugerencia:.4f})"
            })

            print(f"\nMutación {orig}{pos}{mut}")
            print(f"    Zona: {zona}")
            print(f"    Estabilidad (LLR): {llr:2f}")

            if score_final > 1.5 and llr > -0.5:
                print("     ESTADO: 🔴 AMENAZA POTENCIAL (Escape Inmunológico)")
            else:
                print("     ESTADO: ⚪ Variante bajo observación")

            if score_riesgo > 50:
                print(f"    Riesgo Biológico: {score_riesgo:.1f}/100 - SEGUIMIENTO URGENTE")
            elif score_riesgo > 30:
                print(f"    Riesgo Biológico: {score_riesgo:.1f}/100 - Variante de Interés")

            print(f"    Sugerencia IA para esta posición: {sugerencia_ia} ({prob_sugerencia:.4f})")

        if ref_seq[i] == "-":
            # Esto es una insercion en la variante
            print(f"Insercion Detectada en poscion aproximada {i+1}")
            print(f"Estado: 🟠 ALERTA (Nuevo material genético detectado)")
            continue # Saltamos la IA porque no hay "original" contra qué comparar LLR

        if var_seq[i] == "-":
            # Esto es una delecion en la variante
            print(f"Delecion Detectada en posicion {i+1}")
            print(f"Estado: 🟡 OBSERVACIÓN (Pérdida de aminoácido)")
            continue
        
    if resultados_acumulados:
        nombre_archivo = f"reporte_{var_path.replace('.txt', '').replace('output/s/spike_aligned/','')}.csv"
        guardar_reporte_csv(resultados_acumulados, nombre_archivo)
    else:
        print("\nNo se detectaron cambios entre las cepas.")

def guardar_reporte_csv(resultados, nombre_archivo):
    folder_path = "output/s/report"
    ruta_completa = f"{folder_path}/{nombre_archivo}"
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Folder '{folder_path}' ensured to exist.")

        keys = ["Mutacion", "Zona", "LLR", "Estado", "Score", "Sugerencia_IA"]
        with open(ruta_completa, "w", newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(resultados)
            
        print(f"\nReporte guardado con éxito en: {ruta_completa}")
            
    except OSError as e:
        # Handle other potential errors like permission issues
        print(f"Error creating folder '{folder_path}': {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python3 comparador_inteligente.py spike_wuhan_final.txt spike_variante_final.txt")
    else:
        comparar_con_inteligencia(sys.argv[1], sys.argv[2])