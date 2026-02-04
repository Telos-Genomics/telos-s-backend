import os
import sys
import math
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import torch.nn.functional as F
import csv
import time

# ---------------------------------------------------------------------------
# Estrategia de dispositivos:
#   - El modelo y los inputs viven en GPU (MPS o CUDA) durante la inferencia.
#   - En el momento en que los logits salen del modelo, los traemos a CPU con
#     .cpu(). Todas las operaciones de post-procesamiento (nonzero, softmax,
#     topk, log) se ejecutan en CPU, donde están 100 % soportadas.
#   - Esto evita el trace trap que MPS lanza con nonzero(as_tuple=True),
#     indexación avanzada y otras ops no implementadas en Metal.
# ---------------------------------------------------------------------------

def obtener_dispositivo():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 CUDA detectado: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("🍎 MPS detectado (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("💻 Usando CPU")
    return device


def analizar_contexto(posicion):
    if 437 <= posicion <= 508:
        return "CRITICA (RBM - Contacto Directo)", 3
    elif 319 <= posicion <= 541:
        return "ALTA (RBD - Dominio de Unión)", 2
    elif 681 <= posicion <= 685:
        return "Media (Sitio de Furina)", 1.5
    else:
        return "Normal (Cuerpo Estructural)", 1


def comparar_con_inteligencia(ref_path, var_path, forzar_cpu=False):
    # ------------------------------------------------------------------
    # 1. Dispositivo
    # ------------------------------------------------------------------
    if forzar_cpu:
        device = torch.device("cpu")
        print("💻 CPU forzado por el usuario")
    else:
        device = obtener_dispositivo()

    # ------------------------------------------------------------------
    # 2. Cargar modelo y tokenizer
    # ------------------------------------------------------------------
    model_name = "facebook/esm2_t33_650M_UR50D"
    print(f"\n📥 Cargando modelo {model_name}...")

    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name, torch_dtype=torch.float32)

    try:
        model = model.to(device)
        print(f"✅ Modelo en {device}")
    except Exception as e:
        print(f"⚠️  No se pudo cargar en {device}: {e}")
        device = torch.device("cpu")
        model = model.to(device)
        print("🔄 Cayó a CPU")

    model.eval()

    # ------------------------------------------------------------------
    # 3. Leer secuencias
    # ------------------------------------------------------------------
    with open(ref_path, "r") as f:
        ref_seq = f.read().strip()
    with open(var_path, "r") as f:
        var_seq = f.read().strip()

    if len(ref_seq) != len(var_seq):
        print("❌ Error: las secuencias tienen diferente longitud. "
              "Deben estar alineadas previamente.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Encontrar la posición del [MASK] de una sola forma confiable
    #    Usamos el tokenizer para saber cuál es el token_id del mask,
    #    y luego buscamos su posición con un bucle simple en CPU.
    #    Esto reemplaza .nonzero(as_tuple=True) que crashea en MPS.
    # ------------------------------------------------------------------
    def encontrar_mask_idx(input_ids_cpu, mask_token_id):
        """Retorna el índice (int) del token [MASK] en la secuencia."""
        for idx in range(input_ids_cpu.shape[1]):
            if input_ids_cpu[0, idx].item() == mask_token_id:
                return idx
        return None  # No debería pasar si el mask está en la secuencia

    # ------------------------------------------------------------------
    # 5. Loop principal
    # ------------------------------------------------------------------
    resultados_acumulados = []
    tiempo_inicio = time.time()

    print("\n" + "=" * 60)
    print("  INFORME DE VIGILANCIA GENÓMICA")
    print("=" * 60)

    for i in range(len(ref_seq)):
        # --- Inserciones y deleciones primero (no necesitan el modelo) ---
        if ref_seq[i] == "-":
            print(f"\n🟠 Inserción en posición ~{i + 1}")
            continue

        if var_seq[i] == "-":
            print(f"\n🟡 Deleción en posición {i + 1}")
            continue

        # --- Solo procesar si hay mutación ---
        if ref_seq[i] == var_seq[i]:
            continue

        pos = i + 1
        orig, mut = ref_seq[i], var_seq[i]
        zona, multiplicador = analizar_contexto(pos)

        # --- Preparar entrada con [MASK] ---
        seq_list = list(ref_seq)
        seq_list[i] = tokenizer.mask_token
        inputs = tokenizer("".join(seq_list), return_tensors="pt")

        # Guardar input_ids en CPU para encontrar el mask index
        input_ids_cpu = inputs["input_ids"].clone()  # siempre CPU aquí

        # Mover inputs a la GPU para la inferencia
        inputs_gpu = {k: v.to(device) for k, v in inputs.items()}

        # --- Inferencia en GPU ---
        with torch.no_grad():
            logits = model(**inputs_gpu).logits

        # --- Traer logits a CPU inmediatamente ---
        logits_cpu = logits.cpu()

        # --- Todo lo de aquí hacia abajo es CPU, sin trace trap posible ---
        mask_idx = encontrar_mask_idx(input_ids_cpu, tokenizer.mask_token_id)
        if mask_idx is None:
            print(f"⚠️  No se encontró [MASK] en posición {pos}, saltando")
            continue

        # Extraer los logits solo de la posición del mask
        logits_mask = logits_cpu[0, mask_idx, :]          # shape: [vocab_size]
        probs = F.softmax(logits_mask, dim=-1)            # shape: [vocab_size]

        # Probabilidades para original y mutante
        orig_id = tokenizer.convert_tokens_to_ids(orig)
        mut_id = tokenizer.convert_tokens_to_ids(mut)

        p_orig = probs[orig_id].item()
        p_mut  = probs[mut_id].item()

        # LLR con protección contra log(0)
        if p_orig > 0 and p_mut > 0:
            llr = math.log(p_mut / p_orig)
        else:
            llr = -10.0  # mutación prácticamente imposible según el modelo

        # Sugerencia del modelo (aminoácido más probable)
        top_prob, top_idx = torch.topk(probs, 1)
        sugerencia_ia   = tokenizer.decode(top_idx[0].item())
        prob_sugerencia = top_prob[0].item()

        # --- Scoring ---
        score_final  = (1 - abs(llr)) * multiplicador
        score_riesgo = (multiplicador * 20) + (llr * 10)

        es_amenaza = score_final > 1.5 and llr > -0.5
        estado_txt = "🔴 AMENAZA" if es_amenaza else "⚪ OBSERVACION"

        # --- Acumular resultado ---
        resultados_acumulados.append({
            "Mutacion":       f"{orig}{pos}{mut}",
            "Zona":           zona,
            "LLR":            round(llr, 4),
            "Estado":         estado_txt,
            "Score":          round(score_riesgo, 1),
            "Sugerencia_IA":  f"{sugerencia_ia} ({prob_sugerencia:.4f})",
            "P_Original":     round(p_orig, 6),
            "P_Mutante":      round(p_mut, 6),
        })

        # --- Imprimir ---
        print(f"\nMutación {orig}{pos}{mut}")
        print(f"    Zona:        {zona}")
        print(f"    LLR:         {llr:.4f}")
        print(f"    P(orig):     {p_orig:.6f}  |  P(mut): {p_mut:.6f}")
        print(f"    Estado:      {estado_txt}")

        if score_riesgo > 50:
            print(f"    Riesgo:      {score_riesgo:.1f}/100 — SEGUIMIENTO URGENTE")
        elif score_riesgo > 30:
            print(f"    Riesgo:      {score_riesgo:.1f}/100 — Variante de Interés")

        print(f"    Sugerencia:  {sugerencia_ia} ({prob_sugerencia:.4f})")

    # ------------------------------------------------------------------
    # 6. Resumen y reporte
    # ------------------------------------------------------------------
    tiempo_total = time.time() - tiempo_inicio
    n = len(resultados_acumulados)

    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("=" * 60)
    print(f"  Dispositivo:      {device}")
    print(f"  Mutaciones:       {n}")
    print(f"  Tiempo total:     {tiempo_total:.2f}s")
    if n > 0:
        print(f"  Tiempo/mutación:  {tiempo_total / n:.2f}s")

    if resultados_acumulados:
        nombre_base   = os.path.basename(var_path).replace(".txt", "")
        nombre_archivo = f"reporte_{nombre_base}.csv"
        guardar_reporte_csv(resultados_acumulados, nombre_archivo)
    else:
        print("\n✅ No se detectaron mutaciones entre las cepas.")


def guardar_reporte_csv(resultados, nombre_archivo):
    folder_path   = "output/s/report"
    ruta_completa = os.path.join(folder_path, nombre_archivo)

    try:
        os.makedirs(folder_path, exist_ok=True)

        keys = ["Mutacion", "Zona", "LLR", "Estado", "Score",
                "Sugerencia_IA", "P_Original", "P_Mutante"]

        with open(ruta_completa, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(resultados)

        print(f"\n✅ Reporte guardado en: {ruta_completa}")

    except OSError as e:
        print(f"❌ Error al guardar reporte: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso:")
        print("  python3 comparador_inteligente.py <referencia.txt> <variante.txt> [--cpu]")
        print("\nOpciones:")
        print("  --cpu   Forzar CPU (útil para debug)")
        sys.exit(1)

    forzar_cpu = "--cpu" in sys.argv
    comparar_con_inteligencia(sys.argv[1], sys.argv[2], forzar_cpu)