import os
import sys
import math
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import torch.nn.functional as F
import json
 
# ---------------------------------------------------------------------------
# TELOS PROPHET: Predictor de mutaciones en proteína Spike
# ---------------------------------------------------------------------------
# FIX v2: Reemplaza encontrar_posicion_por_motivo() con un mapa directo
# wuhan_pos → idx_alineamiento construido una sola vez.
#
# El problema anterior: la búsqueda por ventana fallaba en secuencias con
# alta densidad mutacional (como RE.2.2.3) porque la validación
# num_aa_antes + 1 == pos_wuhan no convergía dentro de la ventana pequeña
# cuando había múltiples mutaciones adyacentes a la posición objetivo.
#
# La solución: construir el mapa completo en O(n) recorriendo la secuencia
# una sola vez. Cada posición Wuhan (1-indexed) mapea exactamente a un
# índice en la secuencia alineada. Robusto independientemente de la densidad
# mutacional o de si el residuo está mutado respecto a Wuhan.
# ---------------------------------------------------------------------------
 
 
def obtener_dispositivo():
    """Detecta el mejor dispositivo disponible"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Usando CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Usando Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 Usando CPU")
    return device
 
 
def encontrar_mask_idx(input_ids_cpu, mask_token_id):
    """
    Busca el índice del token [MASK] en la secuencia.
    Evita .nonzero(as_tuple=True) que causa trace trap en MPS.
    """
    for idx in range(input_ids_cpu.shape[1]):
        if input_ids_cpu[0, idx].item() == mask_token_id:
            return idx
    return None
 
 
def construir_mapa_wuhan(seq_con_gaps):
    """
    Construye un mapa directo: wuhan_pos (1-indexed) → idx_alineamiento.
 
    Recorre la secuencia una sola vez contando aminoácidos no-gap.
    El carácter en cada posición puede ser cualquier aminoácido (incluso
    mutado respecto a Wuhan) o un gap '-'.
 
    Retorna:
        dict {wuhan_pos: idx} para todas las posiciones con aminoácido real.
        Las posiciones con gap en la variante no tienen entrada (deleción).
    """
    mapa = {}
    contador_wuhan = 0  # posición en numeración Wuhan (1-indexed)
 
    for idx, char in enumerate(seq_con_gaps):
        if char == '-':
            # Gap en variante = deleción en esa posición Wuhan
            # Incrementamos el contador Wuhan igualmente porque la referencia
            # sí tiene un residuo aquí
            contador_wuhan += 1
            # No añadimos al mapa: deleción confirmada
        else:
            contador_wuhan += 1
            mapa[contador_wuhan] = idx
 
    return mapa
 
 
def validar_alineamiento(seq_con_gaps):
    """
    Valida que la secuencia viene de un alineamiento correcto contra Wuhan.
    Criterios:
      - Debe tener exactamente 1273 caracteres
      - Los caracteres deben ser aminoácidos válidos o gaps '-'
    Retorna: (es_valido, mensaje_error)
    """
    longitud = len(seq_con_gaps)
    if longitud != 1273:
        return False, (
            f"❌ Longitud incorrecta: {longitud} caracteres. "
            f"Spike de Wuhan tiene 1273 residuos. "
            f"Asegúrate de que la secuencia está alineada contra la referencia de Wuhan."
        )
    return True, ""
 
 
def predecir_mutaciones(spike_path, ruta_json_imputacion, forzar_cpu=False):
    """
    Predice mutaciones en posiciones clave de la proteína Spike.
 
    Args:
        spike_path: Ruta al archivo con la secuencia alineada de Spike
        ruta_json_imputacion: Ruta al JSON de posiciones imputadas
        forzar_cpu: Si True, usa CPU aunque haya GPU disponible
    """
    # ------------------------------------------------------------------
    # 1. Dispositivo
    # ------------------------------------------------------------------
    if forzar_cpu:
        device = torch.device("cpu")
        print("💻 CPU forzado por el usuario")
    else:
        device = obtener_dispositivo()
 
    # ------------------------------------------------------------------
    # 2. Cargar modelo
    # ------------------------------------------------------------------
    model_name = "facebook/esm2_t33_650M_UR50D"
    print(f"\n📥 Cargando modelo {model_name}...")
 
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name, torch_dtype=torch.float32)
 
    try:
        model = model.to(device)
        print(f"✅ Modelo cargado en {device}")
    except Exception as e:
        print(f"⚠️  Error al mover modelo a {device}: {e}")
        device = torch.device("cpu")
        model = model.to(device)
        print("🔄 Cayó a CPU")
 
    model.eval()
 
    # ------------------------------------------------------------------
    # 3. Leer y validar secuencia
    # ------------------------------------------------------------------
    with open(spike_path, "r") as f:
        seq_con_gaps = f.read().strip()
 
    es_valido, mensaje = validar_alineamiento(seq_con_gaps)
    if not es_valido:
        print(mensaje)
        print("\n💡 Tip: Usa MAFFT o Clustal Omega para alinear tu secuencia contra Wuhan.")
        sys.exit(1)
 
    print(f"✅ Alineamiento válido: {len(seq_con_gaps)} caracteres")
 
    # ------------------------------------------------------------------
    # 4. Construir mapa Wuhan → índice (FIX PRINCIPAL)
    # ------------------------------------------------------------------
    mapa_wuhan = construir_mapa_wuhan(seq_con_gaps)
 
    aa_no_gap = len(mapa_wuhan)
    gaps = seq_con_gaps.count('-')
    print(f"📊 Mapa construido: {aa_no_gap} aminoácidos, {gaps} deleciones")
 
    # ------------------------------------------------------------------
    # 5. Posiciones de interés
    # ------------------------------------------------------------------
    targets = {
        "Sitio_RBM_452": 452,
        "Sitio_RBM_484": 484,
        "Sitio_RBM_501": 501,
        "Sitio_Furina_681": 681,
    }
 
    # ------------------------------------------------------------------
    # 6. Cargar posiciones imputadas
    # ------------------------------------------------------------------
    indices_imputados = set()
    try:
        with open(ruta_json_imputacion, "r") as f:
            datos_imp = json.load(f)
            indices_imputados = {p['idx'] for p in datos_imp['posiciones']}
        print(f"📋 Posiciones imputadas cargadas: {len(indices_imputados)}")
    except Exception as e:
        print(f"⚠️  No se pudo cargar JSON de imputación: {e}")
        print("   Continuando sin filtro de imputación...")
 
    print("\n" + "=" * 60)
    print("🔮 TELOS PROPHET: Análisis de Estabilidad Estructural")
    print("=" * 60)
 
    resultados_json = []
 
    # ------------------------------------------------------------------
    # 7. Procesar cada posición objetivo
    # ------------------------------------------------------------------
    for nombre, pos_wuhan in targets.items():
 
        # Lookup directo — O(1), sin ventana, sin ambigüedad
        idx_alineado = mapa_wuhan.get(pos_wuhan)
 
        if idx_alineado is None:
            # La posición Wuhan no tiene aminoácido en la variante → deleción
            print(f"\n🟡 {nombre} (pos {pos_wuhan}): DELECIÓN confirmada en esta muestra")
            continue
 
        # Verificar si fue imputado
        if idx_alineado in indices_imputados:
            print(f"\n⏩ {nombre} (pos {pos_wuhan}): SALTANDO — dato imputado (no real)")
            continue
 
        # Verificar que no sea gap (doble check, no debería ocurrir dado el mapa)
        if seq_con_gaps[idx_alineado] == '-':
            print(f"\n🟡 {nombre} (pos {pos_wuhan}): GAP — deleción")
            continue
 
        # Aminoácido actual en la variante (puede estar mutado respecto a Wuhan)
        aminoacido_actual = seq_con_gaps[idx_alineado]
 
        print(f"\n📍 {nombre} (pos Wuhan {pos_wuhan})")
        print(f"    Aminoácido actual en variante: {aminoacido_actual}")
        print(f"    Índice en alineamiento: {idx_alineado}")
 
        # ------------------------------------------------------------------
        # 8. Preparar secuencia limpia para el modelo
        # ------------------------------------------------------------------
        seq_limpia = seq_con_gaps.replace('-', '')
 
        # Índice en secuencia limpia = número de no-gaps antes de idx_alineado
        idx_limpio = sum(1 for c in seq_con_gaps[:idx_alineado] if c != '-')
 
        print(f"    Índice en secuencia limpia: {idx_limpio}")
 
        # Enmascarar
        seq_lista = list(seq_limpia)
        seq_lista[idx_limpio] = tokenizer.mask_token
        masked_seq = "".join(seq_lista)
 
        inputs = tokenizer(masked_seq, return_tensors="pt")
        input_ids_cpu = inputs["input_ids"].clone()
 
        # Inferencia en GPU
        inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs_gpu).logits
 
        # Traer a CPU
        logits_cpu = logits.cpu()
 
        # Localizar MASK
        mask_idx = encontrar_mask_idx(input_ids_cpu, tokenizer.mask_token_id)
        if mask_idx is None:
            print(f"    ❌ No se encontró [MASK] en la secuencia tokenizada")
            continue
 
        # ------------------------------------------------------------------
        # 9. Calcular probabilidades
        # ------------------------------------------------------------------
        logits_mask = logits_cpu[0, mask_idx, :]
        probabilities = F.softmax(logits_mask, dim=-1)
 
        top_probs, top_indices = torch.topk(probabilities, 5)
 
        predicciones_pos = []
        print(f"    Predicciones del modelo (top 5):")
        for i in range(5):
            token = tokenizer.decode(top_indices[i].item())
            prob = top_probs[i].item() * 100
            predicciones_pos.append({"amino": token, "confidence": prob})
            marca = "← (actual en variante)" if token == aminoacido_actual else ""
            print(f"      {i+1}. {token:2s}  {prob:5.2f}%  {marca}")
 
        resultados_json.append({
            "target": nombre,
            "detected_position": pos_wuhan,
            "aligned_index": idx_alineado,
            "clean_index": idx_limpio,
            "original": aminoacido_actual,
            "predictions": predicciones_pos
        })
 
    # ------------------------------------------------------------------
    # 10. Guardar JSON
    # ------------------------------------------------------------------
    if resultados_json:
        nombre_base = os.path.basename(spike_path).replace('.txt', '').replace('spike_aligned/', '')
        ruta_json = f"output/prophet/mutation_predictions_{nombre_base}.json"
        os.makedirs("output/prophet", exist_ok=True)
 
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(resultados_json, f, indent=4, ensure_ascii=False)
 
        print("\n" + "=" * 60)
        print(f"✅ Predicciones guardadas en: {ruta_json}")
        print("=" * 60)
    else:
        print("\n⚠️  No se pudieron generar predicciones para ninguna posición")
 
 
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python3 oraculo_mutaciones.py <spike_alineada.txt> <imputacion.json> [--cpu]")
        print("\nEjemplo:")
        print("  python3 oraculo_mutaciones.py output/s/spike_aligned/spike_omicron.txt output/prophet/imputacion_spike_omicron.json")
        print("\nOpciones:")
        print("  --cpu    Forzar uso de CPU (desactivar GPU)")
        sys.exit(1)
 
    forzar_cpu = "--cpu" in sys.argv
    predecir_mutaciones(sys.argv[1], sys.argv[2], forzar_cpu)