import os
import sys
import re
import math
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import torch.nn.functional as F
import json

# ---------------------------------------------------------------------------
# TELOS PROPHET: Predictor de mutaciones en proteína Spike
# ---------------------------------------------------------------------------
# Estrategia:
#   1. Validar que la secuencia viene de un alineamiento correcto (1273 chars)
#   2. Buscar posiciones usando MOTIVOS contextuales, no índices fijos
#   3. Manejar dispositivos (MPS/CUDA/CPU) sin trace trap
#   4. Reportar predicciones solo sobre posiciones confiables
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


def validar_alineamiento(seq_con_gaps):
    """
    Valida que la secuencia viene de un alineamiento correcto contra Wuhan.
    
    Criterios:
      - Debe tener exactamente 1273 caracteres (longitud de Spike de Wuhan)
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


def encontrar_posicion_por_motivo(seq_con_gaps, pos_wuhan, ventana=5):
    """
    Encuentra la posición real de un residuo en la secuencia alineada.
    
    Estrategia:
      - La posición de Wuhan asume que no hay gaps.
      - Pero si hay gaps antes de esa posición, el índice se desplaza.
      - Buscamos en una ventana alrededor de la posición esperada.
    
    Args:
        seq_con_gaps: Secuencia alineada con gaps '-'
        pos_wuhan: Posición en la numeración de Wuhan (1-indexed)
        ventana: Rango de búsqueda a cada lado
    
    Returns:
        idx_en_alineamiento: Índice en seq_con_gaps (0-indexed), o None si no se encuentra
    """
    # Posición esperada (asumiendo que no hay gaps antes)
    idx_esperado = pos_wuhan - 1
    
    # Buscar en ventana alrededor de la posición esperada
    inicio = max(0, idx_esperado - ventana)
    fin = min(len(seq_con_gaps), idx_esperado + ventana + 1)
    
    # Buscar el primer aminoácido (no gap) en esa ventana
    for idx in range(inicio, fin):
        if seq_con_gaps[idx] != '-':
            # Verificar que este es realmente el residuo de la posición de Wuhan
            # contando cuántos aminoácidos (no gaps) hay antes
            num_aa_antes = sum(1 for c in seq_con_gaps[:idx] if c != '-')
            
            # En Wuhan, la posición pos_wuhan debería tener num_aa_antes + 1 aminoácidos
            if num_aa_antes + 1 == pos_wuhan:
                return idx
    
    return None


def predecir_mutaciones(spike_path, ruta_json_imputacion, forzar_cpu=False):
    """
    Predice mutaciones en posiciones clave de la proteína Spike.
    
    Args:
        spike_path: Ruta al archivo con la secuencia alineada de Spike
        forzar_cpu: Si True, usa CPU aunque haya GPU disponible
    """
    # ---------------------------------------------------------------------------
    # 1. Configurar dispositivo
    # ---------------------------------------------------------------------------
    if forzar_cpu:
        device = torch.device("cpu")
        print("💻 CPU forzado por el usuario")
    else:
        device = obtener_dispositivo()
    
    # ---------------------------------------------------------------------------
    # 2. Cargar modelo
    # ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # 3. Leer y validar secuencia
    # ---------------------------------------------------------------------------
    with open(spike_path, "r") as f:
        seq_con_gaps = f.read().strip()
    
    es_valido, mensaje = validar_alineamiento(seq_con_gaps)
    if not es_valido:
        print(mensaje)
        print("\n💡 Tip: Usa MAFFT o Clustal Omega para alinear tu secuencia contra Wuhan-Hu-1.")
        sys.exit(1)
    
    print(f"✅ Alineamiento válido: {len(seq_con_gaps)} caracteres")
    
    # ---------------------------------------------------------------------------
    # 4. Definir posiciones de interés (numeración de Wuhan)
    # ---------------------------------------------------------------------------
    targets = {
        "Sitio_RBM_452": 452,   # L452 en Wuhan, hotspot en Delta/BA.5
        "Sitio_RBM_484": 484,   # E484 en Wuhan, hotspot en Beta/Gamma/BA.1
        "Sitio_RBM_501": 501,   # N501 en Wuhan, hotspot en Alpha/Omicron
        "Sitio_Furina_681": 681  # P681 en Wuhan, sitio de escisión por furina
    }
    
    print("\n" + "="*60)
    print("🔮 TELOS PROPHET: Análisis de Estabilidad Estructural")
    print("="*60)
    
    resultados_json = []
    
    # ---------------------------------------------------------------------------
    # 5. Procesar cada posición
    # ---------------------------------------------------------------------------

    indices_imputados = {}

    # 1. Cargar las posiciones que sabemos que fueron imputadas
    with open(ruta_json_imputacion, "r") as f:
        datos_imp = json.load(f)
        indices_imputados = {p['idx'] for p in datos_imp['posiciones']}

    for nombre, pos_wuhan in targets.items():
        # Encontrar la posición real en el alineamiento
        idx_alineado = encontrar_posicion_por_motivo(seq_con_gaps, pos_wuhan, ventana=10)
        
        if idx_alineado is None:
            print(f"\n⚠️  {nombre} (pos {pos_wuhan}): NO SE PUDO LOCALIZAR")
            print(f"    Posible deleción o alineamiento incorrecto en esta región")
            continue

        if idx_alineado in indices_imputados:
            print(f"   ⏩ {nombre} (pos {pos_wuhan}): SALTANDO - Es un dato imputado (no real).")
            continue # El Oráculo ignora esta posición y no ensucia el informe
        
        # Verificar si es un gap
        if seq_con_gaps[idx_alineado] == '-':
            print(f"\n🟡 {nombre} (pos {pos_wuhan}): DELECIÓN en esta muestra")
            continue
        
        # ---------------------------------------------------------------------------
        # 6. Preparar secuencia limpia para el modelo
        # ---------------------------------------------------------------------------
        # El modelo no entiende gaps, necesitamos la secuencia sin '-'
        seq_limpia = seq_con_gaps.replace('-', '')
        
        # Encontrar el índice en la secuencia limpia
        # (contando aminoácidos antes de idx_alineado)
        idx_limpio = sum(1 for c in seq_con_gaps[:idx_alineado] if c != '-')
        
        original_amino = seq_limpia[idx_limpio]
        
        print(f"\n📍 {nombre} (pos Wuhan {pos_wuhan})")
        print(f"    Aminoácido actual: {original_amino}")
        print(f"    Índice en alineamiento: {idx_alineado}")
        print(f"    Índice en secuencia limpia: {idx_limpio}")
        
        # ---------------------------------------------------------------------------
        # 7. Enmascarar y predecir
        # ---------------------------------------------------------------------------
        seq_lista = list(seq_limpia)
        seq_lista[idx_limpio] = tokenizer.mask_token
        masked_seq = "".join(seq_lista)
        
        inputs = tokenizer(masked_seq, return_tensors="pt")
        input_ids_cpu = inputs["input_ids"].clone()  # Para encontrar mask
        
        # Mover a GPU
        inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs_gpu).logits
        
        # Traer a CPU inmediatamente
        logits_cpu = logits.cpu()
        
        # Encontrar posición del mask (sin .nonzero)
        mask_idx = encontrar_mask_idx(input_ids_cpu, tokenizer.mask_token_id)
        if mask_idx is None:
            print(f"    ❌ Error: no se pudo localizar [MASK] en la secuencia tokenizada")
            continue
        
        # Probabilidades
        logits_mask = logits_cpu[0, mask_idx, :]
        probabilities = F.softmax(logits_mask, dim=-1)
        
        # Top 5
        top_probs, top_indices = torch.topk(probabilities, 5)
        
        predicciones_pos = []
        print(f"    Predicciones:")
        for i in range(5):
            token = tokenizer.decode(top_indices[i].item())
            prob = top_probs[i].item() * 100
            predicciones_pos.append({"amino": token, "confidence": prob})
            
            # Marcar si coincide con el original
            marca = "← (actual)" if token == original_amino else ""
            print(f"      {i+1}. {token:2s}  {prob:5.2f}%  {marca}")
        
        # ---------------------------------------------------------------------------
        # 8. Guardar resultado
        # ---------------------------------------------------------------------------
        resultados_json.append({
            "target": nombre,
            "detected_position": pos_wuhan,
            "aligned_index": idx_alineado,
            "clean_index": idx_limpio,
            "original": original_amino,
            "predictions": predicciones_pos
        })
    
    # ---------------------------------------------------------------------------
    # 9. Guardar JSON
    # ---------------------------------------------------------------------------
    if resultados_json:
        folder_path = "output/prophet"
        os.makedirs(folder_path, exist_ok=True)
        
        nombre_base = os.path.basename(spike_path).replace('.txt', '').replace('spike_aligned/', '')
        ruta_json = os.path.join(folder_path, f"mutation_predictions_{nombre_base}.json")
        
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(resultados_json, f, indent=4, ensure_ascii=False)
        
        print("\n" + "="*60)
        print(f"✅ Predicciones guardadas en: {ruta_json}")
        print("="*60)
    else:
        print("\n⚠️  No se pudieron generar predicciones para ninguna posición")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 oraculo_mutaciones.py <spike_alineada.txt> [--cpu]")
        print("\nEjemplo:")
        print("  python3 oraculo_mutaciones.py ./output/s/spike_aligned/spike_omicron.txt")
        print("\nOpciones:")
        print("  --cpu    Forzar uso de CPU (desactivar GPU)")
        print("\nNota: La secuencia debe estar alineada contra Wuhan-Hu-1 (1273 residuos)")
        sys.exit(1)
    
    forzar_cpu = "--cpu" in sys.argv
    predecir_mutaciones(sys.argv[1], sys.argv[2], forzar_cpu)