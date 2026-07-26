import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import torch.nn.functional as F

def calcular_puntuacion_riesgo(ref_path, var_path):
    # 1. Cargar el "Cerebro"
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)

    #2. Leer secuencias
    with open(ref_path, "r") as f:
        ref_seq = f.read().strip()
    with open(var_path, "r") as f:
        var_seq = f.read().strip()

    # 3. Encontrar diferencias (Mutaciones)
    mutaciones = []
    for i in range(len(ref_seq)):
        if ref_seq[i] != var_seq[i]:
            mutaciones.append((i + 1, ref_seq[i], var_seq[i]))

    print(f"Detectadas {len(mutaciones)} mutaciones entre las cepas.\n")

    # 4. Analisis de IA para cada mutacion
    for pos, original, mutado in mutaciones:
        # Enmascaramos la posicion para evaluar
        seq_list = list(ref_seq)
        seq_list[pos-1] = tokenizer.mask_token
        masked_seq = "".join(seq_list)

        inputs = tokenizer(masked_seq, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        # Calcular Log-Likelihood
        mask_idx = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        probs = F.softmax(logits[0, mask_idx], dim=-1)

        # Obtener ids de los aminoacidos
        id_original = tokenizer.convert_tokens_to_ids(original)
        id_mutado = tokenizer.convert_tokens_to_ids(mutado)

        prob_orig = probs[0, id_original].item()
        prob_mut = probs[0, id_mutado].item()

        # Log-Likehood Ratio (simplificado)
        llr = torch.log(torch.tensor(prob_mut / prob_orig)).item()

        estado = "🔴 ALTO RIESGO" if llr > 0 else "🟢 NEUTRAL/BAJO"
        print(f"Mutacion {original}{pos}{mutado}:")
        print(f"    - Confianza IA en Original: {prob_orig:.4f}")
        print(f"    - Confianza IA en Mutante: {prob_mut:.4f}")
        print(f"    - Score LLR: {llr:.2f} -> {estado}\n")

if __name__ == "__main__":
    calcular_puntuacion_riesgo("spike_wuhan.txt", "spike_latam_simulada.txt")