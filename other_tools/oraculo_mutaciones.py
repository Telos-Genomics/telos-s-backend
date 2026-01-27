import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import torch.nn.functional as F

def predecir_mutaciones(posicion_objetivo=484):
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)

    with open("spike_wuhan.txt", "r") as f:
        sequence = list(f.read().strip())

    # La posicion original en la proteina (E - Glutamato en 484)
    original_amino = sequence[posicion_objetivo - 1]
    print(f"Analizando posicion {posicion_objetivo}. Original: {original_amino}")

    # Paso clave: Enmascaramos la posicion para que la IA "adivine"
    sequence[posicion_objetivo - 1] = tokenizer.mask_token
    masked_seq = "".join(sequence)

    inputs = tokenizer(masked_seq, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # Extraemos las probabilidades para el token enmascarado
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    logits_at_mask = logits[0, mask_token_index]
    probabilities = F.softmax(logits_at_mask, dim=-1)

    # Obtenemos los aminoacidos mas probables segun la IA
    top_probs, top_indices = torch.topk(probabilities, 5, dim=-1)

    print("\nPrediciones de la IA para una variante mas estable:")
    for i in range(5):
        token = tokenizer.decode(top_indices[0][i])
        prob = top_probs[0][i].item() * 100
        print(f"Opcion {i+1}: {token} con {prob:.2f} de probabilidad")

if __name__ == "__main__":
    # Posiciones 484 es famosa por la mutacion E484K (variantes Beta y Gamma)
    predecir_mutaciones(484)