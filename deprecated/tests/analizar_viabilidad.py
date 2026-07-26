import torch
from transformers import EsmTokenizer, EsmForMaskedLM

def test_ia_biologica():
    print("Cargando modelo ESM-2 (Meta AI)...")
    model_name = "facebook/esm2_t6_8M_UR50D" # Version ligera para este MVP
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)

    with open("spike_wuhan.txt", "r") as f:
        sequence = f.read().strip()

    # Convertimos la proteina a tensores que la IA entiende
    inputs = tokenizer(sequence, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        # Aqui obtenemos la 'probabilidad' de la secuencia
        logits = outputs.logits
        print("Procesamiento completado.")
        print(f"Forma de los datos de salida: {logits.shape}")
        print("\nEl modelo ha 'leido' la proteina Spike con exito.")

if __name__ == "__main__":
    test_ia_biologica()