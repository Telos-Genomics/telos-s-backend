import os
import sys
from Bio import Align

def alinear_sincronizar(ref_path, var_path):
    with open(ref_path, "r") as f:
        ref_seq = f.read().strip()
    with open(var_path, "r") as f:
        var_seq = f.read().strip()

    print(f"Alineando secuencias: {len(ref_seq)} vs {len(var_seq)} aa")

    # Usamos un alineador global
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    # Penalizamos los huecos (gaps) para que no se invente alineamientos raros
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(ref_seq, var_seq)
    mejor_alineamiento = alignments[0]

    # Esto nos da las secuencias con guiones si es necesario
    ref_aligned = mejor_alineamiento[0]
    var_aligned = mejor_alineamiento[1]

    folder_path = "output/s/spike_aligned"
    nombre_ref_final = f"{folder_path}/{ref_path.replace('.txt', '').replace('output/s/spike/','')}_final.txt"
    nombre_var_final = f"{folder_path}/{var_path.replace('.txt', '').replace('output/s/spike/','')}_final.txt"

    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Folder '{folder_path}' ensured to exist.")

        with open(nombre_ref_final, "w") as f:
            f.write(ref_aligned)
        with open(nombre_var_final, "w") as f:
            f.write(var_aligned)
            
        print(f"\nProteinas guardadas en '{folder_path}'")
    except OSError as e:
        # Handle other potential errors like permission issues
        print(f"Error creating folder '{folder_path}': {e}")

    print("Secuencias sincronizadas y guardadas")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python3 alinear_secuencias.py spike_wuhan.txt spike_variante.txt")
    else:
        alinear_sincronizar(sys.argv[1], sys.argv[2])