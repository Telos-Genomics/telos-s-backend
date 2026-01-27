import os
import sys
from Bio import SeqIO
from Bio.Seq import Seq

def procesar_referencia(archivo_fasta):
    # Cargamos el genoma completo
    try:
        record = SeqIO.read(archivo_fasta, "fasta")
    except Exception as e:
        print(f"❌ Error al leer el archivo: {e}")
        return

    secuencia_completa = record.seq

    print(f"--- Análisis de {record.id} ---")
    print(f"Longitud total del genoma: {len(secuencia_completa)} nucleótidos")

    # Sondas de busqueda de la proteina Spike en la referencia de Whan (NC_045512.2)
    # Inicio de la Spike (nucleotidos)
    sonda_inicio = "ATGTTTGTTTTT"
    # Fin de la Spike (nucleotidos - STOP codon)

    # Buscamos el inicio de la porteina Spike
    # Nota: En un entorno real, buscariamos la secuencia que codifica MFVL...
    # Para ser mas robustos, buscaremos el marco de lectura abierto (ORF)

    start_pos = secuencia_completa.find("ATGTTTGTTTTT") # MFVF...

    if start_pos == -1:
        print("No se encontró el inicio de la Spike por secuencia exacta. Intentando modo busqueda de ORF...")
        # Aqui podrias implementar una busqueda mas compleja si falla
        return None

    # 2. Buscamos el final de forma más robusta
    # Definimos un rango amplio porque la Spike puede variar de tamaño
    # Buscamos desde 3700 nucleótidos después del inicio hasta 4000
    sub_seq = secuencia_completa[start_pos:]

    # Buscamos el final (Stop codon TAA, TAG, o TGA) en el marco de lectura correcto
    end_pos_relative = -1
    # Recorremos la secuencia de 3 en 3 DESDE EL INICIO (0, 3, 6...)
    # para asegurar que mantenemos el marco de lectura (frame)
    for i in range(0, len(sub_seq), 3):
        # Un gen normal no debería ser más corto de 3700 ni más largo de 3900 nts
        triplete = sub_seq[i:i+3]
        if triplete in ["TAA", "TAG", "TGA"]:
            if 3700 <= i <= 3900: # Filtro de seguridad por tamaño esperado
                end_pos_relative = i + 3
                break
    
    if end_pos_relative == -1:
        # Si no lo encontró en el rango esperado, buscamos el primer STOP que aparezca
        # después de 3700 nts sin límite superior por si hubo una inserción gigante
        for i in range(3700, len(sub_seq), 3):
            if sub_seq[i:i+3] in ["TAA", "TAG", "TGA"]:
                end_pos_relative = i + 3
                break
    
    if end_pos_relative == -1:
        print("❌ No se encontró el codón de parada en el marco de lectura.")
        return None

    # Extraemos el gen S (ADN/ARN)
    gen_s = sub_seq[:end_pos_relative]

    # Traducimos a Proteina (Aminoacidos)
    # El parametro to_stop=True corta la traduccion en el primer codón de parada
    proteina_s = gen_s.translate(to_stop=True)

    print(f"\nSpike extraida dinamicamente.")
    print(f"    Posicion encontrada: {start_pos} a {start_pos + end_pos_relative}")
    print(f"    Longitud proteina: {len(proteina_s)} aa")

    folder_path = "output/spike"
    nombre_archivo = f"{folder_path}/spike_{archivo_fasta.replace('.fasta', '').replace('./','')}.txt"

    # Guardamos la proteina para futuros analisis de IA (ESM-2)
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Folder '{folder_path}' ensured to exist.")

        with open(nombre_archivo, "w") as f:
            f.write(str(proteina_s))
            print(f"\nProteina guardada en '{nombre_archivo}'")
    except OSError as e:
        # Handle other potential errors like permission issues
        print(f"Error creating folder '{folder_path}': {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 extraer_spike.py mi_genoma.fasta")
    else:
        procesar_referencia(sys.argv[1])