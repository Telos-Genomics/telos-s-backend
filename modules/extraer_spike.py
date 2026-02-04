import os
import sys
from Bio import SeqIO
from Bio.Seq import Seq

# Constantes de validación
LONGITUD_ESPERADA_AA = 1273  # Spike de Wuhan
TOLERANCIA_AA = 10           # ±10 aa es aceptable (pequeñas indels)
LONGITUD_MIN_GENOMA = 25000  # Genoma SARS-CoV-2 completo es ~30kb
RANGO_SPIKE_WUHAN = (21563, 25384)  # Posiciones en NC_045512.2

def procesar_referencia(archivo_fasta):
    """
    Extrae la proteína Spike de un genoma SARS-CoV-2.
    
    Soporta:
      - Genoma completo
      - Solo región Spike
      - Variantes con pequeñas mutaciones en el inicio
    """
    # ---------------------------------------------------------------------------
    # 1. Cargar secuencia
    # ---------------------------------------------------------------------------
    record = None

    # Intentar primero con formato FASTA estándar
    try:
        record = SeqIO.read(archivo_fasta, "fasta")
    except ValueError as e:
        # Múltiples secuencias en el FASTA
        error_msg = str(e)
        
        if "multiple records" in error_msg.lower():
            print(f"⚠️  El archivo contiene múltiples secuencias.")
            print(f"   Procesando solo la primera secuencia...")
            
            # Intentar leer todas y tomar la primera
            try:
                records = list(SeqIO.parse(archivo_fasta, "fasta"))
            except Exception:
                # Si falla con 'fasta', intentar con 'fasta-pearson' (soporta comentarios)
                print(f"⚠️  Formato FASTA estándar falló, intentando con formato Pearson...")
                try:
                    records = list(SeqIO.parse(archivo_fasta, "fasta-pearson"))
                except Exception as e2:
                    print(f"❌ Error al leer archivo: {e2}")
                    return None
            
            if not records:
                print(f"❌ No se pudo leer ninguna secuencia del archivo")
                print(f"   El archivo puede estar vacío o tener formato incorrecto")
                return None
            
            record = records[0]
            print(f"   ✓ Procesando: {record.id}")
            
            if len(records) > 1:
                print(f"   ℹ️  Otras {len(records)-1} secuencias ignoradas:")
                for r in records[1:6]:  # Mostrar hasta 5
                    print(f"      - {r.id}")
                if len(records) > 6:
                    print(f"      ... y {len(records)-6} más")
        else:
            # Otro tipo de ValueError
            print(f"❌ Error al leer el archivo: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Error inesperado al leer el archivo: {e}")
        return None
    
    if record is None:
        print(f"❌ No se pudo cargar ninguna secuencia")
        return None

    secuencia_completa = record.seq
    longitud_input = len(secuencia_completa)

    print(f"--- Análisis de {record.id} ---")
    print(f"Longitud total: {longitud_input:,} nucleótidos")

    # ---------------------------------------------------------------------------
    # 2. Detectar tipo de input
    # ---------------------------------------------------------------------------
    es_genoma_completo = longitud_input >= LONGITUD_MIN_GENOMA

    if longitud_input < 3000:
        print(f"\n{'='*60}")
        print(f"❌ ADVERTENCIA: Secuencia muy corta")
        print(f"{'='*60}")
        print(f"  Longitud: {longitud_input} nt")
        print(f"  Mínimo para Spike completa: ~3,800 nt")
        print(f"\n  Esta secuencia es demasiado corta para contener")
        print(f"  la proteína Spike completa de SARS-CoV-2.")
        print(f"\n  Posibles causas:")
        print(f"    - Secuencia fragmentada")
        print(f"    - Solo una región parcial del gen S")
        print(f"    - Archivo incorrecto")
        print(f"{'='*60}\n")
        
        respuesta = input("¿Continuar de todas formas? (s/N): ").strip().lower()
        if respuesta not in ['s', 'si', 'sí', 'y', 'yes']:
            print("Operación cancelada")
            return None
    
    if es_genoma_completo:
        print(f"✓ Detectado: Genoma completo")
    else:
        print(f"✓ Detectado: Secuencia parcial (probablemente solo región Spike)")

    # ---------------------------------------------------------------------------
    # 3. Buscar inicio de Spike
    # ---------------------------------------------------------------------------
    # Estrategia multinivel:
    #   1. Buscar secuencia exacta "ATGTTTGTTTTT" (codifica MFVF...)
    #   2. Si falla y es genoma completo, usar posición conocida de Wuhan
    #   3. Si es parcial, asumir que empieza en posición 0
    
    secuencia_str = str(secuencia_completa).upper()  # Convertir a string para búsqueda
    start_pos = secuencia_str.find("ATGTTTGTTTTT")
    metodo_busqueda = "Búsqueda exacta"
    
    if start_pos == -1:
        if es_genoma_completo:
            # Fallback: usar posición conocida de Wuhan
            start_pos = RANGO_SPIKE_WUHAN[0]
            metodo_busqueda = "Posición de referencia (Wuhan)"
            print(f"⚠️  No se encontró secuencia de inicio exacta")
            print(f"   Usando posición de Wuhan: {start_pos}")
        else:
            # Si es secuencia parcial, probablemente ya es solo Spike
            start_pos = 0
            metodo_busqueda = "Inicio de secuencia parcial"
            print(f"⚠️  Asumiendo que la secuencia completa es la región Spike")
    
    # ---------------------------------------------------------------------------
    # 4. Buscar STOP codon
    # ---------------------------------------------------------------------------
    sub_seq = secuencia_completa[start_pos:]
    sub_seq_str = str(sub_seq).upper()  # String para búsqueda
    end_pos_relative = -1
    
    # Buscar STOP en el marco de lectura correcto
    for i in range(0, len(sub_seq_str), 3):
        triplete = sub_seq_str[i:i+3]
        
        if triplete in ["TAA", "TAG", "TGA"]:
            # Validar que está en el rango esperado para Spike
            if 3700 <= i <= 3900:
                end_pos_relative = i + 3  # Incluir el STOP
                break
    
    # Si no encontramos STOP en rango esperado, buscar el primero que aparezca
    if end_pos_relative == -1:
        print(f"⚠️  STOP codon no encontrado en rango esperado (3700-3900 nt)")
        print(f"   Buscando primer STOP disponible...")
        
        for i in range(3700, len(sub_seq_str), 3):
            if sub_seq_str[i:i+3] in ["TAA", "TAG", "TGA"]:
                end_pos_relative = i + 3
                print(f"   ✓ STOP encontrado en posición {i} (inusual)")
                break
    
    if end_pos_relative == -1:
        print(f"❌ No se encontró codón de parada en el marco de lectura")
        print(f"   Posible frameshift o secuencia incompleta")
        return None
    
    # ---------------------------------------------------------------------------
    # 5. Extraer y traducir
    # ---------------------------------------------------------------------------
    gen_s = sub_seq[:end_pos_relative]
    
    # Validar que la longitud del gen es múltiplo de 3
    if len(gen_s) % 3 != 0:
        print(f"⚠️  Advertencia: longitud del gen ({len(gen_s)} nt) no es múltiplo de 3")
        print(f"   Posible frameshift. La traducción puede ser incorrecta.")
    
    # Traducir
    proteina_s = gen_s.translate(to_stop=True)
    longitud_proteina = len(proteina_s)
    
    # ---------------------------------------------------------------------------
    # 6. Validar longitud de la proteína
    # ---------------------------------------------------------------------------
    diferencia = abs(longitud_proteina - LONGITUD_ESPERADA_AA)
    
    if diferencia <= TOLERANCIA_AA:
        estado = "✓ Normal"
    elif longitud_proteina < LONGITUD_ESPERADA_AA - TOLERANCIA_AA:
        estado = "⚠️  TRUNCADA"
        print(f"\n{'='*60}")
        print(f"ADVERTENCIA: Proteína Spike truncada")
        print(f"{'='*60}")
        print(f"Longitud observada: {longitud_proteina} aa")
        print(f"Longitud esperada:  {LONGITUD_ESPERADA_AA} aa")
        print(f"Diferencia:         {-diferencia} aa")
        print(f"\nPosibles causas:")
        print(f"  - Mutación nonsense (STOP prematuro)")
        print(f"  - Frameshift por indel no múltiplo de 3")
        print(f"  - Secuencia incompleta")
        print(f"{'='*60}\n")
    else:
        estado = "⚠️  EXTENDIDA"
        print(f"\n{'='*60}")
        print(f"ADVERTENCIA: Proteína Spike más larga de lo esperado")
        print(f"{'='*60}")
        print(f"Longitud observada: {longitud_proteina} aa")
        print(f"Longitud esperada:  {LONGITUD_ESPERADA_AA} aa")
        print(f"Diferencia:         +{diferencia} aa")
        print(f"\nPosibles causas:")
        print(f"  - Inserción grande")
        print(f"  - Lectura más allá del STOP correcto")
        print(f"{'='*60}\n")
    
    # ---------------------------------------------------------------------------
    # 7. Resumen
    # ---------------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print(f"📊 RESUMEN DE EXTRACCIÓN")
    print(f"{'─'*60}")
    print(f"  Método de búsqueda:    {metodo_busqueda}")
    print(f"  Posición en genoma:    {start_pos:,} - {start_pos + end_pos_relative:,}")
    print(f"  Longitud del gen:      {len(gen_s):,} nt")
    print(f"  Longitud de proteína:  {longitud_proteina} aa")
    print(f"  Estado:                {estado}")
    print(f"{'─'*60}")
    
    # ---------------------------------------------------------------------------
    # 8. Guardar
    # ---------------------------------------------------------------------------
    folder_path = "output/s/spike"
    nombre_base = os.path.basename(archivo_fasta).replace('.fasta', '').replace('.fa', '')
    nombre_archivo = os.path.join(folder_path, f"spike_{nombre_base}.txt")
    
    try:
        os.makedirs(folder_path, exist_ok=True)
        
        with open(nombre_archivo, "w", encoding="utf-8") as f:
            f.write(str(proteina_s))
        
        print(f"\n✅ Proteína guardada en: {nombre_archivo}")
        
        # Guardar también metadatos
        metadata_path = nombre_archivo.replace('.txt', '_metadata.json')
        import json
        
        metadata = {
            "archivo_origen": archivo_fasta,
            "record_id": record.id,
            "longitud_input_nt": longitud_input,
            "tipo_input": "genoma_completo" if es_genoma_completo else "secuencia_parcial",
            "metodo_busqueda": metodo_busqueda,
            "posicion_inicio": start_pos,
            "posicion_fin": start_pos + end_pos_relative,
            "longitud_gen_nt": len(gen_s),
            "longitud_proteina_aa": longitud_proteina,
            "estado": estado,
            "diferencia_vs_wuhan": longitud_proteina - LONGITUD_ESPERADA_AA
        }
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        
        print(f"📋 Metadata guardada en: {metadata_path}")
        
    except OSError as e:
        print(f"❌ Error al guardar archivos: {e}")
        return None
    
    return proteina_s


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 extraer_spike.py <genoma.fasta>")
        print("\nEjemplo:")
        print("  python3 extraer_spike.py wuhan.fasta")
        print("  python3 extraer_spike.py omicron_BA1.fasta")
        print("\nSoporta:")
        print("  - Genoma completo SARS-CoV-2 (~30kb)")
        print("  - Secuencia parcial (solo región Spike)")
        print("\nOutput:")
        print("  - spike_nombre.txt        (secuencia de aminoácidos)")
        print("  - spike_nombre_metadata.json  (información del proceso)")
        sys.exit(1)
    
    resultado = procesar_referencia(sys.argv[1])
    
    if resultado is None:
        sys.exit(1)