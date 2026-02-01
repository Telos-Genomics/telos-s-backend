import os
import sys
from Bio import Align

# ---------------------------------------------------------------------------
# ALINEADOR DE SECUENCIAS SPIKE
# ---------------------------------------------------------------------------
# Estrategia: Template-based alignment
#
# La referencia (Wuhan) define la NUMERACIÓN CANÓNICA de 1273 posiciones.
# La variante se alinea CONTRA esta referencia como template.
#
# Reglas:
#   1. ref_aligned SIEMPRE tiene 1273 caracteres (la secuencia original sin gaps)
#   2. var_aligned tiene 1273 caracteres con gaps '-' donde hay deleciones
#   3. Inserciones en la variante (residuos extra vs Wuhan) se DESCARTAN
#      porque no tienen posición canónica en la numeración de Wuhan
#
# Esto garantiza que:
#   - Ambas secuencias tienen exactamente 1273 caracteres
#   - La numeración es consistente con Wuhan
#   - El oráculo y el comparador funcionan correctamente
# ---------------------------------------------------------------------------


def validar_referencia(seq, esperada=1273):
    """
    Valida que la secuencia de referencia tiene la longitud esperada.
    
    Args:
        seq: Secuencia de referencia (sin gaps)
        esperada: Longitud esperada (1273 para Spike de Wuhan)
    
    Returns:
        (es_valida, mensaje)
    """
    longitud = len(seq)
    
    if longitud != esperada:
        return False, (
            f"❌ La referencia tiene {longitud} aminoácidos, "
            f"pero se esperaban {esperada}.\n"
            f"   Asegúrate de usar la proteína Spike completa de Wuhan-Hu-1 "
            f"(GenBank: QHD43416.1 o UniProt: P0DTC2)."
        )
    
    # Verificar que no contiene gaps
    if '-' in seq:
        return False, (
            f"❌ La secuencia de referencia contiene gaps '-'.\n"
            f"   La referencia debe ser la secuencia original sin alinear."
        )
    
    # Verificar que solo contiene aminoácidos válidos
    aa_validos = set("ACDEFGHIKLMNPQRSTVWY")
    invalidos = set(seq) - aa_validos
    
    if invalidos:
        return False, (
            f"❌ La referencia contiene caracteres inválidos: {invalidos}.\n"
            f"   Solo se permiten los 20 aminoácidos estándar."
        )
    
    return True, ""


def alinear_template_based(ref_seq, var_seq):
    """
    Alinea la variante contra la referencia usando template-based alignment.
    
    La referencia define la numeración canónica. La variante puede tener:
      - DELECIONES: se insertan gaps '-' en var_aligned
      - SUSTITUCIONES: aparecen como diferencias sin gaps
      - INSERCIONES: se descartan (no tienen posición canónica en Wuhan)
    
    Args:
        ref_seq: Secuencia de referencia (Wuhan, 1273 aa)
        var_seq: Secuencia de la variante (longitud variable)
    
    Returns:
        (ref_aligned, var_aligned): Tupla de secuencias alineadas
        Ambas tienen exactamente len(ref_seq) caracteres.
    """
    print(f"\n🔄 Alineando secuencias...")
    print(f"   Referencia: {len(ref_seq)} aa")
    print(f"   Variante:   {len(var_seq)} aa")
    
    # ---------------------------------------------------------------------------
    # 1. Alineamiento global con penalizaciones optimizadas
    # ---------------------------------------------------------------------------
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    
    # Penalizaciones para gaps:
    #   - open_gap_score: penalización por ABRIR un gap (primera posición)
    #   - extend_gap_score: penalización por EXTENDER un gap (posiciones consecutivas)
    #
    # Queremos penalizar fuertemente los gaps para evitar que se inventen
    # alineamientos raros, pero permitir deleciones reales conocidas.
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    
    # Match/mismatch scores
    aligner.match_score = 2
    aligner.mismatch_score = -1
    
    alignments = aligner.align(ref_seq, var_seq)
    
    if len(alignments) == 0:
        raise ValueError("No se pudo generar ningún alineamiento. Verifica que las secuencias son comparables.")
    
    mejor = alignments[0]
    
    # Estas son las secuencias alineadas que devuelve Biopython
    ref_con_gaps = str(mejor[0])
    var_con_gaps = str(mejor[1])
    
    print(f"   Alineamiento inicial: {len(ref_con_gaps)} caracteres")
    
    # ---------------------------------------------------------------------------
    # 2. Post-procesamiento: Remover gaps de la referencia
    # ---------------------------------------------------------------------------
    # Si Biopython insertó gaps en la referencia (porque la variante tiene
    # inserciones), los removemos y descartamos las posiciones correspondientes
    # en la variante también.
    
    ref_final = []
    var_final = []
    
    for i in range(len(ref_con_gaps)):
        ref_char = ref_con_gaps[i]
        var_char = var_con_gaps[i]
        
        if ref_char == '-':
            # Gap en referencia → inserción en variante
            # Descartamos esta posición completa
            print(f"   ⚠️  Inserción en variante en posición ~{i+1} descartada: '{var_char}'")
            continue
        
        # Si llegamos aquí, ref_char es un aminoácido válido
        ref_final.append(ref_char)
        var_final.append(var_char)  # Puede ser un aa o '-' (deleción)
    
    ref_aligned = "".join(ref_final)
    var_aligned = "".join(var_final)
    
    # ---------------------------------------------------------------------------
    # 3. Validación final
    # ---------------------------------------------------------------------------
    longitud_final = len(ref_aligned)
    
    if longitud_final != len(ref_seq):
        # Esto NO debería pasar si removimos todos los gaps de la ref
        raise AssertionError(
            f"Error interno: ref_aligned tiene {longitud_final} caracteres, "
            f"pero debería tener {len(ref_seq)}. "
            f"Esto indica un problema en el post-procesamiento."
        )
    
    if len(var_aligned) != longitud_final:
        raise AssertionError(
            f"Error interno: las secuencias alineadas tienen longitudes diferentes "
            f"(ref: {longitud_final}, var: {len(var_aligned)})"
        )
    
    # Contar deleciones e inserciones
    num_deleciones = var_aligned.count('-')
    num_inserciones = len(ref_con_gaps) - longitud_final  # gaps que removimos de ref
    num_sustituciones = sum(1 for r, v in zip(ref_aligned, var_aligned)
                           if r != v and v != '-')
    
    print(f"\n✅ Alineamiento completado:")
    print(f"   Longitud final: {longitud_final} posiciones")
    print(f"   Sustituciones:  {num_sustituciones}")
    print(f"   Deleciones:     {num_deleciones}")
    print(f"   Inserciones:    {num_inserciones} (descartadas)")
    
    return ref_aligned, var_aligned


def alinear_sincronizar(ref_path, var_path):
    """
    Lee las secuencias de referencia y variante, las alinea, y guarda el resultado.
    
    Args:
        ref_path: Ruta al archivo con la secuencia de referencia (Wuhan)
        var_path: Ruta al archivo con la secuencia de la variante
    """
    # ---------------------------------------------------------------------------
    # 1. Leer secuencias
    # ---------------------------------------------------------------------------
    with open(ref_path, "r") as f:
        ref_seq = f.read().strip()
    
    with open(var_path, "r") as f:
        var_seq = f.read().strip()
    
    # ---------------------------------------------------------------------------
    # 2. Validar referencia
    # ---------------------------------------------------------------------------
    es_valida, mensaje = validar_referencia(ref_seq, esperada=1273)
    if not es_valida:
        print(mensaje)
        print("\n💡 Tip: Descarga la secuencia de Spike de Wuhan-Hu-1:")
        print("   GenBank: QHD43416.1")
        print("   UniProt: P0DTC2")
        sys.exit(1)
    
    print("✅ Referencia validada: Spike de Wuhan, 1273 aa")
    
    # ---------------------------------------------------------------------------
    # 3. Alinear
    # ---------------------------------------------------------------------------
    try:
        ref_aligned, var_aligned = alinear_template_based(ref_seq, var_seq)
    except Exception as e:
        print(f"\n❌ Error durante el alineamiento: {e}")
        sys.exit(1)
    
    # ---------------------------------------------------------------------------
    # 4. Guardar resultados
    # ---------------------------------------------------------------------------
    folder_path = "output/s/spike_aligned"
    
    # Generar nombres de archivo basados en los paths de entrada
    nombre_ref_base = os.path.basename(ref_path).replace('.txt', '')
    nombre_var_base = os.path.basename(var_path).replace('.txt', '')
    
    ruta_ref_final = os.path.join(folder_path, f"{nombre_ref_base}.txt")
    ruta_var_final = os.path.join(folder_path, f"{nombre_var_base}.txt")
    
    try:
        os.makedirs(folder_path, exist_ok=True)
        
        with open(ruta_ref_final, "w", encoding="utf-8") as f:
            f.write(ref_aligned)
        
        with open(ruta_var_final, "w", encoding="utf-8") as f:
            f.write(var_aligned)
        
        print(f"\n📁 Secuencias alineadas guardadas:")
        print(f"   Referencia: {ruta_ref_final}")
        print(f"   Variante:   {ruta_var_final}")
        
    except OSError as e:
        print(f"❌ Error al guardar archivos: {e}")
        sys.exit(1)
    
    # ---------------------------------------------------------------------------
    # 5. Verificación final (paranoia)
    # ---------------------------------------------------------------------------
    with open(ruta_ref_final, "r") as f:
        test_ref = f.read().strip()
    with open(ruta_var_final, "r") as f:
        test_var = f.read().strip()
    
    assert len(test_ref) == 1273, f"Error: ref_aligned tiene {len(test_ref)} chars"
    assert len(test_var) == 1273, f"Error: var_aligned tiene {len(test_var)} chars"
    assert '-' not in test_ref, "Error: la referencia alineada contiene gaps"
    
    print("✅ Verificación exitosa: ambas secuencias tienen 1273 caracteres")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python3 alineador_secuencias.py <referencia.txt> <variante.txt>")
        print("\nEjemplo:")
        print("  python3 alineador_secuencias.py spike_wuhan.txt spike_omicron.txt")
        print("\nNota:")
        print("  La referencia debe ser la proteína Spike completa de Wuhan-Hu-1 (1273 aa)")
        print("  La variante puede tener cualquier longitud")
        sys.exit(1)
    
    alinear_sincronizar(sys.argv[1], sys.argv[2])