import os
import sys
import json

# ---------------------------------------------------------------------------
# IMPUTADOR DE SECUENCIAS TELOS-S (VERSIÓN 2.0)
# ---------------------------------------------------------------------------

UMBRAL_BLOQUE_GRANDE = 5 

def detectar_bloques_x(secuencia):
    bloques = []
    en_bloque = False
    inicio = None
    
    for i, aa in enumerate(secuencia):
        if aa == 'X':
            if not en_bloque:
                # Inicio de un nuevo bloque
                en_bloque = True
                inicio = i
        else:
            if en_bloque:
                # Fin del bloque
                bloques.append((inicio, i, i - inicio))
                en_bloque = False
    # Si terminamos en bloque
    if en_bloque:
        bloques.append((inicio, len(secuencia), len(secuencia) - inicio))
    return bloques

def imputar_secuencia(var_path, ref_path, umbral_bloque=UMBRAL_BLOQUE_GRANDE):
    """
    Imputa bloques grandes de X desde la secuencia de referencia.
    
    Args:
        var_path: Ruta a la secuencia con X (variante)
        ref_path: Ruta a la secuencia de referencia (Wuhan o linaje cercano)
        umbral_bloque: Tamaño mínimo de bloque para imputar (default: 5)
    
    Output:
        - Secuencia imputada en output/s/spike_aligned/
        - Reporte JSON en output/prophet/
    """
    # ---------------------------------------------------------------------------
    # 1. Leer secuencias
    # ---------------------------------------------------------------------------
    with open(var_path, "r") as f:
        seq_variante = f.read().strip()
    with open(ref_path, "r") as f:
        seq_referencia = f.read().strip()
    
    # VALIDACIÓN CRÍTICA DE ALINEAMIENTO
    if len(seq_variante) != len(seq_referencia):
        print(f"❌ ERROR FATAL: Las secuencias no están alineadas.")
        print(f"Var: {len(seq_variante)} | Ref: {len(seq_referencia)}")
        sys.exit(1)
    
    longitud_total = len(seq_variante)
    
    # ---------------------------------------------------------------------------
    # 2. Detectar bloques de X
    # ---------------------------------------------------------------------------
    bloques = detectar_bloques_x(seq_variante)
    num_x_total = sum(1 for aa in seq_variante if aa == 'X')

    print(f"\n🔍 Análisis de posiciones X:")
    print(f"   Total de X: {num_x_total}")
    print(f"   Bloques detectados: {len(bloques)}")

    # Clasificar bloques
    bloques_grandes = [b for b in bloques if b[2] >= umbral_bloque]
    bloques_pequeños = [b for b in bloques if b[2] < umbral_bloque]

    x_en_bloques_pequeños = sum(b[2] for b in bloques_pequeños)
    x_en_bloques_grandes = sum(b[2] for b in bloques_grandes)
    
    print(f"\n   Bloques pequeños (<{umbral_bloque} X): {len(bloques_pequeños)}  ({x_en_bloques_pequeños} posiciones)")
    print(f"   Bloques grandes (≥{umbral_bloque} X):  {len(bloques_grandes)}  ({x_en_bloques_grandes} posiciones)")

    if bloques_grandes:
        print(f"\n   Detalles de bloques grandes:")
        for inicio, fin, longitud in bloques_grandes:
            # Calcular posición Wuhan (1-indexed, contando aminoácidos hasta ese punto)
            pos_inicio_wuhan = sum(1 for c in seq_variante[:inicio] if c not in ['-', 'X']) + 1
            pos_fin_wuhan = sum(1 for c in seq_variante[:fin] if c not in ['-', 'X'])
            print(f"      Índices {inicio}-{fin} ({longitud} X) → posiciones Wuhan ~{pos_inicio_wuhan}-{pos_fin_wuhan}")

    # ---------------------------------------------------------------------------
    # 3. Imputación "Espejo" (In-place)
    # ---------------------------------------------------------------------------
    # Usamos una lista para evitar la inmutabilidad de los strings y errores de concatenación
    seq_imputada_lista = list(seq_variante)
    posiciones_imputadas = []
    
    # Pre-calculamos un mapa de posiciones Wuhan para el reporte
    # Esto evita calcular sum() dentro de los bucles anidados
    mapa_wuhan = []
    contador_aa = 0
    for char in seq_variante:
        if char != '-':
            contador_aa += 1
            mapa_wuhan.append(contador_aa)
        else:
            mapa_wuhan.append(None) # Es un gap, no tiene posición Wuhan

    for inicio, fin, longitud in bloques_grandes:
        for i in range(inicio, fin):
            if seq_variante[i] == 'X':
                # REGLA DE ORO: Solo imputamos si la referencia tiene un dato útil (no gap ni X)
                res_ref = seq_referencia[i]
                if res_ref not in ['-', 'X']:
                    seq_imputada_lista[i] = res_ref
                    posiciones_imputadas.append({
                        "idx": i,
                        "wuhan_pos": mapa_wuhan[i],
                        "num_bloques_total": len(bloques),
                        "num_bloques_pequeños": len(bloques_pequeños),
                        "num_bloques_grandes": len(bloques_grandes),
                        "res": res_ref
                    })

    # ---------------------------------------------------------------------------
    # 4. Reconstruir secuencia completa con gaps
    # ---------------------------------------------------------------------------
    seq_final = "".join(seq_imputada_lista)
    
    if len(seq_final) != longitud_total:
        raise ValueError(f"Diferencia de longitud detectada: {len(seq_final)} vs {longitud_total}")

    # ---------------------------------------------------------------------------
    # 5. Guardar secuencia reconstruida
    # ---------------------------------------------------------------------------
    nombre_base = os.path.basename(var_path).replace('.txt', '')
    ruta_out = f"output/s/spike_aligned/{nombre_base}_imputada.txt"
    os.makedirs("output/s/spike_aligned", exist_ok=True)
    
    with open(ruta_out, "w") as f:
        f.write(seq_final)

    # Generar JSON de metadatos para el Analizador
    ruta_json = f"output/prophet/imputacion_{nombre_base}.json"
    os.makedirs("output/prophet", exist_ok=True)
    with open(ruta_json, "w") as f:
        json.dump({
            "metodo": "Imputación por Referencia",
            "total_imputados": len(posiciones_imputadas),
            "posiciones": posiciones_imputadas
        }, f, indent=4)

    print(f"✅ Proceso exitoso. {len(posiciones_imputadas)} posiciones restauradas.")
    print(f"📄 Archivo: {ruta_out}")

if __name__ == "__main__":
    imputar_secuencia(sys.argv[1], sys.argv[2])