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
                en_bloque = True
                inicio = i
        else:
            if en_bloque:
                bloques.append((inicio, i, i - inicio))
                en_bloque = False
    if en_bloque:
        bloques.append((inicio, len(secuencia), len(secuencia) - inicio))
    return bloques

def imputar_secuencia(var_path, ref_path, umbral_bloque=UMBRAL_BLOQUE_GRANDE):
    # 1. Leer secuencias
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
    
    # 2. Detectar bloques
    bloques = detectar_bloques_x(seq_variante)
    bloques_grandes = [b for b in bloques if b[2] >= umbral_bloque]
    
    # 3. Imputación "Espejo" (In-place)
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
                        "res": res_ref
                    })

    # 4. Reconstrucción Final y Validación
    seq_final = "".join(seq_imputada_lista)
    
    if len(seq_final) != longitud_total:
        raise ValueError(f"Diferencia de longitud detectada: {len(seq_final)} vs {longitud_total}")

    # 5. Guardado de Archivos
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