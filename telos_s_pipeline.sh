#!/bin/bash

# Comprobar si se pasó el archivo fasta como argumento
if [ [ $# -lt 1 ] ]; then
    echo "Uso: ./telos_analize_pipeline.sh <variante.fasta> [referencia.fasta]"
    echo "Si no se provee referencia, se usará 'wuhan_ref.fasta' por defecto."
    exit 1
fi

# Configuración de variables
VAR_GENOMA=$1
REF_GENOMA=${2:-"wuhan_ref.fasta"} # Usa wuhan_ref.fasta si no se especifica otro

# Extraer nombres base para manejar las rutas (quita extensiones y rutas)
VAR_NAME=$(basename "$VAR_GENOMA" .fasta)
REF_NAME=$(basename "$REF_GENOMA" .fasta)

echo "🚀 Iniciando Pipeline de Análisis BioAlerta para: $VAR_NAME"
echo "----------------------------------------------------------"

# Paso 1: Extracción de la proteína Spike
echo "🧬 [1/6] Extrayendo secuencia Spike..."
python3 extraer_spike.py "$REF_GENOMA"
python3 extraer_spike.py "$VAR_GENOMA"

# Definir rutas de salida del paso 1
SPIKE_REF="output/s/spike/spike_${REF_NAME}.txt"
SPIKE_VAR="output/s/spike/spike_${VAR_NAME}.txt"

# Paso 2: Alineamiento de secuencias
echo "📏 [2/6] Alineando secuencias (Sincronización)..."
python3 alineador_secuencias.py "$SPIKE_REF" "$SPIKE_VAR"

# Definir rutas de salida del paso 2
ALIGNED_REF="output/s/spike_aligned/spike_${REF_NAME}.txt"
ALIGNED_VAR="output/s/spike_aligned/spike_${VAR_NAME}.txt"

# Paso 3: Imputar secuencias dañadas
echo "[3/6] Imputando secuencias dañadas"
python3 imputar_secuencia.py "$ALIGNED_VAR" "$ALIGNED_REF"

INDICTED_JSON="output/prophet/imputacion_spike_${VAR_NAME}.json"

# Paso 4: Obteniendo mutaciones en las posiciones criticas
echo "[4/6] Obteniendo mutaciones posibles en 452, 484, 501 y 681..."
python3 oraculo_mutaciones.py "$ALIGNED_VAR" "$INDICTED_JSON" --cpu

# Paso 5: Comparación Inteligente (ESM-2)
echo "🧠 [5/6] Ejecutando análisis de IA con ESM-2..."
python3 comparador_inteligente.py "$ALIGNED_REF" "$ALIGNED_VAR" --cpu

# El nombre del CSV generado por tu script depende de la ruta del var_path
REPORT_CSV="output/s/report/reporte_spike_${VAR_NAME}.csv"

# Paso 6: Análisis Final y Visualización
echo "📊 [6/6] Generando reporte ejecutivo y heatmap..."
python3 analizador_final.py "$REPORT_CSV"

echo "----------------------------------------------------------"
echo "✅ Análisis completado con éxito."
echo "Reporte: informe_ejecutivo_spike_${VAR_NAME}.txt"
echo "Heatmap: output/s/report/heatmap_spike_${VAR_NAME}.png"