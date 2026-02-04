#!/bin/bash

# Comprobar si se pasó el archivo fasta como argumento
if [ [ $# -lt 1 ] ]; then
    echo "Uso: ./telos_prophet_pipeline.sh <variante.txt>"
    echo "Si no se provee referencia, se usará 'spike_wuhan_ref.txt' por defecto."
    exit 1
fi

# 1. Configuración de variables
REF_GENOMA=$1

# Extraer nombres base para manejar las rutas (quita extensiones y rutas)
REF_NAME=$(basename "$REF_GENOMA" .txt)

echo "🚀 Iniciando Pipeline de Análisis de probabilidades para: $REF_NAME"
echo "----------------------------------------------------------"

# 2. Paso 1: Extracción de la proteína Spike
echo "🧬 Extrayendo secuencia Spike..."
python3 oraculo_mutaciones.py "$REF_GENOMA" --cpu

# El nombre del CSV generado por tu script depende de la ruta del var_path
REPORT_CSV="output/s/report/reporte_${REF_NAME}.csv"

# 5. Paso 4: Análisis Final y Visualización
echo "📊 [4/4] Generando reporte ejecutivo y heatmap..."
python3 analizador_final.py "$REPORT_CSV"

echo "----------------------------------------------------------"
echo "✅ Análisis completado con éxito."