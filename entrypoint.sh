#!/bin/bash
# =============================================================================
# TELOS-S — entrypoint.sh
# Ejecutado al arrancar el contenedor.
# Si la referencia Wuhan no existe, la descarga de NCBI y extrae la Spike.
# Luego arranca el servidor FastAPI.
# =============================================================================
 
set -e
 
WUHAN_FASTA="/app/wuhan_ref.fasta"
WUHAN_SPIKE="/app/spike_wuhan.txt"
NCBI_URL="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_045512.2&rettype=fasta&retmode=text"
 
echo "========================================"
echo "  TELOS-S Backend — Starting up"
echo "========================================"
 
# -----------------------------------------------------------------------
# PASO 1: Verificar / descargar referencia Wuhan
# -----------------------------------------------------------------------
if [ -f "$WUHAN_SPIKE" ]; then
    echo "✅ Wuhan Spike reference found at $WUHAN_SPIKE — skipping download"
else
    echo "📥 Wuhan Spike reference not found. Downloading NC_045512.2 from NCBI..."
 
    # Intentar descarga (con reintentos)
    MAX_RETRIES=3
    RETRY=0
    SUCCESS=false
 
    while [ $RETRY -lt $MAX_RETRIES ]; do
        if curl -f -L --retry 3 --retry-delay 5 \
            -o "$WUHAN_FASTA" \
            "$NCBI_URL"; then
            SUCCESS=true
            break
        else
            RETRY=$((RETRY + 1))
            echo "⚠️  Download attempt $RETRY failed. Retrying in 10s..."
            sleep 10
        fi
    done
 
    if [ "$SUCCESS" = false ]; then
        echo ""
        echo "❌ ERROR: Could not download NC_045512.2 from NCBI after $MAX_RETRIES attempts."
        echo ""
        echo "   Possible causes:"
        echo "   - No internet access from the container"
        echo "   - NCBI temporarily unavailable"
        echo ""
        echo "   Manual fix: copy the reference file to /app/data/spike_wuhan.txt"
        echo "   and restart the container."
        echo ""
        echo "   You can also mount it via docker-compose:"
        echo "     volumes:"
        echo "       - ./spike_wuhan.txt:/app/data/spike_wuhan.txt:ro"
        echo ""
        exit 1
    fi
 
    echo "✅ NC_045512.2 downloaded ($(wc -c < "$WUHAN_FASTA") bytes)"
    echo ""
 
    # -----------------------------------------------------------------------
    # PASO 2: Extraer la proteína Spike
    # -----------------------------------------------------------------------
    echo "🧬 Extracting Spike protein from Wuhan genome..."
 
    python3 modules/extraer_spike.py "$WUHAN_FASTA"
 
    # extraer_spike.py guarda en output/s/spike/spike_<nombre>.txt
    # necesitamos copiarlo a la ubicación estándar que usa el pipeline
    EXTRACTED=$(find /app/output/s/spike -name "spike_*.txt" | head -1)
 
    if [ -z "$EXTRACTED" ]; then
        echo "❌ ERROR: extraer_spike.py ran but no output file was found."
        echo "   Check the script logs above for errors."
        exit 1
    fi
 
    cp "$EXTRACTED" "$WUHAN_SPIKE"
    echo "✅ Spike extracted and saved to $WUHAN_SPIKE"
    echo "   Length: $(wc -c < "$WUHAN_SPIKE") characters"
    echo ""
fi
 
# -----------------------------------------------------------------------
# PASO 3: Verificar que el spike de referencia tiene longitud correcta
# -----------------------------------------------------------------------
SPIKE_LEN=$(wc -c < "$WUHAN_SPIKE" | tr -d ' ')
 
if [ "$SPIKE_LEN" -lt 1273 ] || [ "$SPIKE_LEN" -gt 1300 ]; then
    echo "⚠️  WARNING: Spike reference has unexpected length ($SPIKE_LEN chars, expected ~1273)"
    echo "   The file may be corrupted. Delete $WUHAN_SPIKE and restart to re-download."
fi
 
# -----------------------------------------------------------------------
# PASO 4: Crear directorios de output necesarios para el pipeline
# -----------------------------------------------------------------------
mkdir -p \
    /app/output/uploads \
    /app/output/jobs \
    /app/output/s/spike \
    /app/output/s/spike_aligned \
    /app/output/s/report \
    /app/output/prophet
 
echo "✅ Output directories ready"
echo ""
 
# -----------------------------------------------------------------------
# PASO 5: Arrancar el servidor FastAPI
# -----------------------------------------------------------------------
echo "🚀 Starting Telos-S API on port 8000..."
echo ""
 
exec python backend.py