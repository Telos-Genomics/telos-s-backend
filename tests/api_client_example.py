"""
Cliente de ejemplo para Telos-S API
------------------------------------
Demuestra cómo interactuar con la API programáticamente.
"""

import requests
import time
import json
from pathlib import Path

# Configuración
API_BASE_URL = "http://localhost:8000"

def check_health():
    """Verifica que la API esté corriendo"""
    response = requests.get(f"{API_BASE_URL}/health")
    return response.json()

def upload_and_analyze(variant_fasta_path: str, reference_fasta_path: str = None):
    """
    Sube un FASTA y ejecuta análisis.
    
    Returns:
        job_id para consultar el estado
    """
    files = {
        'variant_file': open(variant_fasta_path, 'rb')
    }
    
    if reference_fasta_path:
        files['reference_file'] = open(reference_fasta_path, 'rb')
    
    params = {
        'use_cpu': True,
        'impute_gaps': True
    }
    
    response = requests.post(
        f"{API_BASE_URL}/api/v1/analysis/upload",
        files=files,
        params=params
    )
    
    # Cerrar archivos
    for f in files.values():
        f.close()
    
    if response.status_code == 202:
        return response.json()['job_id']
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

def get_status(job_id: str):
    """Consulta el estado de un análisis"""
    response = requests.get(f"{API_BASE_URL}/api/v1/analysis/{job_id}")
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

def wait_for_completion(job_id: str, max_wait_seconds=600, poll_interval=5):
    """
    Espera a que el análisis se complete.
    
    Args:
        job_id: ID del job
        max_wait_seconds: Tiempo máximo de espera
        poll_interval: Segundos entre consultas
    
    Returns:
        Estado final del job
    """
    start_time = time.time()
    
    while True:
        status = get_status(job_id)
        
        print(f"Estado: {status['status']} | "
              f"Progreso: {status['progress']*100:.0f}% | "
              f"Paso: {status['current_step']}")
        
        if status['status'] == 'completed':
            print("✅ Análisis completado!")
            return status
        
        elif status['status'] == 'failed':
            print(f"❌ Análisis falló: {status.get('error', 'Error desconocido')}")
            return status
        
        # Verificar timeout
        if time.time() - start_time > max_wait_seconds:
            print(f"⏱️  Timeout: análisis no completó en {max_wait_seconds}s")
            return status
        
        # Esperar antes de la siguiente consulta
        time.sleep(poll_interval)

def get_results(job_id: str):
    """Obtiene los resultados completos"""
    response = requests.get(f"{API_BASE_URL}/api/v1/analysis/{job_id}/results")
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

def download_file(job_id: str, file_type: str, output_path: str):
    """
    Descarga un archivo generado.
    
    file_type: 'csv', 'heatmap', 'informe'
    """
    response = requests.get(
        f"{API_BASE_URL}/api/v1/analysis/{job_id}/files/{file_type}",
        stream=True
    )
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Archivo descargado: {output_path}")
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

def run_simulation(job_id: str, scenario: str = "airport", location: str = "PTY", days: int = 30):
    """Ejecuta simulación epidemiológica (placeholder en MVP)"""
    payload = {
        "job_id": job_id,
        "scenario": scenario,
        "location_code": location,
        "duration_days": days,
        "initial_cases": 1
    }
    
    response = requests.post(
        f"{API_BASE_URL}/api/v1/simulation/run",
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# ============================================================================
# EJEMPLO DE USO COMPLETO
# ============================================================================

def ejemplo_completo():
    """Flujo completo de análisis"""
    
    print("="*60)
    print("EJEMPLO DE USO DE TELOS-S API")
    print("="*60)
    
    # 1. Health check
    print("\n1️⃣  Verificando que la API esté corriendo...")
    try:
        health = check_health()
        print(f"   ✅ API operacional: {health}")
    except Exception as e:
        print(f"   ❌ API no disponible: {e}")
        return
    
    # 2. Upload y análisis
    print("\n2️⃣  Subiendo FASTA y ejecutando análisis...")
    
    # CAMBIAR ESTAS RUTAS por tus archivos reales
    variant_path = "omicron_ba286.fasta"
    reference_path = "wuhan_ref.fasta"  # O None para usar el del servidor
    
    if not Path(variant_path).exists():
        print(f"   ⚠️  Archivo no encontrado: {variant_path}")
        print("   Modifica 'variant_path' en el script con una ruta válida")
        return
    
    try:
        job_id = upload_and_analyze(variant_path, reference_path)
        print(f"   ✅ Análisis iniciado. Job ID: {job_id}")
    except Exception as e:
        print(f"   ❌ Error al subir: {e}")
        return
    
    # 3. Esperar a que complete
    print(f"\n3️⃣  Esperando a que complete (job_id: {job_id})...")
    final_status = wait_for_completion(job_id)
    
    if final_status['status'] != 'completed':
        print("   Análisis no completó exitosamente")
        return
    
    # 4. Obtener resultados
    print("\n4️⃣  Obteniendo resultados...")
    results = get_results(job_id)
    
    print(f"\n   📊 RESULTADOS:")
    print(f"   Variante: {results['variant_name']}")
    print(f"   Aggression Score: {results['aggression_score']:.1f}")
    print(f"   Linaje: {results['lineage']}")
    print(f"   Mutaciones detectadas: {len(results['mutations'])}")
    
    # Mostrar parámetros epidemiológicos
    epi = results['epi_params']
    print(f"\n   🦠 PARÁMETROS EPIDEMIOLÓGICOS:")
    print(f"   R0 estimado: {epi['r0_estimated']}")
    print(f"   Periodo incubación: {epi['incubation_period_days']} días")
    print(f"   Transmisibilidad base: {epi['transmissibility_base']*100:.1f}%")
    
    # 5. Descargar archivos
    print("\n5️⃣  Descargando archivos generados...")
    try:
        download_file(job_id, "csv", f"results_{job_id}.csv")
        download_file(job_id, "heatmap", f"heatmap_{job_id}.png")
        download_file(job_id, "informe", f"informe_{job_id}.txt")
    except Exception as e:
        print(f"   ⚠️  Error descargando archivos: {e}")
    
    # 6. Simulación (placeholder)
    print("\n6️⃣  Ejecutando simulación epidemiológica...")
    try:
        sim_result = run_simulation(job_id, scenario="airport", location="PTY", days=30)
        print(f"   ✅ Simulación: {sim_result['message']}")
        print(f"   Parámetros disponibles para visualización frontend")
    except Exception as e:
        print(f"   ⚠️  Error en simulación: {e}")
    
    print("\n" + "="*60)
    print("✅ FLUJO COMPLETO EJECUTADO")
    print("="*60)

if __name__ == "__main__":
    ejemplo_completo()