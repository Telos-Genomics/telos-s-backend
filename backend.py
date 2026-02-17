"""
Telos-S API
-----------
API REST para análisis de variantes SARS-CoV-2 con predicción de impacto epidemiológico.

Arquitectura:
    - FastAPI backend
    - Procesamiento asíncrono de análisis largos
    - Almacenamiento de resultados en JSON
    - Ready para conectar con frontend de simulación
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
import subprocess
import json
import uuid
import shutil
from datetime import datetime
import os

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

app = FastAPI(
    title="Telos-S API",
    description="Análisis genómico de variantes SARS-CoV-2 con predicción epidemiológica",
    version="0.1.0-mvp",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS para permitir requests desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorios
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
JOBS_DIR = OUTPUT_DIR / "jobs"

# Crear directorios si no existen
for directory in [UPLOAD_DIR, JOBS_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================================================
# MODELOS DE DATOS (Pydantic)
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request para iniciar análisis de variante"""
    variant_name: Optional[str] = Field(None, description="Nombre de la variante (opcional)")
    use_cpu: bool = Field(True, description="Forzar uso de CPU en lugar de GPU")
    impute_gaps: bool = Field(True, description="Imputar bloques grandes de X desde referencia")
    
    class Config:
        schema_extra = {
            "example": {
                "variant_name": "Omicron_BA.2.86",
                "use_cpu": True,
                "impute_gaps": True
            }
        }


class AnalysisStatus(BaseModel):
    """Estado de un análisis en progreso"""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 - 1.0
    current_step: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class VariantMutation(BaseModel):
    """Mutación individual en la proteína Spike"""
    mutation: str
    position: int
    zone: str
    llr: float
    score: float
    confidence: str  # CONFIABLE, SOSPECHOSA, IMPUTADA, INVALIDA


class AnalysisResults(BaseModel):
    """Resultados completos del análisis"""
    job_id: str
    variant_name: str
    aggression_score: float
    lineage: str
    lineage_confidence: float
    sequence_quality: float
    mutations: List[VariantMutation]
    prophet_predictions: Optional[List[Dict[str, Any]]] = None
    
    # Parámetros epidemiológicos derivados (para simulación)
    epi_params: Dict[str, float] = Field(
        default_factory=dict,
        description="Parámetros epidemiológicos calculados (R0, transmisibilidad, etc.)"
    )
    
    # Metadata
    processed_at: datetime
    files: Dict[str, str]  # Rutas a archivos generados


class SimulationRequest(BaseModel):
    """Request para ejecutar simulación epidemiológica"""
    job_id: str = Field(..., description="ID del análisis de variante")
    scenario: str = Field(..., description="Tipo de escenario: 'airport' o 'urban'")
    location_code: str = Field(..., description="Código de localización (ej: 'PTY' para aeropuerto)")
    duration_days: int = Field(30, ge=1, le=365, description="Duración de simulación en días")
    initial_cases: int = Field(1, ge=1, description="Número de casos iniciales")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "abc123",
                "scenario": "airport",
                "location_code": "PTY",
                "duration_days": 30,
                "initial_cases": 1
            }
        }


# ============================================================================
# UTILIDADES
# ============================================================================

def create_job_id() -> str:
    """Genera ID único para el job"""
    return f"job_{uuid.uuid4().hex[:12]}"


def save_job_status(job_id: str, status_data: dict):
    """Guarda estado del job en JSON"""
    job_file = JOBS_DIR / f"{job_id}.json"
    with open(job_file, "w") as f:
        json.dump(status_data, f, indent=2, default=str)


def load_job_status(job_id: str) -> dict:
    """Carga estado del job desde JSON"""
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} no encontrado")
    
    with open(job_file, "r") as f:
        return json.load(f)


def run_pipeline_step(command: List[str], step_name: str) -> dict:
    """
    Ejecuta un paso del pipeline y captura output.
    
    Returns:
        dict con "success", "stdout", "stderr"
    """
    try:
        result = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutos max por paso
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Timeout en paso {step_name}",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }


def calculate_epi_parameters(results_csv: Path) -> dict:
    """
    Calcula parámetros epidemiológicos a partir del CSV de mutaciones.
    
    Esto es el "puente" entre Telos-S y Telos-SIM.
    """
    import pandas as pd
    
    try:
        df = pd.read_csv(results_csv)
        
        # Filtrar solo mutaciones confiables
        df_confiable = df[df['Confiabilidad'] == 'CONFIABLE'].copy()
        
        # Aggression Score total
        aggression_score = df_confiable['Score'].abs().sum()
        
        # Score de RBM (afinidad ACE2)
        rbm_mutations = df_confiable[
            (df_confiable['Pos'] >= 437) & 
            (df_confiable['Pos'] <= 508)
        ]
        rbm_score = rbm_mutations['Score'].abs().sum()
        
        # Score de Furina (eficiencia entrada)
        furina_mutations = df_confiable[
            (df_confiable['Pos'] >= 681) & 
            (df_confiable['Pos'] <= 685)
        ]
        furina_score = furina_mutations['Score'].abs().sum()
        
        # Calcular R0 estimado
        # Base (Wuhan) = 2.5
        # Cada 100 puntos RBM → +0.2 R0
        # Cada 50 puntos Furina → +0.15 R0
        r0_estimated = 2.5 + (rbm_score / 100) * 0.2 + (furina_score / 50) * 0.15
        r0_estimated = min(r0_estimated, 8.0)  # Cap en 8.0 (realista)
        
        # Calcular periodo de incubación estimado
        # Base (Wuhan) = 5.5 días
        # Cada 100 puntos Furina → -0.3 días
        incubation_period = 5.5 - (furina_score / 100) * 0.3
        incubation_period = max(incubation_period, 2.0)  # Mínimo 2 días
        
        # Transmisibilidad base (prob. transmisión por contacto)
        # Base (Wuhan) = 0.10
        # Score como multiplicador log
        import math
        multiplier = 1 + math.log10(max(aggression_score, 100) / 100)
        transmissibility = 0.10 * multiplier
        transmissibility = min(transmissibility, 0.35)  # Cap en 35%
        
        return {
            "aggression_score": float(aggression_score),
            "rbm_score": float(rbm_score),
            "furina_score": float(furina_score),
            "r0_estimated": round(r0_estimated, 2),
            "incubation_period_days": round(incubation_period, 1),
            "transmissibility_base": round(transmissibility, 3),
            "infectious_period_days": 10.0  # Relativamente constante
        }
    
    except Exception as e:
        # Si falla, retornar parámetros default de Wuhan
        return {
            "aggression_score": 0.0,
            "rbm_score": 0.0,
            "furina_score": 0.0,
            "r0_estimated": 2.5,
            "incubation_period_days": 5.5,
            "transmissibility_base": 0.10,
            "infectious_period_days": 10.0,
            "error": str(e)
        }


# ============================================================================
# BACKGROUND TASK: Pipeline Completo
# ============================================================================

def run_analysis_pipeline(
    job_id: str,
    variant_fasta_path: Path,
    reference_fasta_path: Path,
    use_cpu: bool,
    impute_gaps: bool
):
    """
    Ejecuta el pipeline completo de análisis en background.
    Actualiza el estado del job a medida que progresa.
    """
    
    # Estado inicial
    status = {
        "job_id": job_id,
        "status": "processing",
        "progress": 0.0,
        "current_step": "Iniciando",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None
    }
    save_job_status(job_id, status)
    
    try:
        variant_name = variant_fasta_path.stem
        reference_name = reference_fasta_path.stem
        
        cpu_flag = ["--cpu"] if use_cpu else []
        
        # ====================================================================
        # PASO 1: Extracción de Spike
        # ====================================================================
        status["current_step"] = "Extrayendo secuencia Spike"
        status["progress"] = 0.17
        save_job_status(job_id, status)
        
        result = run_pipeline_step(
            ["python3", "modules/extraer_spike.py", str(reference_fasta_path)],
            "Extracción Spike (referencia)"
        )
        if not result["success"]:
            raise Exception(f"Fallo en extracción de referencia: {result['stderr']}")
        
        result = run_pipeline_step(
            ["python3", "modules/extraer_spike.py", str(variant_fasta_path)],
            "Extracción Spike (variante)"
        )
        if not result["success"]:
            raise Exception(f"Fallo en extracción de variante: {result['stderr']}")
        
        # ====================================================================
        # PASO 2: Alineamiento
        # ====================================================================
        status["current_step"] = "Alineando secuencias"
        status["progress"] = 0.33
        save_job_status(job_id, status)
        
        spike_ref = OUTPUT_DIR / "s" / "spike" / f"spike_{reference_name}.txt"
        spike_var = OUTPUT_DIR / "s" / "spike" / f"spike_{variant_name}.txt"
        
        result = run_pipeline_step(
            ["python3", "modules/alineador_secuencias.py", str(spike_ref), str(spike_var)],
            "Alineamiento"
        )
        if not result["success"]:
            raise Exception(f"Fallo en alineamiento: {result['stderr']}")
        
        # ====================================================================
        # PASO 3: Imputación (opcional)
        # ====================================================================
        aligned_ref = OUTPUT_DIR / "s" / "spike_aligned" / f"spike_{reference_name}.txt"
        aligned_var = OUTPUT_DIR / "s" / "spike_aligned" / f"spike_{variant_name}.txt"
        
        if impute_gaps:
            status["current_step"] = "Imputando secuencias dañadas"
            status["progress"] = 0.50
            save_job_status(job_id, status)
            
            result = run_pipeline_step(
                ["python3", "modules/imputar_secuencia.py", str(aligned_var), str(aligned_ref)],
                "Imputación"
            )
            if not result["success"]:
                # No es crítico si falla - continuar con secuencia original
                print(f"Warning: Imputación falló, continuando sin imputar")
            else:
                # Si imputación exitosa, usar secuencia imputada
                imputada = OUTPUT_DIR / "s" / "spike_aligned" / f"spike_{variant_name}_imputada.txt"
        
        # ====================================================================
        # PASO 4: Oráculo (predicciones Prophet)
        # ====================================================================
        status["current_step"] = "Prediciendo mutaciones críticas (Prophet)"
        status["progress"] = 0.67
        save_job_status(job_id, status)
        imputed_json = OUTPUT_DIR / "prophet" / f"imputacion_spike_{variant_name}.json"
        
        result = run_pipeline_step(
            ["python3", "modules/oraculo_mutaciones.py", str(aligned_var), str(imputed_json)] + cpu_flag,
            "Oráculo Prophet"
        )
        if not result["success"]:
            raise Exception(f"Fallo en la prediccion de mutaciones: {result['stderr']}")
        
        # ====================================================================
        # PASO 5: Comparación con ESM-2
        # ====================================================================
        status["current_step"] = "Analizando con IA (ESM-2)"
        status["progress"] = 0.83
        save_job_status(job_id, status)
        
        result = run_pipeline_step(
            ["python3", "modules/comparador_inteligente.py", str(aligned_ref), str(aligned_var)] + cpu_flag,
            "Comparador ESM-2"
        )
        if not result["success"]:
            raise Exception(f"Fallo en comparación ESM-2: {result['stderr']}")
        
        # ====================================================================
        # PASO 6: Análisis Final
        # ====================================================================
        status["current_step"] = "Generando reporte ejecutivo"
        status["progress"] = 1.0
        save_job_status(job_id, status)
        
        report_csv = OUTPUT_DIR / "s" / "report" / f"reporte_spike_{variant_name}.csv"
        
        result = run_pipeline_step(
            ["python3", "modules/analizador_final.py", str(report_csv)],
            "Análisis Final"
        )
        if not result["success"]:
            raise Exception(f"Fallo en análisis final: {result['stderr']}")
        
        # ====================================================================
        # PROCESAMIENTO DE RESULTADOS
        # ====================================================================
        # Calcular parámetros epidemiológicos
        epi_params = calculate_epi_parameters(report_csv)
        
        # Cargar resultados para respuesta
        import pandas as pd
        df_results = pd.read_csv(report_csv)
        
        # Leer informe ejecutivo
        informe_path = OUTPUT_DIR / "s" / "report" / f"informe_ejecutivo_spike_{variant_name}.txt"
        lineage = "Desconocido"
        lineage_conf = ""
        
        if informe_path.exists():
            with open(informe_path, "r") as f:
                for line in f:
                    if "LINAJE PROBABLE:" in line:
                        parts = line.split("LINAJE PROBABLE:")
                        if len(parts) > 1:
                            lineage = parts[1].strip()
                    if "PROBABILIDAD DE LINAJE:" in line:
                        parts = line.split("PROBABILIDAD DE LINAJE:")
                        if len(parts) > 1:
                            lineage_conf = parts[1].strip()


        # Dentro de la lógica que construye el JSON final
        base_url = "http://localhost:8000" # Esto puede venir de un .env
        heatmap_rel_path = f"/output/s/report/heatmap_spike_{variant_name}.svg"
        reportcsv_rel_path = f"/output/s/report/reporte_spike_{variant_name}.csv"
        informe_txt_rel_path = f"/output/s/report/informe_ejecutivo_spike_{variant_name}.txt"
        informe_pdf_rel_path = f"/output/s/report/informe_ejecutivo_spike_{variant_name}.pdf"
        
        # Construir respuesta final
        results = {
            "job_id": job_id,
            "variant_name": variant_name,
            "lineage": lineage,
            "lineage_confidence": lineage_conf,
            "mutations": df_results.to_dict(orient="records"),
            "epi_params": epi_params,
            "processed_at": datetime.now().isoformat(),
            "files": {
                "csv": f"{base_url}{reportcsv_rel_path}",
                "heatmap": f"{base_url}{heatmap_rel_path}",
                "informe_txt": f"{base_url}{informe_txt_rel_path}",
                "informe_pdf": f"{base_url}{informe_pdf_rel_path}"
            }
        }
        
        # Actualizar estado final
        status["status"] = "completed"
        status["completed_at"] = datetime.now().isoformat()
        status["results"] = results
        save_job_status(job_id, status)
    
    except Exception as e:
        # Error en el pipeline
        status["status"] = "failed"
        status["error"] = str(e)
        status["completed_at"] = datetime.now().isoformat()
        save_job_status(job_id, status)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "api": "Telos-S",
        "version": "0.3.1-mvp",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "analysis": "/api/v1/analysis",
            "simulation": "/api/v1/simulation"
        }
    }

# Montar la carpeta de salida para acceso público vía URL
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

@app.get("/health")
async def health_check():
    """Health check para monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/analysis/upload", status_code=202)
async def upload_and_analyze(
    background_tasks: BackgroundTasks,
    variant_file: UploadFile = File(..., description="Archivo FASTA del genoma de la variante"),
    reference_file: Optional[UploadFile] = File(None, description="Archivo FASTA de referencia (opcional, default: Wuhan)"),
    use_cpu: bool = Query(True, description="Forzar CPU en lugar de GPU"),
    impute_gaps: bool = Query(True, description="Imputar bloques grandes de X")
):
    """
    Sube un archivo FASTA y ejecuta el análisis completo.
    
    Retorna inmediatamente un job_id para consultar el estado.
    El análisis se ejecuta en background.
    """
    
    # Crear job ID
    job_id = create_job_id()
    
    # Guardar archivos subidos
    variant_path = UPLOAD_DIR / f"{job_id}_variant.fasta"
    
    with open(variant_path, "wb") as f:
        shutil.copyfileobj(variant_file.file, f)
    
    # Referencia (usar Wuhan si no se provee)
    if reference_file:
        reference_path = UPLOAD_DIR / f"{job_id}_reference.fasta"
        with open(reference_path, "wb") as f:
            shutil.copyfileobj(reference_file.file, f)
    else:
        reference_path = BASE_DIR / "wuhan_ref.fasta"
        if not reference_path.exists():
            raise HTTPException(
                status_code=400,
                detail="No se proveyó referencia y 'wuhan_ref.fasta' no existe en el servidor"
            )
    
    # Ejecutar pipeline en background
    background_tasks.add_task(
        run_analysis_pipeline,
        job_id,
        variant_path,
        reference_path,
        use_cpu,
        impute_gaps
    )
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Análisis iniciado. Usa GET /api/v1/analysis/{job_id} para consultar el estado.",
        "estimated_time_minutes": 5
    }


@app.get("/api/v1/analysis/{job_id}")
async def get_analysis_status(job_id: str):
    """
    Consulta el estado de un análisis.
    
    Retorna el progreso actual y resultados cuando está completado.
    """
    status = load_job_status(job_id)
    return status


@app.get("/api/v1/analysis/{job_id}/results")
async def get_analysis_results(job_id: str):
    """
    Obtiene los resultados completos de un análisis completado.
    """
    status = load_job_status(job_id)
    
    if status["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"El análisis está en estado '{status['status']}', no 'completed'"
        )
    
    if "results" not in status:
        raise HTTPException(
            status_code=500,
            detail="Análisis completado pero sin resultados disponibles"
        )
    
    return status["results"]


@app.post("/api/v1/simulation/run")
async def run_simulation(request: SimulationRequest):
    """
    Ejecuta simulación epidemiológica basada en resultados de análisis.
    
    NOTA: Este es un placeholder para MVP.
    La simulación completa se implementará en la siguiente fase.
    """
    
    # Cargar resultados del análisis
    status = load_job_status(request.job_id)
    
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="El análisis debe estar completado primero")
    
    results = status.get("results", {})
    epi_params = results.get("epi_params", {})
    
    # Por ahora, retornar parámetros que el frontend puede usar
    # En fase 2, esto ejecutará Telos-SIM real
    
    return {
        "simulation_id": f"sim_{uuid.uuid4().hex[:12]}",
        "status": "completed",
        "variant": {
            "name": results.get("variant_name", "Unknown"),
            "aggression_score": results.get("aggression_score", 0),
            "lineage": results.get("lineage", "Unknown")
        },
        "scenario": request.scenario,
        "location": request.location_code,
        "duration_days": request.duration_days,
        "epi_params": epi_params,
        "message": "Simulación completa se implementará en Fase 2. Parámetros disponibles para visualización."
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload durante desarrollo
    )