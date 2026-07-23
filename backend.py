"""
Telos-S API
-----------
REST API for SARS-CoV-2 variant analysis with epidemiological impact prediction.

Architecture:
    - FastAPI backend
    - Asynchronous processing of long-running analyses
    - Results stored as JSON
    - Ready to connect to a simulation frontend
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
# CONFIGURATION
# ============================================================================

app = FastAPI(
    title="Telos-S API",
    description="Genomic analysis of SARS-CoV-2 variants with epidemiological prediction",
    version="0.1.0-mvp",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
JOBS_DIR = OUTPUT_DIR / "jobs"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, JOBS_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================================================
# DATA MODELS (Pydantic)
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request to start a variant analysis"""
    variant_name: Optional[str] = Field(None, description="Variant name (optional)")
    use_cpu: bool = Field(True, description="Force CPU usage instead of GPU")
    impute_gaps: bool = Field(True, description="Impute large blocks of X from the reference")

    class Config:
        schema_extra = {
            "example": {
                "variant_name": "Omicron_BA.2.86",
                "use_cpu": True,
                "impute_gaps": True
            }
        }


class AnalysisStatus(BaseModel):
    """Status of an analysis in progress"""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 - 1.0
    current_step: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class VariantMutation(BaseModel):
    """Individual mutation in the Spike protein"""
    mutation: str
    position: int
    zone: str
    llr: float
    score: float
    confidence: str  # CONFIABLE, SOSPECHOSA, IMPUTADA, INVALIDA


class AnalysisResults(BaseModel):
    """Complete analysis results"""
    job_id: str
    variant_name: str
    aggression_score: float
    lineage: str
    lineage_confidence: float
    sequence_quality: float
    mutations: List[VariantMutation]
    prophet_predictions: Optional[List[Dict[str, Any]]] = None

    # Derived epidemiological parameters (for simulation)
    epi_params: Dict[str, float] = Field(
        default_factory=dict,
        description="Computed epidemiological parameters (R0, transmissibility, etc.)"
    )

    # Metadata
    processed_at: datetime
    files: Dict[str, str]  # Paths to generated files


class SimulationRequest(BaseModel):
    """Request to run an epidemiological simulation"""
    job_id: str = Field(..., description="ID of the variant analysis")
    scenario: str = Field(..., description="Scenario type: 'airport' or 'urban'")
    location_code: str = Field(..., description="Location code (e.g. 'PTY' for airport)")
    duration_days: int = Field(30, ge=1, le=365, description="Simulation duration in days")
    initial_cases: int = Field(1, ge=1, description="Number of initial cases")

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
# UTILITIES
# ============================================================================

def create_job_id() -> str:
    """Generates a unique ID for the job"""
    return f"job_{uuid.uuid4().hex[:12]}"


def save_job_status(job_id: str, status_data: dict):
    """Saves the job status to JSON"""
    job_file = JOBS_DIR / f"{job_id}.json"
    with open(job_file, "w") as f:
        json.dump(status_data, f, indent=2, default=str)


def load_job_status(job_id: str) -> dict:
    """Loads the job status from JSON"""
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    with open(job_file, "r") as f:
        return json.load(f)


def run_pipeline_step(command: List[str], step_name: str) -> dict:
    """
    Runs a pipeline step and captures its output.

    Returns:
        dict with "success", "stdout", "stderr"
    """
    try:
        result = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=1200  # 10 minute max per step
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
            "stderr": f"Timeout in step {step_name}",
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
    Computes epidemiological parameters from the mutations CSV.

    This is the "bridge" between Telos-S and Telos-SIM.
    """
    import pandas as pd

    try:
        df = pd.read_csv(results_csv)

        # Keep only reliable mutations
        df_reliable = df[df['Reliability'] == 'RELIABLE'].copy()

        # Total Aggression Score
        aggression_score = df_reliable['Score'].abs().sum()

        # RBM score (ACE2 affinity)
        rbm_mutations = df_reliable[
            (df_reliable['Pos'] >= 437) &
            (df_reliable['Pos'] <= 508)
        ]
        rbm_score = rbm_mutations['Score'].abs().sum()

        # Furin score (entry efficiency)
        furin_mutations = df_reliable[
            (df_reliable['Pos'] >= 681) &
            (df_reliable['Pos'] <= 685)
        ]
        furin_score = furin_mutations['Score'].abs().sum()

        # Estimate R0
        # Base (Wuhan) = 2.5
        # Every 100 RBM points -> +0.2 R0
        # Every 50 Furin points -> +0.15 R0
        r0_estimated = 2.5 + (rbm_score / 100) * 0.2 + (furin_score / 50) * 0.15
        r0_estimated = min(r0_estimated, 8.0)  # Cap at 8.0 (realistic)

        # Estimate incubation period
        # Base (Wuhan) = 5.5 days
        # Every 100 Furin points -> -0.3 days
        incubation_period = 5.5 - (furin_score / 100) * 0.3
        incubation_period = max(incubation_period, 2.0)  # Minimum 2 days

        # Base transmissibility (probability of transmission per contact)
        # Base (Wuhan) = 0.10
        # Score used as a log multiplier
        import math
        multiplier = 1 + math.log10(max(aggression_score, 100) / 100)
        transmissibility = 0.10 * multiplier
        transmissibility = min(transmissibility, 0.35)  # Cap at 35%

        return {
            "aggression_score": float(aggression_score),
            "rbm_score": float(rbm_score),
            "furina_score": float(furin_score),
            "r0_estimated": round(r0_estimated, 2),
            "incubation_period_days": round(incubation_period, 1),
            "transmissibility_base": round(transmissibility, 3),
            "infectious_period_days": 10.0  # Relatively constant
        }

    except Exception as e:
        # If it fails, return Wuhan default parameters
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
# BACKGROUND TASK: Full Pipeline
# ============================================================================

def run_analysis_pipeline(
    job_id: str,
    variant_fasta_path: Path,
    reference_fasta_path: Path,
    use_cpu: bool,
    impute_gaps: bool
):
    """
    Runs the full analysis pipeline in the background.
    Updates the job status as it progresses.
    """

    # Initial status
    status = {
        "job_id": job_id,
        "status": "processing",
        "progress": 0.0,
        "current_step": "Starting",
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
        # STEP 1: Spike extraction
        # ====================================================================
        status["current_step"] = "Extracting Spike sequence"
        status["progress"] = 0.17
        save_job_status(job_id, status)

        result = run_pipeline_step(
            ["python3", "modules/spike_extractor.py", str(reference_fasta_path)],
            "Spike extraction (reference)"
        )
        if not result["success"]:
            raise Exception(f"Reference extraction failed: {result['stderr']}")

        result = run_pipeline_step(
            ["python3", "modules/spike_extractor.py", str(variant_fasta_path)],
            "Spike extraction (variant)"
        )
        if not result["success"]:
            raise Exception(f"Variant extraction failed: {result['stderr']}")

        # ====================================================================
        # STEP 2: Alignment
        # ====================================================================
        status["current_step"] = "Aligning sequences"
        status["progress"] = 0.33
        save_job_status(job_id, status)

        spike_ref = OUTPUT_DIR / "s" / "spike" / f"spike_{reference_name}.txt"
        spike_var = OUTPUT_DIR / "s" / "spike" / f"spike_{variant_name}.txt"

        result = run_pipeline_step(
            ["python3", "modules/sequence_aligner.py", str(spike_ref), str(spike_var)],
            "Alignment"
        )
        if not result["success"]:
            raise Exception(f"Alignment failed: {result['stderr']}")

        # ====================================================================
        # STEP 3: Imputation (optional)
        # ====================================================================
        aligned_ref = OUTPUT_DIR / "s" / "spike_aligned" / f"spike_{reference_name}.txt"
        aligned_var = OUTPUT_DIR / "s" / "spike_aligned" / f"spike_{variant_name}.txt"

        if impute_gaps:
            status["current_step"] = "Imputing damaged sequences"
            status["progress"] = 0.50
            save_job_status(job_id, status)

            result = run_pipeline_step(
                ["python3", "modules/impute_sequence.py", str(aligned_var), str(aligned_ref)],
                "Imputation"
            )
            if not result["success"]:
                # Not critical if it fails - continue with the original sequence
                print(f"Warning: Imputation failed, continuing without imputation")
            else:
                # If imputation succeeded, use the imputed sequence
                imputed_var = OUTPUT_DIR / "s" / "spike_aligned" / f"spike_{variant_name}_imputada.txt"

        # ====================================================================
        # STEP 4: Oracle (Prophet predictions)
        # ====================================================================
        status["current_step"] = "Predicting critical mutations (Prophet)"
        status["progress"] = 0.67
        save_job_status(job_id, status)
        imputed_json = OUTPUT_DIR / "prophet" / f"imputacion_spike_{variant_name}.json"

        result = run_pipeline_step(
            ["python3", "modules/mutations_oracle.py", str(aligned_var), str(imputed_json)] + cpu_flag,
            "Prophet Oracle"
        )
        if not result["success"]:
            raise Exception(f"Mutation prediction failed: {result['stderr']}")

        # ====================================================================
        # STEP 5: ESM-2 comparison
        # ====================================================================
        status["current_step"] = "Analyzing with AI (ESM-2)"
        status["progress"] = 0.83
        save_job_status(job_id, status)

        result = run_pipeline_step(
            ["python3", "modules/variant_comparator.py", str(aligned_ref), str(aligned_var)] + cpu_flag,
            "ESM-2 Comparator"
        )
        if not result["success"]:
            raise Exception(f"ESM-2 comparison failed: {result['stderr']}")

        # ====================================================================
        # STEP 6: Final analysis
        # ====================================================================
        status["current_step"] = "Generating executive report"
        status["progress"] = 1.0
        save_job_status(job_id, status)

        report_csv = OUTPUT_DIR / "s" / "reports" / f"report_spike_{variant_name}.csv"

        result = run_pipeline_step(
            ["python3", "modules/final_analyzer.py", str(report_csv)],
            "Final Analysis"
        )
        if not result["success"]:
            raise Exception(f"Final analysis failed: {result['stderr']}")

        # ====================================================================
        # RESULTS PROCESSING
        # ====================================================================
        # Compute epidemiological parameters
        epi_params = calculate_epi_parameters(report_csv)

        # Load results for the response
        import pandas as pd
        df_results = pd.read_csv(report_csv)

        # Read the executive report
        report_path = OUTPUT_DIR / "s" / "reports" / f"executive_report_spike_{variant_name}.txt"
        lineage = "Unknown"
        lineage_conf = ""

        if report_path.exists():
            with open(report_path, "r") as f:
                for line in f:
                    if "PROBABLE LINEAGE:" in line:
                        parts = line.split("PROBABLE LINEAGE:")
                        if len(parts) > 1:
                            lineage = parts[1].strip()
                    if "LINEAGE PROBABILITY:" in line:
                        parts = line.split("LINEAGE PROBABILITY:")
                        if len(parts) > 1:
                            lineage_conf = parts[1].strip()


        # Inside the logic that builds the final JSON
        base_url = os.getenv("PUBLIC_URL", f"http://localhost:{os.environ['API_PORT']}")  # This can come from a .env
        heatmap_rel_path = f"/output/s/reports/heatmap_spike_{variant_name}.svg"
        report_csv_rel_path = f"/output/s/reports/report_spike_{variant_name}.csv"
        report_txt_rel_path = f"/output/s/reports/executive_report_spike_{variant_name}.txt"
        report_pdf_rel_path = f"/output/s/reports/executive_report_spike_{variant_name}.pdf"

        # Build the final response
        results = {
            "job_id": job_id,
            "variant_name": variant_name.split("_", 3)[-1],
            "lineage": lineage,
            "lineage_confidence": lineage_conf,
            # Promoted to root level for direct access -- also exists in epi_params
            "aggression_score": epi_params.get("aggression_score", 0.0),
            "sequence_quality": round(
                (1 - df_results[df_results["Reliability"] == "INVALID"].shape[0]
                 / max(df_results.shape[0], 1)) * 100, 1
            ),
            "mutations": df_results.to_dict(orient="records"),
            "epi_params": epi_params,
            "processed_at": datetime.now().isoformat(),
            "files": {
                "csv": f"{base_url}{report_csv_rel_path}",
                "heatmap": f"{base_url}{heatmap_rel_path}",
                "txt_report": f"{base_url}{report_txt_rel_path}",
                "pdf_report": f"{base_url}{report_pdf_rel_path}"
            }
        }

        # Update final status
        status["status"] = "completed"
        status["completed_at"] = datetime.now().isoformat()
        status["results"] = results
        save_job_status(job_id, status)

    except Exception as e:
        # Pipeline error
        status["status"] = "failed"
        status["error"] = str(e)
        status["completed_at"] = datetime.now().isoformat()
        save_job_status(job_id, status)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
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

# Mount the output folder for public access via URL
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/analysis/upload", status_code=202)
async def upload_and_analyze(
    background_tasks: BackgroundTasks,
    variant_file: UploadFile = File(..., description="FASTA file of the variant genome"),
    reference_file: Optional[UploadFile] = File(None, description="Reference FASTA file (optional, default: Wuhan)"),
    use_cpu: bool = Query(True, description="Force CPU instead of GPU"),
    impute_gaps: bool = Query(True, description="Impute large blocks of X")
):
    """
    Uploads a FASTA file and runs the full analysis.

    Returns immediately with a job_id to poll for status.
    The analysis runs in the background.
    """

    # Create job ID
    job_id = create_job_id()

    # Save the uploaded files
    original_name = Path(variant_file.filename).stem
    variant_path = UPLOAD_DIR / f"{job_id}_{original_name}.fasta"

    with open(variant_path, "wb") as f:
        shutil.copyfileobj(variant_file.file, f)

    # Reference (use Wuhan if none is provided)
    if reference_file:
        reference_path = UPLOAD_DIR / f"{job_id}_reference.fasta"
        with open(reference_path, "wb") as f:
            shutil.copyfileobj(reference_file.file, f)
    else:
        reference_path = BASE_DIR / "NC_0455122.fasta"
        if not reference_path.exists():
            raise HTTPException(
                status_code=400,
                detail="No reference was provided and 'NC_0455122.fasta' does not exist on the server"
            )

    # Run the pipeline in the background
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
        "message": "Analysis started. Use GET /api/v1/analysis/{job_id} to check its status.",
        "estimated_time_minutes": 5
    }


@app.get("/api/v1/analysis/{job_id}")
async def get_analysis_status(job_id: str):
    """
    Checks the status of an analysis.

    Returns the current progress and results once completed.
    """
    status = load_job_status(job_id)
    return status


@app.get("/api/v1/analysis/{job_id}/results")
async def get_analysis_results(job_id: str):
    """
    Retrieves the full results of a completed analysis.
    """
    status = load_job_status(job_id)

    if status["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"The analysis is in status '{status['status']}', not 'completed'"
        )

    if "results" not in status:
        raise HTTPException(
            status_code=500,
            detail="Analysis completed but no results are available"
        )

    return status["results"]


@app.post("/api/v1/simulation/run")
async def run_simulation(request: SimulationRequest):
    """
    Runs an epidemiological simulation based on analysis results.

    NOTE: This is an MVP placeholder.
    The full simulation will be implemented in the next phase.
    """

    # Load the analysis results
    status = load_job_status(request.job_id)

    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="The analysis must be completed first")

    results = status.get("results", {})
    epi_params = results.get("epi_params", {})

    # For now, return parameters the frontend can use
    # In phase 2, this will run the real Telos-SIM

    return {
        "simulation_id": f"sim_{uuid.uuid4().hex[:12]}",
        "status": "completed",
        "variant": {
            "name": results.get("variant_name", "Unknown"),
            "aggression_score": results.get("aggression_score") or results.get("epi_params", {}).get("aggression_score", 0),
            "lineage": results.get("lineage", "Unknown")
        },
        "scenario": request.scenario,
        "location": request.location_code,
        "duration_days": request.duration_days,
        "epi_params": epi_params,
        "message": "Full simulation will be implemented in Phase 2. Parameters available for visualization."
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend:app",
        host=os.environ['API_HOST'],
        port=int(os.environ['API_PORT']),
        reload=True  # Auto-reload during development
    )