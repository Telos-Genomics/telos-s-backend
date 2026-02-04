<script>
    import { navigate } from '../router';
    import { Button } from 'flowbite-svelte';
    const API_URL = 'http://localhost:8000';

    let isDisabled = false;
    let progressVisible = false;
    let progressText = '';
    let progressPercent = 0;

    async function uploadAndAnalyze() {
        const fileInput = document.getElementById('fasta-file');
        const file = fileInput.files[0];

        if (!file) {
            alert('Por favor selecciona un archivo FASTA');
            return;
        }

        const formData = new FormData();
        formData.append('variant_file', file);

        // Upload
        try {

            const response = await fetch(`${API_URL}/api/v1/analysis/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Response status: ${response.status}`);
            }
            
            const { job_id } = await response.json();
            isDisabled = true;
            progressVisible = true;
                
            // Poll status
            pollStatus(job_id);

        } catch(err){
            console.log(err.message)
            isDisabled = false;
        }
        
    }

    async function pollStatus(jobId) {
        try {
            const response = await fetch(`${API_URL}/api/v1/analysis/${jobId}`);
            const status = await response.json();

            // Update progress (reactive)
            progressText = status.current_step;
            progressPercent = status.progress * 100;

            if (status.status === 'completed') {
                // Navigate to results - SIN los dos puntos
                navigate(`/results/${jobId}`);
            } else if (status.status === 'failed') {
                alert('Analysis failed: ' + status.error);
                isDisabled = false;
                progressVisible = false;
            } else {
                // Continue polling
                setTimeout(() => pollStatus(jobId), 2000);
            }
        } catch(err) {
            console.error('Error al consultar status:', err);
            alert('Error al consultar status: ' + err.message);
            isDisabled = false;
            progressVisible = false;
        }
    }
</script>

<div class="container mx-auto px-4 py-8">
    <div class="max-w-2xl mx-auto my-10">
        <h1 class="text-3xl font-bold mb-8">Telos-S Analysis</h1>
        
        <div id="upload-section">
            <div class="mb-4">
                <label for="fasta-file" class="block text-sm font-medium mb-2">
                    Select FASTA file
                </label>
                <input 
                    type="file" 
                    id="fasta-file" 
                    accept=".fasta,.fa" 
                    disabled={isDisabled}
                    class="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none"
                />
            </div>
            
            <Button 
                onclick={() => uploadAndAnalyze()}
                disabled={isDisabled}
                class="w-full"
            >
                {isDisabled ? 'Analyzing...' : 'Analyze'}
            </Button>
        </div>

        {#if progressVisible}
            <div id="progress-section" class="mt-8">
                <div class="mb-2 text-sm text-gray-600">
                    {progressText}
                </div>
                <div class="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                        class="bg-blue-600 h-2.5 rounded-full transition-all duration-300" 
                        style="width: {progressPercent}%"
                    ></div>
                </div>
                <div class="mt-1 text-xs text-gray-500 text-right">
                    {progressPercent.toFixed(0)}%
                </div>
            </div>
        {/if}
    </div>
</div>