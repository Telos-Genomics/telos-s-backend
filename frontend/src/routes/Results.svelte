<script>
    import { onMount, onDestroy } from 'svelte';
    import mapboxgl from 'mapbox-gl';
    import 'mapbox-gl/dist/mapbox-gl.css';

    import { createSimStore } from '../stores/simStore.svelte';
    import { route } from '../router';

    const API_URL = 'http://localhost:8000';

    let map =  null;

    // --- ESTADO REACTIVO (Svelte 5 Runes) ---
    let loading = $state(true);
    let results = $state(null);
    let svgContent = $state("");
    let error = $state(null);

    let sim = $state(null);
    let mapContainer = $state(null);

    // Derivamos el jobId del router automáticamente
    let jobId = $derived(route.params.jobId);

    async function loadResults() {
        try {
            loading = true;
            const response = await fetch(`${API_URL}/api/v1/analysis/${jobId}/results`);
            if (!response.ok) throw new Error("Error cargando resultados");
            
            results = await response.json();

            // Cargar el contenido del SVG para inyectarlo
            const svgRes = await fetch(results.files.heatmap);
            svgContent = await svgRes.text();
            
            loading = false;
        } catch (err) {
            error = err.message;
            loading = false;
        }
    }

    // Estilo dinámico para el Score de Agresividad
    let scoreColor = $derived(results?.epi_params.aggression_score > 1000 ? 'text-red-600' : 'text-blue-600');

    mapboxgl.accessToken = 'pk.eyJ1Ijoia3VyYWkwMjEiLCJhIjoiY2ptbTRnMWsxMDBkeTN2cXhlaWM2aXJ0OSJ9.KI3udYVSC-oXgNueokCw0g';

    // Efecto que dispara la carga cuando el jobId cambia
    $effect(() => {
        if (jobId) loadResults();
    });

    // Inicializar store cuando lleguen los epi_params
    $effect(() => {
        if (results?.epi_params && !sim) {
            sim = createSimStore(results.epi_params);
        }
    });

    // Logica del mapa
    $effect(() => {
        if (mapContainer && !map && !loading) {
                map = new mapboxgl.Map({
                    container: mapContainer,
                    style: 'mapbox://styles/mapbox/dark-v11',
                    center: [-79.5, 9.0], // Centrado en Panamá
                    zoom: 10.5,
                    pitch: 45 // Perspectiva 3D para mayor impacto
                });

            map.on('load', () => {
                
                // Capa de los círculos de infección
                map.addSource('outbreaks', {
                    type: 'geojson',
                    data: { type: 'FeatureCollection', features: [] }
                });

                map.addLayer({
                    id: 'infection-layer',
                    type: 'circle',
                    source: 'outbreaks',
                    paint: {
                        'circle-radius': ['get', 'radius'],
                        'circle-color': '#ff4444',
                        'circle-stroke-width': 2,
                        'circle-stroke-color': '#ff0000',
                        'circle-opacity': [
                            'interpolate', ['linear'], ['get', 'radius'],
                            10, 0.8,  // Al inicio, muy rojo (concentrado)
                            150, 0.3  // Al final, más transparente (disperso)
                        ]
                    }
                });
            });
        }
    });

    $effect(() => {
        // Mencionamos sim.day para que el efecto se dispare cada vez que el reloj avance
        const currentTick = sim?.day;

        if (sim && map && map.getSource('outbreaks')) {
            const features = sim.getMapFeatures();
            
            console.log("Radios actuales:", features.map(f => f.properties.radius));
            
            map.getSource('outbreaks').setData({
                type: 'FeatureCollection',
                features: features // <--- Pasamos el array directamente
            });
        }
    });

    onDestroy(() => map?.remove());
</script>

<div class="w-full max-w-9/10 mx-auto px-4 py-8">
    {#if loading}
        <div class="flex flex-col items-center justify-center h-64">
            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-brand mb-4"></div>
            <p class="text-gray-500 animate-pulse">Decodificando firma biológica...</p>
        </div>
    {:else if results}
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 my-10">
            
            <div class="lg:col-span-1 space-y-6">
                <div class="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                    <h2 class="text-sm font-black uppercase tracking-widest text-gray-400 mb-4">Intelligence Report</h2>
                    <div class="mb-6">
                        <span class="text-xs font-bold px-2 py-1 bg-gray-900 text-white rounded uppercase">
                            {results.lineage}
                        </span>
                        <h1 class="text-5xl font-black mt-2 {scoreColor}">
                            {results.epi_params.aggression_score.toFixed(0)}
                        </h1>
                        <p class="text-gray-500 text-sm font-medium">Aggression Score (ESM-2 High Confidence)</p>
                    </div>

                    <div class="grid grid-cols-2 gap-4 border-t pt-6">
                        <div>
                            <p class="text-2xl font-bold text-gray-800">{results.epi_params.r0_estimated}</p>
                            <p class="text-xs text-gray-400 uppercase">Estimated R0</p>
                        </div>
                        <div>
                            <p class="text-2xl font-bold text-gray-800">{results.epi_params.incubation_period_days.toFixed(1)}d</p>
                            <p class="text-xs text-gray-400 uppercase">Incubation</p>
                        </div>
                    </div>
                </div>

                <div class="flex flex-col gap-2">
                    <button onclick={() => window.open(results.files.informe)} class="w-full py-3 bg-gray-800 text-white rounded-xl font-bold hover:bg-black transition-all">
                        📄 Open Executive Report
                    </button>
                    <button onclick={() => window.open(results.files.csv)} class="w-full py-3 border border-gray-200 rounded-xl font-bold hover:bg-gray-50 transition-all">
                        📊 Export Raw Data (CSV)
                    </button>
                </div>
            </div>

            <div class="lg:col-span-2 space-y-6">
                <div class="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                    <h3 class="font-bold text-gray-800 mb-4 flex items-center gap-2">
                        <span class="w-2 h-2 bg-red-500 rounded-full animate-ping"></span>
                        Genomic Mutation Heatmap (SVG)
                    </h3>
                    <div class="svg-container overflow-hidden">
                        {@html svgContent}
                    </div>
                </div>
            </div>

            <div class="lg:col-span-3 space-y-6">
                <div class="space-y-2">
                    <p class="text-slate-400 text-[10px] uppercase font-bold">Timeline Log</p>
                    {#each sim?.logs || [] as log}
                        <div class="text-xs p-2 bg-slate-900 border-l-2 border-blue-500 rounded-r animate-fade-in">
                            <span class="text-blue-400 font-mono">{log.time}:</span> 
                            <span class="text-slate-300">{log.msg}</span>
                        </div>
                    {/each}
                </div>

                <div class="bg-slate-900 p-4 rounded-2xl border border-slate-800">
                    <div class="flex justify-between items-center mb-4">
                        <span class="text-3xl font-mono font-black text-white">{sim?.day || 0}d</span>
                        <div class="flex gap-2">
                            <button onclick={() => sim.isPlaying ? sim.pause() : sim.start()} class="bg-blue-600 p-3 rounded-full hover:scale-105 transition-transform text-white">
                                {sim?.isPlaying ? '⏸' : '▶'}
                            </button>
                            <button onclick={() => sim.reset()} class="bg-slate-800 p-3 rounded-full text-white">🔄</button>
                        </div>
                    </div>
                    <div class="h-1 bg-slate-800 rounded-full overflow-hidden">
                        <div class="h-full bg-blue-500 transition-all" style="width: {(sim?.day / 30) * 100}%"></div>
                    </div>
                </div>

                <div bind:this={mapContainer} class="h-[500px] w-full"></div>
                <div class="absolute bottom-4 left-4 bg-slate-900/80 backdrop-blur px-3 py-1 rounded-full text-[10px] text-slate-400 border border-slate-700">
                    RADAR ACTIVO: MPTO/PTY SECTOR
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
    :global(.mapboxgl-ctrl-attrib) { display: none !important; }
    /* Esto permite que el SVG sea responsivo y estilizable */
    .svg-container :global(svg) {
        width: 100% !important;
        height: auto !important;
        filter: drop-shadow(0 4px 6px rgba(0,0,0,0.05));
    }
    
    /* Efecto hover sobre las barras del heatmap si tienen IDs */
    .svg-container :global(rect:hover) {
        fill: #ff0000 !important;
        cursor: pointer;
    }
</style>