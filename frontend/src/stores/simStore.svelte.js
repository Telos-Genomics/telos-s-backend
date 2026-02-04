//import airports from '../data/airports.json' with { type: 'json' };

export function createSimStore(epiParams) {
    // --- ESTADO ---
    let day = $state(0);
    let isPlaying = $state(false);
    let logs = $state([]);
    let timer = null;

    // --- LÓGICA DE EVENTOS ---
    const addLog = (msg) => {
        logs = [{ time: `Día ${day}`, msg }, ...logs.slice(0, 4)];
    };

    // --- CÁLCULO DE EXPANSIÓN ---
    // El radio crece según: Radio_base * (R0 ^ (días_post_incubación / 4))
    const calculateRadius = (base) => {
        if (day < epiParams.incubation_period_days) return 0;
        const deltaT = day - epiParams.incubation_period_days;
        
        const r0 = epiParams.r0_estimated;

        // --- NUEVA FÓRMULA: LOGÍSTICA ESCALADA ---
        // Simula: Crecimiento rápido inicial -> Desaceleración -> Saturación
    
        const MAX_PIXELS = 150; // Tamaño máximo visual (evita comerse el planeta)
        const GROWTH_RATE = 0.4; // Ajuste para que la animación dure los 30 segs

        // Fórmula Sigmoidea: K / (1 + e^(-k*(t-t0)))
        // Adaptada para visualización:
        const growth = 1 / (1 + Math.exp(-GROWTH_RATE * (deltaT - 10)));
    
        // Multiplicamos por el R0 para que variantes más agresivas (Delta) sean más grandes
        // pero siempre respetando el límite visual.
        const r0_factor = Math.min(r0 / 2, 1.5);


        // Ajustamos la escala para que sea visible en el mapa (metros a pixeles o escala Mapbox)
        return base + (MAX_PIXELS * growth * r0_factor);
    };

    return {
        get day() { return day; },
        get isPlaying() { return isPlaying; },
        get logs() { return logs; },

        start() {
            if (timer) return;
            isPlaying = true;
            addLog("Iniciando vigilancia de fronteras...");
            
            timer = setInterval(() => {
                if (day >= 30) return this.pause();
                day++; // Al ser $state, esto disparará los $effect en el componente

                // Disparar eventos narrativos basados en tu JSON de aeropuertos
                if (day === 1) addLog("Aterrizaje detectado: Vuelo DFW (Dallas) con carga viral.");
                if (day === Math.floor(epiParams.incubation_period_days)) {
                    addLog("⚠️ Fin de incubación. Brote detectado en Terminal Tocumen.");
                }
                if (day === 10) addLog("Transmisión comunitaria activa en Ciudad de Panamá.");
            }, 1000);
        },

        pause() {
            clearInterval(timer);
            timer = null;
            isPlaying = false;
        },

        reset() {
            this.pause();
            day = 0;
            logs = [];
        },

        getMapFeatures() {
            // Nodos clave en Panamá para la simulación
            const pois = [
                { id: 'tocumen', name: 'Tocumen PTY', coords: [-79.383, 9.071], base: 40 },
                { id: 'albrook', name: 'Albrook / Terminal', coords: [-79.551, 8.973], base: 30 },
                { id: 'multiplaza', name: 'Multiplaza / Paitilla', coords: [-79.513, 8.985], base: 25 }
            ];

            return pois.map(p => ({
                type: 'Feature',
                geometry: { type: 'Point', coordinates: p.coords },
                properties: { radius: calculateRadius(p.base) }
            }));
        }
    };
}