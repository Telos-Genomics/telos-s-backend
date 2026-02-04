import { createRouter } from 'sv-router';
import Home from './routes/Home.svelte'; // Mueve tu lógica de subida aquí
import Results from './routes/Results.svelte';

export const { p, navigate, isActive, route } = createRouter({
	'/': Home,                      // Ruta inicial
	'/results/:jobId': Results,  // Define jobId como parámetro de ruta
});