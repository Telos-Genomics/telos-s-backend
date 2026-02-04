<script>
    import { Router } from 'sv-router';
    import { route } from './router';
    // Importamos los estilos globales aquí
    import 'flowbite/dist/flowbite.css';
</script>

<nav class="fixed w-full bg-gray-800 text-white p-4">
    <div class="w-full max-w-9/10 mx-auto font-bold tracking-tighter text-xl">
        TELOS-S <span class="text-blue-400">BIODEFENSE</span>
    </div>
</nav>

<main>
    <Router />
</main>
