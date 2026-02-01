#!/usr/bin/env python3
"""
Test de rendimiento REALISTA para modelos de IA
Simula el uso real de ESM-2 con secuencias proteicas
"""

import torch
import torch.nn as nn
import time

def obtener_dispositivo():
    """Detecta el mejor dispositivo disponible"""
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA"
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "MPS"
    else:
        return torch.device("cpu"), "CPU"

def test_realista_modelo_grande():
    """
    Simula el comportamiento de ESM-2:
    - Modelo grande que permanece en GPU
    - Inputs pequeños que se transfieren
    - Múltiples inferencias consecutivas
    """
    print("="*60)
    print("🧬 TEST REALISTA: Simulación de ESM-2")
    print("="*60)
    
    device, device_name = obtener_dispositivo()
    print(f"\n📍 Dispositivo seleccionado: {device_name}")
    
    # Simular un modelo grande (similar a ESM-2)
    class ModeloGrande(nn.Module):
        def __init__(self):
            super().__init__()
            # Simular capas pesadas como en un transformer
            self.layers = nn.ModuleList([
                nn.Linear(1280, 1280) for _ in range(33)  # 33 capas como ESM-2
            ])
            self.norm = nn.LayerNorm(1280)
            self.activation = nn.GELU()
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
                x = self.activation(x)
                x = self.norm(x)
            return x
    
    print("\n📥 Creando modelo (similar a ESM-2: 33 capas, 1280 dimensiones)...")
    modelo = ModeloGrande()
    
    # Contar parámetros
    num_params = sum(p.numel() for p in modelo.parameters())
    print(f"   Parámetros: {num_params:,} (~{num_params/1e6:.1f}M)")
    
    # Test en CPU
    print("\n" + "-"*60)
    print("💻 TEST EN CPU")
    print("-"*60)
    modelo_cpu = modelo.to('cpu')
    modelo_cpu.eval()
    
    # Warmup
    with torch.no_grad():
        x = torch.randn(8, 512, 1280)  # batch=8, seq_len=512
        _ = modelo_cpu(x)
    
    # Test real
    tiempos_cpu = []
    with torch.no_grad():
        for i in range(10):
            x = torch.randn(8, 512, 1280)
            start = time.time()
            _ = modelo_cpu(x)
            tiempos_cpu.append(time.time() - start)
    
    tiempo_cpu = sum(tiempos_cpu) / len(tiempos_cpu)
    print(f"⏱️  Tiempo promedio: {tiempo_cpu:.3f} segundos/batch")
    print(f"   Por secuencia: {tiempo_cpu/8*1000:.1f} ms")
    
    # Test en GPU (si está disponible)
    if device.type != 'cpu':
        print("\n" + "-"*60)
        print(f"🚀 TEST EN {device_name}")
        print("-"*60)
        
        try:
            modelo_gpu = modelo.to(device)
            modelo_gpu.eval()
            
            # Warmup (IMPORTANTE para GPU)
            print("   Warming up GPU... ", end="", flush=True)
            with torch.no_grad():
                x = torch.randn(8, 512, 1280, device=device)
                _ = modelo_gpu(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
            print("✓")
            
            # Test real
            tiempos_gpu = []
            with torch.no_grad():
                for i in range(10):
                    x = torch.randn(8, 512, 1280, device=device)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start = time.time()
                    _ = modelo_gpu(x)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    elif device.type == 'mps':
                        torch.mps.synchronize()
                    
                    tiempos_gpu.append(time.time() - start)
            
            tiempo_gpu = sum(tiempos_gpu) / len(tiempos_gpu)
            print(f"⏱️  Tiempo promedio: {tiempo_gpu:.3f} segundos/batch")
            print(f"   Por secuencia: {tiempo_gpu/8*1000:.1f} ms")
            
            # Comparación
            print("\n" + "="*60)
            print("📊 COMPARACIÓN DE RENDIMIENTO")
            print("="*60)
            speedup = tiempo_cpu / tiempo_gpu
            
            print(f"\n{'Dispositivo':<15} {'Tiempo/batch':<15} {'Tiempo/seq':<15}")
            print("-"*45)
            print(f"{'CPU':<15} {tiempo_cpu:.3f}s{'':<8} {tiempo_cpu/8*1000:.1f}ms")
            print(f"{device_name:<15} {tiempo_gpu:.3f}s{'':<8} {tiempo_gpu/8*1000:.1f}ms")
            print("-"*45)
            
            if speedup > 1:
                print(f"\n🎉 {device_name} es {speedup:.1f}x MÁS RÁPIDO que CPU")
                print(f"   Ahorro de tiempo: {(tiempo_cpu - tiempo_gpu):.3f}s por batch")
                print(f"   Para 100 mutaciones: {(tiempo_cpu - tiempo_gpu)*100/60:.1f} minutos ahorrados")
            else:
                print(f"\n⚠️  {device_name} es {1/speedup:.1f}x más lento que CPU en este test")
                print("\n   Posibles razones:")
                print("   - Overhead de transferencia de datos")
                print("   - El modelo aún no es lo suficientemente grande")
                print("   - Driver/software de GPU necesita optimización")
                print("\n   PERO: Para ESM-2 real (650M parámetros), GPU debería ser más rápida")
            
        except Exception as e:
            print(f"\n❌ Error usando {device_name}: {e}")
            print("   Esto podría ser el origen del 'trace trap'")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("💡 CONCLUSIÓN PARA TU CASO DE USO")
    print("="*60)
    print("""
Para el análisis de proteínas Spike con ESM-2:

1. ✅ USAR GPU si speedup > 1.5x
2. ⚠️  CONSIDERAR CPU si speedup < 1.5x pero > 0.8x
3. ❌ USAR CPU si speedup < 0.8x (GPU más lenta)

IMPORTANTE: El modelo ESM-2 real es MUCHO más grande que esta simulación,
por lo que el speedup real probablemente sea mayor.

RECOMENDACIÓN: Prueba ambos con tu script real y mide el tiempo.
""")

def test_operaciones_especificas():
    """Test de operaciones específicas que usa tu script"""
    print("\n" + "="*60)
    print("🔬 TEST DE OPERACIONES ESPECÍFICAS")
    print("="*60)
    
    device, device_name = obtener_dispositivo()
    
    if device.type == 'cpu':
        print("\n⚠️  Solo CPU disponible, saltando test de GPU")
        return
    
    print(f"\nProbando operaciones que usa tu script en {device_name}...")
    
    operaciones_exitosas = []
    operaciones_fallidas = []
    
    # Test 1: Softmax
    try:
        x = torch.randn(1, 100, 1000, device=device)
        y = torch.nn.functional.softmax(x, dim=-1)
        if device.type == 'mps':
            torch.mps.synchronize()
        operaciones_exitosas.append("✅ Softmax")
    except Exception as e:
        operaciones_fallidas.append(f"❌ Softmax: {e}")
    
    # Test 2: Log
    try:
        x = torch.randn(1000, device=device).abs()
        y = torch.log(x)
        if device.type == 'mps':
            torch.mps.synchronize()
        operaciones_exitosas.append("✅ Log")
    except Exception as e:
        operaciones_fallidas.append(f"❌ Log: {e}")
    
    # Test 3: División
    try:
        x = torch.randn(1000, device=device)
        y = torch.randn(1000, device=device)
        z = x / y
        if device.type == 'mps':
            torch.mps.synchronize()
        operaciones_exitosas.append("✅ División")
    except Exception as e:
        operaciones_fallidas.append(f"❌ División: {e}")
    
    # Test 4: TopK
    try:
        x = torch.randn(1, 1000, device=device)
        values, indices = torch.topk(x, 5, dim=-1)
        if device.type == 'mps':
            torch.mps.synchronize()
        operaciones_exitosas.append("✅ TopK")
    except Exception as e:
        operaciones_fallidas.append(f"❌ TopK: {e}")
    
    # Test 5: Indexing complejo
    try:
        x = torch.randn(1, 512, device=device)
        mask = x > 0
        indices = mask.nonzero(as_tuple=True)
        if device.type == 'mps':
            torch.mps.synchronize()
        operaciones_exitosas.append("✅ Indexing complejo")
    except Exception as e:
        operaciones_fallidas.append(f"❌ Indexing complejo: {e}")
    
    print("\nResultados:")
    for op in operaciones_exitosas:
        print(f"  {op}")
    
    if operaciones_fallidas:
        print("\n⚠️  Operaciones con problemas:")
        for op in operaciones_fallidas:
            print(f"  {op}")
        print("\n💡 Activa PYTORCH_ENABLE_MPS_FALLBACK=1")
    else:
        print(f"\n✅ Todas las operaciones funcionan correctamente en {device_name}")

if __name__ == "__main__":
    test_realista_modelo_grande()
    test_operaciones_especificas()