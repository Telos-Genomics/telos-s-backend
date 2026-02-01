def generar_variante_simulada(archivo_original, mutaciones):
    with open(archivo_original, "r") as f:
        secuencia = list(f.read().strip())
    
    print(f"Generando variante sintetica basada en {archivo_original}")

    for pos, nuevo_amino in mutaciones.items():
        original = secuencia[pos-1]
        secuencia[pos-1] = nuevo_amino
        print(f"Mutacion aplicada: {original}{pos}{nuevo_amino}")

    nombre_archivo = "spike_latam_simulada.txt"
    with open(nombre_archivo, "w") as f:
        f.write("".join(secuencia))

    return nombre_archivo

if __name__ == "__main__":
    # Vamos a simular una variante con mutaciones clave:
    # 484K: Escape de anticuerpos
    # 501Y: Mayor transmisibilidad
    # 681H: Mejora la entrada a la celula
    mis_mutaciones = {484: 'K', 501: "Y", 681: "H"}
    generar_variante_simulada("spike_wuhan_ref.txt", mis_mutaciones)