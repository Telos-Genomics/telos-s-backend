import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analizar_cepa(csv_path):
    df = pd.read_csv(csv_path)

    # Extraemos la posición numérica inmediatamente
    # Usamos r'(\d+)' para evitar el SyntaxWarning y sacamos el número de la mutación
    df['Pos'] = df['Mutacion'].str.extract(r'(\d+)').astype(float)

    # 1. Calidad de la Muestra
    total_mutaciones = len(df)
    indeterminadas = df['Mutacion'].str.contains('X').sum()
    calidad = ((total_mutaciones - indeterminadas) / total_mutaciones) * 100

    # Filtrar 'X' para calculos de score, pero mantenerlas para el reporte
    df_clean = df[~df['Mutacion'].str.contains('X')].copy()
    df_clean = df_clean.dropna(subset=['Pos'])

    # Dentro de analizar_cepa, antes de calcular el score
    # Filtramos solo las posiciones biológicas de la Spike (1-1273)
    df_biologico = df_clean[
        (df_clean['Pos'] >= 1) & 
        (df_clean['Pos'] <= 1273) & 
        (~df_clean['Mutacion'].str.startswith('-'))
    ]

    # 2. Calculo de Aggression Score
    # Aplicamos la formula: Sumatoria de (Score_Individual)
    # Usamos el Score que ya calculamos que integra Zona y LLR
    # Usamos .abs() para que las deleciones sumen riesgo en lugar de restarlo
    aggression_score = df_biologico['Score'].abs().sum()

    # 3. Identificacion de Linaje (Firmas)
    firmas = {
        "Omicron (BA.5/Gris)": ["E484A", "F486V", "N501Y", "D614G"],
        "Delta": ["L452R", "P681R", "D614G"],
        "Alpha": ["N501Y", "P681H", "D614G"]
    }

    mutaciones_presentes = df['Mutacion'].tolist()
    prediccion_linaje = ""
    max_coincidencia = 0

    for linaje, marcadores in firmas.items():
        coincidencias = len(set(marcadores) & set(mutaciones_presentes))
        porcentaje = (coincidencias / len(marcadores)) * 100
        if porcentaje > max_coincidencia:
            max_coincidencia = porcentaje
            prediccion_linaje = linaje

    # --- REPORTE EJECUTIVO ---
    print(f"--- RESUMEN EJECUTIVO DE VARIANTE ---")
    print(f"Calidad de Secuenciacion: {calidad:.2f}%")
    print(f"Puntaje de Agresividad: {aggression_score:.1f}")
    print(f"Linaje Probable: {prediccion_linaje} ({max_coincidencia:.1f}% de coincidencia)")
    print(f"-------------------------------------")

    # 3. Generacion del Heatmap
    generar_heatmap(df_clean, aggression_score, prediccion_linaje, csv_path)
    generar_informe_ejecutivo(df_biologico, aggression_score, prediccion_linaje, calidad, csv_path)

def generar_heatmap(df, score_total, linaje, csv_path):
    plt.figure(figsize=(15, 6))

    # Limpieza de posiciones: extraer solo los números (ej: E484K -> 484)
    # Usamos (\d+) para encontrar dígitos.
    df['Pos'] = df['Mutacion'].str.extract(r'(\d+)').astype(float)

    # Eliminamos filas donde no se pudo encontrar una posición numérica (por seguridad)
    df = df.dropna(subset=['Pos'])
    df['Pos'] = df['Pos'].astype(int)

    # Fondo de la proteina (1 a 1273 aminoacidos)
    plt.axhline(0, color='lightgrey', linewidth=20, alpha=0.3, zorder=1)

    # Sombras de Zonas Criticas
    plt.axvspan(319, 541, color='blue', alpha=0.1, label='Domino RBD')
    plt.axvspan(437, 508, color='cyan', alpha=0.2, label='Motivo RBM')
    plt.axvspan(681, 685, color='purple', alpha=0.2, label='Sitio Furina')

    # Mapeo de colores por estado
    color_map = {
        "🔴 AMENAZA": "red",
        "⚪ OBSERVACION": "orange",
        "⚠️ INTERES": "yellow" # Asumiendo lógica de score_riesgo > 30
    }

    # Dibujar puntos de mutacion
    for _, row in df.iterrows():
        color = "red" if "🔴" in row['Estado'] else ("orange" if row['Score'] > 30 else "yellow")

        # Altura basada en LLR (normalizada para visibilidad)
        altura = abs(row['LLR']) if row['LLR'] < 0 else row['LLR'] + 1

        plt.scatter(row['Pos'], altura, color=color, s=100, edgecolor='black', zorder=5)
        plt.text(row['Pos'], altura + 0.2, row['Mutacion'], fontsize=8, rotation=45)

        # Configuración de estética
    plt.title(f"Mapa de Calor: Variante {linaje} | Aggression Score: {score_total:.1f}")
    plt.xlabel("Posición en la Proteína Spike (Residuos)")
    plt.ylabel("Impacto Estructural (Abs LLR)")
    plt.xlim(0, 1273)
    plt.ylim(0, max(df['LLR'].abs()) + 2)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Leyenda personalizada
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Amenaza Crítica', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Variante Interés', markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Bajo Riesgo', markerfacecolor='yellow', markersize=10),
        Line2D([0], [0], color='blue', lw=4, alpha=0.3, label='Zona RBD'),
        Line2D([0], [0], color='cyan', lw=4, alpha=0.3, label='Zona RBM'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    folder_path = "output/report"

    try:
        plt.savefig(f"{folder_path}/heatmap_{csv_path.replace('.csv', '').replace('./output/report/','')}.png")
        print(f"🎨 Heatmap generado: '{folder_path}/heatmap_{csv_path.replace('.csv', '').replace('./output/report/','')}.png'")
    except OSError as e:
        # Handle other potential errors like permission issues
        print(f"Error creating folder '{folder_path}': {e}")
    
    

def generar_informe_ejecutivo(df_biologico, score, linaje, calidad, archivo_salida):
    # Identificar las 3 mutaciones más peligrosas
    top_amenazas = df_biologico.sort_values(by='Score', ascending=False).head(3)
    
    with open(f"informe_ejecutivo_{archivo_salida.replace('.csv', '').replace('reporte_','').replace('./output/report/','')}.txt", "w") as f:
        f.write("==============================================\n")
        f.write("      INFORME DE INTELIGENCIA GENÓMICA\n")
        f.write("==============================================\n\n")
        
        f.write(f"ID DE LA MUESTRA: {archivo_salida.replace('informe_', '').replace('.txt', '')}\n")
        f.write(f"VERDICTO: {'🔴 ALERTA MÁXIMA' if score > 1200 else '🟡 MONITOREO ACTIVO'}\n")
        f.write(f"PUNTAJE DE AGRESIVIDAD: {score:.1f}\n")
        f.write(f"LINAJE PROBABLE: {linaje}\n\n")
        
        f.write("--- ANÁLISIS DE RIESGO ---\n")
        f.write(f"La variante presenta un nivel de riesgo {'CRÍTICO' if score > 1200 else 'MODERADO'}.\n")
        f.write("Se observa una acumulación de mutaciones en el RBM (Receptor Binding Motif),\n")
        f.write("lo que sugiere una alta capacidad de escape inmunológico.\n\n")
        
        f.write("--- TOP 3 MUTACIONES CRÍTICAS ---\n")
        for _, row in top_amenazas.iterrows():
            f.write(f"• {row['Mutacion']}: Zona {row['Zona']} | Score: {row['Score']:.1f}\n")
        
        f.write("\n--- CALIDAD Y FILTRADO ---\n")
        f.write(f"Calidad de secuencia: {calidad:.2f}%\n")
        f.write("Nota: Se han filtrado artefactos de laboratorio (His-tags/Linkers)\n")
        f.write("para garantizar la precisión del score funcional.\n\n")
        
        f.write("==============================================\n")
        f.write("Generado por: BioAlerta MVP - Oráculo Genómico\n")
        f.write("==============================================\n")

    print(f"📄 Informe ejecutivo listo: informe_ejecutivo_{archivo_salida.replace('.csv', '').replace('reporte_','').replace('./output/report/','')}.txt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 analizador_final.py reporte.csv")
    else:
        analizar_cepa(sys.argv[1])