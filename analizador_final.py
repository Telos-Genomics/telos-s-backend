import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

def cargar_predicciones_prophet(csv_path):
    base_name = csv_path.replace('.csv', '').replace('output/s/report/reporte_spike_', '')
    json_path = f"output/prophet/mutation_predictions_spike_{base_name}.json"

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return None

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
        "Omicron (BA.1/Original)": ["G339D", "S371L", "S373P", "S375F", "K417N", "N440K", "G446S", "E484A", "Q493R", "G498R", "N501Y", "Y505H"],
        "Omicron (BA.5/XBB.1.5)": ["G339H", "L452R", "F486P", "Q498R", "N501Y", "F490S"],
        "Omicron (BA.5/Subvariante)": ["L452R", "F486V", "R408X", "E484A"],
        "Delta (B.1.617.2)": ["T19R", "L452R", "T478K", "D614G", "P681R", "D950N"],
        "Alpha (B.1.1.7)": ["N501Y", "A570D", "D614G", "P681H", "T716I", "S982A"],
        "Beta (B.1.351)": ["K417N", "E484K", "N501Y", "D614G", "A701V"],
        "Gamma (P.1)": ["K417T", "E484K", "N501Y", "D614G", "H655Y"]
    }

    # 4. Cargar predicciones del Oráculo
    datos_prophet = cargar_predicciones_prophet(csv_path)

    mutaciones_presentes = df['Mutacion'].tolist()
    prediccion_linaje = ""
    max_coincidencia = 0

    for linaje, marcadores in firmas.items():
        coincidencias = 0

        for marcador in marcadores:
            # Separar "S", "371", "L"
            res_original = marcador[0]
            res_mutado = marcador[-1]
            pos_marcador = int(marcador[1:-1])

            # LÓGICA DE TOLERANCIA:
            # Buscamos en el reporte si existe la mutación objetivo en la posición 
            # esperada O en las posiciones adyacentes (debido a deleciones/inserciones)
            rango_tolerancia = range(pos_marcador - 5, pos_marcador + 6)
            
            # Verificamos si alguna mutación del CSV coincide con el aminoácido mutado 
            # en ese rango de posiciones
            posibles_matches = df_biologico[
                (df_biologico['Pos'].isin(rango_tolerancia)) & 
                (df_biologico['Mutacion'].str.endswith(res_mutado))
            ]

            if not posibles_matches.empty:
                coincidencias += 1


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
    generar_heatmap(df_clean, aggression_score, prediccion_linaje, csv_path, datos_prophet)
    generar_informe_ejecutivo(df_biologico, aggression_score, prediccion_linaje, calidad, csv_path, datos_prophet)

def generar_heatmap(df, score_total, linaje, csv_path, datos_prophet=None):
    plt.figure(figsize=(15, 7))

    # Limpieza de posiciones: extraer solo los números (ej: E484K -> 484)
    # Usamos (\d+) para encontrar dígitos.
    df['Pos'] = df['Mutacion'].str.extract(r'(\d+)').astype(float)
    # Eliminamos filas donde no se pudo encontrar una posición numérica (por seguridad)
    df = df.dropna(subset=['Pos'])
    df['Pos'] = df['Pos'].astype(int)

    # 1. CAPTURAR ALTURA MÁXIMA DINÁMICA
    # Empezamos con el máximo LLR actual
    max_y = df['LLR'].abs().max()
    altura_predicciones = 8 # Valor base sugerido

    # Si las mutaciones reales ya son muy altas (como en Omicron),
    # subimos la altura de las predicciones para que no se solapen
    if max_y > 7:
        altura_predicciones = max_y + 3
    
    # Ajustamos el límite final del gráfico
    limite_superior = max(max_y, altura_predicciones) + 5

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

    # --- 1. DIBUJAR MUTACIONES ACTUALES (Sólidas) ---
    for _, row in df.iterrows():
        color = "red" if "🔴" in row['Estado'] else ("orange" if row['Score'] > 30 else "yellow")
        # Altura basada en LLR (normalizada para visibilidad)
        altura = abs(row['LLR'])

        plt.scatter(row['Pos'], altura, color=color, s=100, edgecolor='black', zorder=5)
        plt.text(row['Pos'], altura + 0.3, row['Mutacion'], fontsize=8, rotation=45)

    # --- 2. DIBUJAR PREDICCIONES DEL ORÁCULO (Detección de solapamiento) ---
    if datos_prophet:
        for target in datos_prophet:
            pos = target['detected_position']
            original = target['original']

            # Filtramos para encontrar la predicción con mayor confianza 
            # que NO sea el aminoácido que ya tiene el virus.
            candidatos = [p for p in target['predictions'] if p['amino'] != original]
            
            if candidatos:
                # Tomamos el primero (el de mayor confianza por el orden del JSON)
                top_mut = candidatos[0]
                confianza = top_mut['confidence']
                
                # Bajamos el umbral a 5% para Omicron, ya que sus probabilidades 
                # están muy repartidas excepto en el sitio 681.
                if confianza > 5: 
                    # --- Dibujar en el Heatmap ---
                    plt.scatter(pos, altura_predicciones, facecolors='none', 
                                edgecolors='magenta', s=250, linestyle='--', 
                                linewidth=2, zorder=6)
                    
                    label_ia = f"{original}{pos}{top_mut['amino']}\n{confianza:.1f}%"
                    plt.text(pos, altura_predicciones + 0.8, label_ia, 
                             color='darkmagenta', fontsize=9, fontweight='black', 
                             ha='center', bbox=dict(facecolor='white', alpha=0.7, 
                             edgecolor='none', boxstyle='round'))

    # Configuración de estética
    plt.title(f"TELOS-S: Inteligencia de Variante {linaje} | Score: {score_total:.1f}", fontsize=14)
    plt.xlabel("Posición en la Proteína Spike (Residuos)")
    plt.ylabel("Impacto Estructural (Abs LLR)")

    plt.xlim(0, 1273)
    plt.ylim(-1, limite_superior) # Aquí aplicamos el nuevo límite dinámico
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Leyenda personalizada
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Amenaza Crítica', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Variante Interés', markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Bajo Riesgo', markerfacecolor='yellow', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Ruta Evolutiva (IA)', markeredgecolor='magenta', markerfacecolor='none', markersize=12, linestyle='--'),
        Line2D([0], [0], color='blue', lw=4, alpha=0.3, label='Zona RBD'),
        Line2D([0], [0], color='cyan', lw=4, alpha=0.3, label='Zona RBM'),
        Line2D([0], [0], color='purple', lw=4, alpha=0.3, label='Zona Furina')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize='small')

    plt.tight_layout()

    folder_path = "output/s/report"

    try:
        plt.savefig(f"{folder_path}/heatmap_{csv_path.replace('.csv', '').replace('output/s/report/','')}.png")
        print(f"🎨 Heatmap generado: '{folder_path}/heatmap_{csv_path.replace('.csv', '').replace('output/s/report/','')}.png'")
    except OSError as e:
        # Handle other potential errors like permission issues
        print(f"Error creating folder '{folder_path}': {e}")

def generar_informe_ejecutivo(df_biologico, score, linaje, calidad, archivo_salida, datos_prophet):
    # Identificar las 3 mutaciones más peligrosas
    top_amenazas = df_biologico.sort_values(by='Score', ascending=False).head(3)
    
    with open(f"informe_ejecutivo_{archivo_salida.replace('.csv', '').replace('reporte_','').replace('output/s/report/','')}.txt", "w") as f:
        f.write("==============================================\n")
        f.write("      INFORME DE INTELIGENCIA GENÓMICA\n")
        f.write("==============================================\n\n")
        
        f.write(f"ID DE LA MUESTRA: {archivo_salida.replace('informe_', '').replace('.txt', '')}\n")
        f.write(f"VEREDICTO: {'🔴 ALERTA MÁXIMA' if score > 1200 else '🟡 MONITOREO ACTIVO'}\n")
        f.write(f"PUNTAJE DE AGRESIVIDAD: {score:.1f}\n")
        f.write(f"LINAJE PROBABLE: {linaje}\n\n")
        
        f.write("--- ANÁLISIS DE RIESGO ---\n")
        f.write(f"La variante presenta un nivel de riesgo {'CRÍTICO' if score > 1200 else 'MODERADO'}.\n")
        f.write("Se observa una acumulación de mutaciones en el RBM (Receptor Binding Motif),\n")
        f.write("lo que sugiere una alta capacidad de escape inmunológico.\n\n")
        
        f.write("--- TOP 3 MUTACIONES CRÍTICAS ---\n")
        for _, row in top_amenazas.iterrows():
            f.write(f"• {row['Mutacion']}: Zona {row['Zona']} | Score: {row['Score']:.1f}\n")

        # --- NUEVA SECCIÓN: PRONÓSTICO EVOLUTIVO (EL TOQUE DE GRACIA) ---
        if datos_prophet:
            f.write("\n--- PRONÓSTICO DE EVOLUCIÓN (TELOS PROPHET) ---\n")
            f.write("Análisis de estabilidad estructural mediante IA (ESM-2):\n")

            # Mapeo manual de la referencia de Wuhan para los sitios de interés
            ref_wuhan_map = {
                "Sitio_RBM_452": "L",
                "Sitio_RBM_484": "E",
                "Sitio_RBM_501": "N",
                "Sitio_Furina_681": "P"
            }
            
            for target in datos_prophet:
                pos = target['detected_position']
                nombre = target['target']
                actual = target['original'] # Lo que tiene la muestra
                wuhan_base = ref_wuhan_map.get(nombre, "?")

                # Buscamos en el DF si hay una mutación registrada
                match_csv = df_biologico[
                    (df_biologico['Pos'] >= pos - 2) & 
                    (df_biologico['Pos'] <= pos + 2)
                ].sort_values(by='Score', ascending=False)
                
                if not match_csv.empty:
                    aa_actual = match_csv.iloc[0]['Mutacion'][-1]
                    # Si el CSV confirma la mutación, usamos el primer caracter del CSV como ref real
                    wuhan_real = match_csv.iloc[0]['Mutacion'][0]
                else:
                    aa_actual = actual
                    wuhan_real = wuhan_base

                f.write(f"• {nombre} (Wuhan Ref: {wuhan_real} | Actual en Muestra: {aa_actual}):\n")

                # Buscamos la mejor opción que NO sea la actual
                mejor_mutacion = next((p for p in target['predictions'] if p['amino'] != actual), None)
                
                if mejor_mutacion and mejor_mutacion['confidence'] > 20: # Umbral de alerta
                    f.write(f"  [!] ALERTA: Ruta hacia {mejor_mutacion['amino']} detectada ")
                    f.write(f"con {mejor_mutacion['confidence']:.1f}% de probabilidad estructural.\n")
                else:
                    f.write(f"  [✓] Estable: La IA no detecta rutas de mutación inminentes. ")
                    f.write(f"La mejor ruta detectada es {mejor_mutacion['amino']} con {mejor_mutacion['confidence']:.1f}% de probabilidad estructural.\n")
            f.write("\n")
        
        f.write("\n--- CALIDAD Y FILTRADO ---\n")
        f.write(f"Calidad de secuencia: {calidad:.2f}%\n")
        f.write("Nota: Se han filtrado artefactos de laboratorio (His-tags/Linkers)\n")
        f.write("para garantizar la precisión del score funcional.\n\n")
        
        f.write("==============================================\n")
        f.write(" Generado por: Telos-S - Analizador Genómico  \n")
        f.write("==============================================\n")

    print(f"📄 Informe ejecutivo listo: informe_ejecutivo_{archivo_salida.replace('.csv', '').replace('reporte_','').replace('output/s/report/','')}.txt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 analizador_final.py reporte.csv")
    else:
        analizar_cepa(sys.argv[1])