import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# SISTEMA DE CONFIABILIDAD
#
# El problema:
#   ESM-2 predice el aminoácido más probable en una posición basándose en el
#   contexto de los residuos vecinos. Si algún vecino es X (indeterminado),
#   el contexto está corrompido y la predicción —y por tanto el LLR y el
#   Score— no son fiables.
#
# La solución:
#   1. Identificar todas las posiciones que contienen X en el CSV.
#   2. Crear una "zona de exclusión" alrededor de cada X (ventana de ±5
#      residuos, que es aproximadamente el radio de contexto inmediato que
#      más afecta la predicción de ESM-2 en modelos de esta escala).
#   3. Clasificar cada mutación en tres niveles:
#        CONFIABLE   → fuera de cualquier zona de exclusión
#        SOSPECHOSA  → dentro de una zona de exclusión
#        INVALIDA    → contiene X directamente
#   4. Alertas, scoring y linaje solo sobre datos CONFIABLE.
#      Las SOSPECHOSAS aparecen en el heatmap (en gris) y en el informe
#      como advertencia, pero nunca disparan una alerta.
# ---------------------------------------------------------------------------

VENTANA_CONTEXTO = 5  # residuos a cada lado de un X


def cargar_predicciones_prophet(csv_path):
    base_name = csv_path.replace('.csv', '').replace('output/s/report/reporte_spike_', '')
    json_path = f"output/prophet/mutation_predictions_spike_{base_name}.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return None


def clasificar_confiabilidad(df):
    """
    Añade la columna 'Confiabilidad' a cada fila del DataFrame.

    Lógica:
      - Primero encontramos todas las posiciones donde hay un X (invalida).
      - Expandimos cada una en una zona de ±VENTANA_CONTEXTO (sospechosa).
      - Todo lo demás es confiable.
    """
    # Extraer posición numérica
    df['Pos'] = df['Mutacion'].str.extract(r'(\d+)').astype('Int64')  # Int64 nullable

    # 1. Posiciones que contienen X directamente
    mask_invalida = df['Mutacion'].str.contains('X', na=False)
    posiciones_x = set(df.loc[mask_invalida, 'Pos'].dropna().astype(int).tolist())

    # 2. Expandir zonas de exclusión
    posiciones_sospechosas = set()
    for pos_x in posiciones_x:
        for offset in range(-VENTANA_CONTEXTO, VENTANA_CONTEXTO + 1):
            posiciones_sospechosas.add(pos_x + offset)
    # Restar las posiciones X propias (esas son INVALIDA, no SOSPECHOSA)
    posiciones_sospechosas -= posiciones_x

    # 3. Clasificar cada fila (vectorizado, sin apply)
    #
    #   El orden importa: cada condición sobreescribe la anterior.
    #   Empezamos asumiendo CONFIABLE y vamos marcando excepciones.
    #
    df['Confiabilidad'] = 'CONFIABLE'

    # Sospechosa: posición dentro de la zona de exclusión
    df.loc[df['Pos'].isin(posiciones_sospechosas), 'Confiabilidad'] = 'SOSPECHOSA'

    # Inválida (sobreescribe SOSPECHOSA si aplica):
    #   - Pos es NaN (no se pudo extraer número)
    #   - El texto de la mutación contiene X
    #   - La posición coincide exactamente con una posición X conocida
    df.loc[df['Pos'].isna(),                                          'Confiabilidad'] = 'INVALIDA'
    df.loc[df['Mutacion'].str.contains('X', na=False),                'Confiabilidad'] = 'INVALIDA'
    df.loc[df['Pos'].isin(posiciones_x),                              'Confiabilidad'] = 'INVALIDA'
    return df, posiciones_x


def analizar_cepa(csv_path):
    df = pd.read_csv(csv_path)

    # --- Clasificación de confiabilidad ---
    df, posiciones_x = clasificar_confiabilidad(df)

    # --- Subconjuntos por nivel de confiabilidad ---
    df_confiable  = df[df['Confiabilidad'] == 'CONFIABLE'].copy()
    df_sospechosa = df[df['Confiabilidad'] == 'SOSPECHOSA'].copy()
    df_invalida   = df[df['Confiabilidad'] == 'INVALIDA'].copy()

    # Filtro adicional: solo posiciones biológicas de Spike (1-1273)
    df_confiable = df_confiable[
        (df_confiable['Pos'] >= 1) &
        (df_confiable['Pos'] <= 1273) &
        (~df_confiable['Mutacion'].str.startswith('-', na=False))
    ].copy()

    # ---------------------------------------------------------------------------
    # 1. Calidad de secuencia
    # ---------------------------------------------------------------------------
    total = len(df)
    n_invalidas = len(df_invalida)
    n_sospechosas = len(df_sospechosa)
    calidad = ((total - n_invalidas) / total) * 100 if total > 0 else 0.0

    # ---------------------------------------------------------------------------
    # 2. Aggression Score (solo sobre datos confiables)
    # ---------------------------------------------------------------------------
    aggression_score = df_confiable['Score'].abs().sum()

    # ---------------------------------------------------------------------------
    # 3. Identificación de linaje (solo sobre datos confiables)
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # FIRMAS DE LINAJE
    #
    # Cada entrada contiene mutaciones DIFERENCIALES: las que distinguen ese
    # linaje de los otros. No se incluyen mutaciones ancestrales compartidas
    # por casi todos los linajes post-2020 (como D614G) porque no aportan
    # poder discriminativo.
    #
    # Criterio de inclusión:
    #   - Mutaciones reportadas como "defining" en la literatura peer-reviewed
    #     o en CoV-Lineages/Nextstrain.
    #   - Se excluyen marcadores cuyo destino es X (nunca matchean con datos
    #     confiables) y mutaciones que aparecen en >3 linajes distintos.
    #
    # Fuentes:
    #   - WHO VOC/VOI tracking (2024-2025)
    #   - Nature Communications 2024 (XBB/BA.2.86/JN.1 characterization)
    #   - PMC 10661123 (BA.2.12.1 y BA.4/5 functional impact)
    #   - PMC 10782855 (Evolution of Omicron spike)
    #   - mBio 2024 (BA.2.86 → JN.1 → KP.2/KP.3 comparative)
    #   - Lancet Infect Dis 2024 (JN.1 virological characterization)
    #   - bioRxiv 2024.12.10 (KP.3.1.1 structural analysis)
    # ---------------------------------------------------------------------------
    firmas = {
        # --- VOCs originales (pre-Omicron) ---
        # Firmas propias que no se repiten en los otros VOC de esta lista.
        "Alpha (B.1.1.7)":              ["A570D", "P681H", "T716I", "S982A", "N501Y"],
        "Beta (B.1.351)":               ["K417N", "E484K", "N501Y", "A701V"],
        "Gamma (P.1)":                  ["K417T", "E484K", "N501Y", "H655Y"],
        "Delta (B.1.617.2)":            ["T19R",  "L452R", "T478K", "P681R", "D950N"],

        # --- Omicron: línea BA.1 ---
        # BA.1 se diferenció fuertemente de Wuhan en el RBD.
        # S371L es específica de BA.1 (BA.2 y descendientes tienen S371F).
        "Omicron BA.1 (B.1.1.529)":     ["G339D", "S371L", "S373P", "S375F",
                                          "K417N", "N440K", "G446S", "S477N",
                                          "E484A", "Q493R", "G498R", "N501Y", "Y505H"],

        # --- Omicron: línea BA.2 ---
        # BA.2 reemplazó a BA.1. Diferencias clave vs BA.1:
        #   S371F (vs S371L), L212I, V213G, T376A, D405N, R408S, S704L.
        "Omicron BA.2 (B.1.1.529.2)":   ["S371F", "L212I", "V213G", "T376A",
                                          "D405N", "R408S", "S477N", "E484A",
                                          "N501Y", "S704L"],

        # --- Omicron: línea BA.4/BA.5 ---
        # BA.4 y BA.5 tienen spike idéntico. Diferencias vs BA.2:
        #   del69-70, L452R, F486V, R493Q (reversion).
        # L452R y F486V son las firmas clave.
        "Omicron BA.4/BA.5":            ["L452R", "F486V", "R493Q",
                                          "N440K", "G446S", "S477N", "N501Y"],

        # --- Omicron: BQ.1.1 ("Cerberus") ---
        # Descendiente de BA.5. Añadió R346T y K526M sobre BA.5.
        # L452R y F486V las hereda de BA.5.
        "Omicron BQ.1.1":               ["R346T", "L452R", "F486V", "K526M",
                                          "N440K", "G446S", "N501Y"],

        # --- Omicron: XBB.1.5 ("Kraken") ---
        # Recombinante BA.2. Firmas RBM clave: V445P, F486P, F490S.
        # F486P (vs F486V en BA.5) es la firma más importante.
        "Omicron XBB.1.5":              ["V445P", "F486P", "F490S",
                                          "S477N", "N501Y", "G446S"],

        # --- Omicron: EG.5 ("Eris") ---
        # Descendiente de XBB.1.9.2. La firma definitoria es F456L,
        # que aparece junto a F486P heredada de XBB.
        "Omicron EG.5 (Eris)":          ["F456L", "F486P", "F490S",
                                          "V445P", "S477N", "N501Y"],

        # --- Omicron: BA.2.86 ("Pirola") ---
        # Salto evolutivo mayor: >30 mutaciones vs BA.2.
        # Firmas absolutamente únicas respecto a todos los otros linajes:
        #   D339H, N394K, A484K (reversion de E484A back a K, no es E484K),
        #   V483A, N501Y se mantiene.
        # Q493 regresa a la forma original (reversion).
        "Omicron BA.2.86 (Pirola)":     ["D339H", "N394K", "V483A", "A484K",
                                          "N501Y", "V445P", "S477N", "N440K"],

        # --- Omicron: JN.1 ("Juno") ---
        # Descendiente directo de BA.2.86. Una única mutación adicional
        # en el RBD: L455S. Esta es la firma definitoria.
        # Fue dominante global en invierno 2023-2024.
        "Omicron JN.1 (Juno)":          ["L455S", "D339H", "N394K", "V483A",
                                          "A484K", "N501Y", "V445P", "S477N"],

        # --- Omicron: KP.2 / KP.3 ("FLiRT") ---
        # Descendientes de JN.1. Añadieron R346T y F456L sobre JN.1.
        # La combinación R346T + L455S + F456L es la firma "FLiRT".
        "Omicron KP.2/KP.3 (FLiRT)":   ["R346T", "L455S", "F456L",
                                          "D339H", "N394K", "A484K", "N501Y"],

        # --- Omicron: KP.3.1.1 ---
        # Añadió Q493E sobre KP.3. Fue dominante en segundo semestre 2024.
        # La combinación L455S + F456L + Q493E tiene epistasia positiva
        # que mantiene afinidad ACE2 alta.
        "Omicron KP.3.1.1":            ["Q493E", "R346T", "L455S", "F456L",
                                          "D339H", "N394K", "A484K", "N501Y"],

        # --- Omicron: XFG ---
        # Dominante global en segundo semestre 2025 según WHO.
        # Descendiente de JN.1 con firmas convergentes similares a FLiRT.
        # Comparte L455S y F456L, diferenciador: L452R (reversion/convergencia
        # con Delta y BA.5).
        "Omicron XFG":                  ["L452R", "L455S", "F456L",
                                          "D339H", "N394K", "A484K", "N501Y"],

        # --- Omicron: BA.3.2 ("Cicada") ---
        # VUM declarado por WHO en diciembre 2025. >70 mutaciones vs Wuhan.
        # Descendiente de BA.3 ancestral (no circuló desde 2022).
        # Firmas que lo diferencian: recuperó G496S (presente en BA.1,
        # ausente desde entonces), tiene A435S, R403S, P681H.
        # NO tiene L455S (a diferencia de toda la línea JN.1).
        "Omicron BA.3.2 (Cicada)":      ["G496S", "A435S", "R403S", "P681H",
                                          "N501Y", "R493Q", "N440K"],
    }

    prediccion_linaje, max_coincidencia = identificar_linaje(df_confiable, firmas)

    # ---------------------------------------------------------------------------
    # 4. Cargar predicciones Prophet
    # ---------------------------------------------------------------------------
    datos_prophet = cargar_predicciones_prophet(csv_path)

    # ---------------------------------------------------------------------------
    # 5. Reporte en consola
    # ---------------------------------------------------------------------------
    print("=" * 50)
    print("  RESUMEN EJECUTIVO DE VARIANTE")
    print("=" * 50)
    print(f"  Calidad de secuenciación:  {calidad:.2f}%")
    print(f"  Mutaciones confiables:     {len(df_confiable)}")
    print(f"  Mutaciones sospechosas:    {n_sospechosas}  (contexto corrompido por X)")
    print(f"  Mutaciones inválidas:      {n_invalidas}  (contienen X)")
    print(f"  Aggression Score:          {aggression_score:.1f}  (solo datos confiables)")
    print(f"  Linaje probable:           {prediccion_linaje} ({max_coincidencia:.1f}%)")
    print("=" * 50)

    # ---------------------------------------------------------------------------
    # 6. Heatmap e informe
    # ---------------------------------------------------------------------------
    generar_heatmap(df_confiable, df_sospechosa, posiciones_x,
                    aggression_score, prediccion_linaje, csv_path, datos_prophet)

    generar_informe_ejecutivo(df_confiable, df_sospechosa, df_invalida,
                              aggression_score, prediccion_linaje, calidad,
                              csv_path, datos_prophet)


def identificar_linaje(df_confiable, firmas):
    """Busca marcadores de linaje solo en mutaciones confiables."""
    prediccion_linaje  = "Desconocido"
    max_coincidencia   = 0.0

    for linaje, marcadores in firmas.items():
        coincidencias = 0
        for marcador in marcadores:
            # Separar componentes del marcador (ej "K417N" → K, 417, N)
            res_mutado  = marcador[-1]
            pos_marcador = int(marcador[1:-1])

            # Ignorar marcadores que piden X como destino (no es dato real)
            if res_mutado == 'X':
                continue

            # Buscar con tolerancia de posición (±5, por posibles indels)
            rango = range(pos_marcador - 5, pos_marcador + 6)
            match = df_confiable[
                (df_confiable['Pos'].isin(rango)) &
                (df_confiable['Mutacion'].str.endswith(res_mutado, na=False))
            ]
            if not match.empty:
                coincidencias += 1

        # Calcular porcentaje excluyendo marcadores X del denominador
        marcadores_validos = sum(1 for m in marcadores if m[-1] != 'X')
        if marcadores_validos > 0:
            porcentaje = (coincidencias / marcadores_validos) * 100
        else:
            porcentaje = 0.0

        if porcentaje > max_coincidencia:
            max_coincidencia  = porcentaje
            prediccion_linaje = linaje

    return prediccion_linaje, max_coincidencia


def generar_heatmap(df_confiable, df_sospechosa, posiciones_x,
                    score_total, linaje, csv_path, datos_prophet=None):
    fig, ax = plt.subplots(figsize=(15, 7))

    # ---------------------------------------------------------------------------
    # Fondo y zonas críticas
    # ---------------------------------------------------------------------------
    ax.axhline(0, color='lightgrey', linewidth=20, alpha=0.3, zorder=1)
    ax.axvspan(319, 541, color='blue',   alpha=0.08, label='Dominio RBD')
    ax.axvspan(437, 508, color='cyan',   alpha=0.15, label='Motivo RBM')
    ax.axvspan(681, 685, color='purple', alpha=0.20, label='Sitio Furina')

    # ---------------------------------------------------------------------------
    # Zonas de exclusión por X (fondo naranja tenue)
    # ---------------------------------------------------------------------------
    if posiciones_x:
        # Agrupar posiciones X contiguas en rangos para dibujar un solo bloque
        posiciones_x_sorted = sorted(posiciones_x)
        bloques = []
        inicio = posiciones_x_sorted[0]
        fin    = posiciones_x_sorted[0]
        for pos in posiciones_x_sorted[1:]:
            if pos <= fin + VENTANA_CONTEXTO * 2 + 1:
                fin = pos  # Fusionar bloques cercanos
            else:
                bloques.append((inicio - VENTANA_CONTEXTO, fin + VENTANA_CONTEXTO))
                inicio = pos
                fin    = pos
        bloques.append((inicio - VENTANA_CONTEXTO, fin + VENTANA_CONTEXTO))

        for (x0, x1) in bloques:
            ax.axvspan(x0, x1, color='orange', alpha=0.12, zorder=2)

    # ---------------------------------------------------------------------------
    # Escala vertical dinámica
    # ---------------------------------------------------------------------------
    max_llr_confiable = df_confiable['LLR'].abs().max() if not df_confiable.empty else 1
    altura_predicciones = max(max_llr_confiable + 3, 8)
    limite_superior = max(max_llr_confiable, altura_predicciones) + 5

    # ---------------------------------------------------------------------------
    # Dibujar mutaciones CONFIABLES (sólidas, coloreadas por riesgo)
    # ---------------------------------------------------------------------------
    for _, row in df_confiable.iterrows():
        if "🔴" in str(row['Estado']):
            color = "red"
        elif row['Score'] > 30:
            color = "orange"
        else:
            color = "gold"

        altura = abs(row['LLR'])
        ax.scatter(row['Pos'], altura, color=color, s=100,
                   edgecolor='black', linewidth=0.8, zorder=5)
        ax.text(row['Pos'], altura + 0.3, row['Mutacion'],
                fontsize=7.5, rotation=45, ha='left', color='black')

    # ---------------------------------------------------------------------------
    # Dibujar mutaciones SOSPECHOSAS (gris, sin etiqueta de alerta)
    # ---------------------------------------------------------------------------
    for _, row in df_sospechosa.iterrows():
        if pd.isna(row['Pos']):
            continue
        altura = abs(row['LLR']) if pd.notna(row['LLR']) else 0
        ax.scatter(row['Pos'], altura, color='grey', s=60, alpha=0.5,
                   edgecolor='grey', linewidth=0.8, zorder=4, marker='x')
        ax.text(row['Pos'], altura + 0.2, row['Mutacion'],
                fontsize=6.5, rotation=45, ha='left', color='grey', style='italic')

    # ---------------------------------------------------------------------------
    # Predicciones Prophet (solo sobre posiciones confiables)
    # ---------------------------------------------------------------------------
    if datos_prophet:
        posiciones_confiables = set(df_confiable['Pos'].dropna().astype(int).tolist())

        for target in datos_prophet:
            pos = target['detected_position']

            # Solo dibujar si la posición está en zona confiable
            if pos not in posiciones_confiables:
                continue

            original   = target['original']
            candidatos = [p for p in target['predictions'] if p['amino'] != original]

            if candidatos:
                top_mut    = candidatos[0]
                confianza  = top_mut['confidence']

                if confianza > 5:
                    ax.scatter(pos, altura_predicciones, facecolors='none',
                               edgecolors='magenta', s=250, linestyles='--',
                               linewidth=2, zorder=6)
                    label_ia = f"{original}{pos}{top_mut['amino']}\n{confianza:.1f}%"
                    ax.text(pos, altura_predicciones + 0.8, label_ia,
                            color='darkmagenta', fontsize=8.5, fontweight='bold',
                            ha='center',
                            bbox=dict(facecolor='white', alpha=0.75,
                                      edgecolor='none', boxstyle='round'))

    # ---------------------------------------------------------------------------
    # Estética
    # ---------------------------------------------------------------------------
    ax.set_title(
        f"TELOS-S: Inteligencia de Variante {linaje} | Score: {score_total:.1f}  "
        f"(solo datos confiables)",
        fontsize=13
    )
    ax.set_xlabel("Posición en la Proteína Spike (Residuos)")
    ax.set_ylabel("Impacto Estructural (|LLR|)")
    ax.set_xlim(0, 1273)
    ax.set_ylim(-1, limite_superior)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Leyenda
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Amenaza Crítica',
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Variante de Interés',
               markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Bajo Riesgo',
               markerfacecolor='gold', markersize=10),
        Line2D([0], [0], marker='x', color='grey', label='Sospechosa (contexto X)',
               markersize=9, linestyle='None'),
        Line2D([0], [0], marker='o', color='w', label='Ruta Evolutiva (IA)',
               markeredgecolor='magenta', markerfacecolor='none', markersize=12),
        Line2D([0], [0], color='orange', lw=6, alpha=0.2, label='Zona de exclusión (X)'),
        Line2D([0], [0], color='blue',   lw=4, alpha=0.3, label='Zona RBD'),
        Line2D([0], [0], color='cyan',   lw=4, alpha=0.3, label='Zona RBM'),
        Line2D([0], [0], color='purple', lw=4, alpha=0.3, label='Zona Furina'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize='small')

    plt.tight_layout()

    # Guardar
    folder_path = "output/s/report"
    nombre_base = csv_path.replace('.csv', '').replace('output/s/report/', '')
    ruta_imagen = os.path.join(folder_path, f"heatmap_{nombre_base}.png")

    try:
        plt.savefig(ruta_imagen, dpi=150)
        print(f"🎨 Heatmap generado: {ruta_imagen}")
    except OSError as e:
        print(f"❌ Error al guardar heatmap: {e}")
    finally:
        plt.close(fig)


def generar_informe_ejecutivo(df_confiable, df_sospechosa, df_invalida,
                              score, linaje, calidad, csv_path, datos_prophet):
    """Genera el informe .txt con secciones diferenciadas por confiabilidad."""

    nombre_base = (csv_path
                   .replace('.csv', '')
                   .replace('output/s/report/', '')
                   .replace('reporte_', ''))
    ruta_informe = f"output/s/report/informe_ejecutivo_{nombre_base}.txt"

    # Top 3 amenazas (solo confiables)
    top_amenazas = df_confiable.sort_values(by='Score', ascending=False).head(3)

    # Veredicto basado en datos confiables
    if score > 1200:
        veredicto = "🔴 ALERTA MÁXIMA"
        nivel_riesgo = "CRÍTICO"
    elif score > 600:
        veredicto = "🟠 MONITOREO ACTIVO"
        nivel_riesgo = "ALTO"
    else:
        veredicto = "🟡 OBSERVACIÓN"
        nivel_riesgo = "MODERADO"

    # Referencia Wuhan para sitios de interés
    ref_wuhan_map = {
        "Sitio_RBM_452": "L",
        "Sitio_RBM_484": "E",
        "Sitio_RBM_501": "N",
        "Sitio_Furina_681": "P",
    }

    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("==============================================\n")
        f.write("   INFORME DE INTELIGENCIA GENÓMICA\n")
        f.write("   Telos-S — Análisis de Variante\n")
        f.write("==============================================\n\n")

        f.write(f"MUESTRA:              {nombre_base}\n")
        f.write(f"VEREDICTO:            {veredicto}\n")
        f.write(f"AGGRESSION SCORE:     {score:.1f}  (calculado exclusivamente sobre datos confiables)\n")
        f.write(f"LINAJE PROBABLE:      {linaje}\n\n")

        # --- Calidad y confiabilidad ---
        f.write("--- CALIDAD DE DATOS ---\n")
        f.write(f"  Calidad de secuenciación:   {calidad:.2f}%\n")
        f.write(f"  Mutaciones confiables:      {len(df_confiable)}\n")
        f.write(f"  Mutaciones sospechosas:     {len(df_sospechosa)}  "
                f"(dentro de ±{VENTANA_CONTEXTO} residuos de un X)\n")
        f.write(f"  Mutaciones inválidas:       {len(df_invalida)}  "
                f"(contienen X directamente)\n\n")

        # --- Análisis de riesgo ---
        f.write("--- ANÁLISIS DE RIESGO ---\n")
        f.write(f"  Nivel de riesgo: {nivel_riesgo}\n")
        f.write(f"  Se observa una acumulación de mutaciones en el RBD/RBM, "
                f"lo que sugiere capacidad de escape inmunológico.\n\n")

        # --- Top 3 ---
        f.write("--- TOP 3 MUTACIONES CRÍTICAS (datos confiables) ---\n")
        if not top_amenazas.empty:
            for _, row in top_amenazas.iterrows():
                f.write(f"  • {row['Mutacion']}: Zona {row['Zona']} | Score: {row['Score']:.1f}\n")
        else:
            f.write("  No se detectaron mutaciones confiables.\n")
        f.write("\n")

        # --- Mutaciones sospechosas (advertencia, sin alerta) ---
        if not df_sospechosa.empty:
            f.write("--- MUTACIONES SOSPECHOSAS (no generan alerta) ---\n")
            f.write(f"  Estas {len(df_sospechosa)} mutaciones están dentro de la zona de "
                    f"exclusión de un residuo X. Sus valores de LLR y Score no son "
                    f"fiables porque el contexto que usó ESM-2 para predecirlas estaba "
                    f"corrompido. Se listan a continuación para referencia, pero NO "
                    f"contribuyen al Score ni al veredicto.\n\n")

            df_sosp_ordenada = df_sospechosa.sort_values(by='Score', ascending=False)
            for _, row in df_sosp_ordenada.iterrows():
                if pd.isna(row['Pos']):
                    continue
                f.write(f"  ⚠ {row['Mutacion']}: Zona {row['Zona']} | "
                        f"LLR: {row['LLR']:.4f} | Score: {row['Score']:.1f} "
                        f"(NO FIABLE)\n")
            f.write("\n")

        # --- Pronóstico evolutivo Prophet ---
        if datos_prophet:
            posiciones_confiables = set(df_confiable['Pos'].dropna().astype(int).tolist())

            f.write("--- PRONÓSTICO DE EVOLUCIÓN (TELOS PROPHET) ---\n")
            f.write("  Análisis de estabilidad estructural mediante IA (ESM-2).\n")
            f.write("  Solo se incluyen predicciones sobre posiciones confiables.\n\n")

            for target in datos_prophet:
                pos      = target['detected_position']
                nombre   = target['target']
                actual   = target['original']

                # Verificar si la posición es confiable
                es_confiable = pos in posiciones_confiables

                # Buscar en el DF confiable
                match_csv = df_confiable[
                    (df_confiable['Pos'] >= pos - 2) &
                    (df_confiable['Pos'] <= pos + 2)
                ].sort_values(by='Score', ascending=False)

                if not match_csv.empty:
                    aa_actual   = match_csv.iloc[0]['Mutacion'][-1]
                    wuhan_real  = match_csv.iloc[0]['Mutacion'][0]
                else:
                    aa_actual  = actual
                    wuhan_real = ref_wuhan_map.get(nombre, "?")

                if not es_confiable:
                    f.write(f"  • {nombre} ⚠ EXCLUIDO — "
                            f"posición dentro de zona de exclusión por X. "
                            f"No se puede emitir predicción fiable.\n")
                    continue

                f.write(f"  • {nombre} (Wuhan Ref: {wuhan_real} | "
                        f"Actual: {aa_actual}):\n")

                mejor_mutacion = next(
                    (p for p in target['predictions'] if p['amino'] != actual), None
                )

                if mejor_mutacion and mejor_mutacion['confidence'] > 20:
                    f.write(f"    [!] ALERTA: Ruta hacia {mejor_mutacion['amino']} "
                            f"con {mejor_mutacion['confidence']:.1f}% de probabilidad "
                            f"estructural.\n")
                elif mejor_mutacion:
                    f.write(f"    [✓] Estable. Mejor ruta detectada: "
                            f"{mejor_mutacion['amino']} con {mejor_mutacion['confidence']:.1f}% "
                            f"de probabilidad estructural.\n")
                else:
                    f.write(f"    [✓] Estable. No se detectan rutas de mutación.\n")

            f.write("\n")

        # --- Nota final ---
        f.write("--- NOTA DE METOLOGÍA ---\n")
        f.write("  El Aggression Score y el veredicto se calculan exclusivamente sobre\n")
        f.write("  mutaciones clasificadas como CONFIABLE. Las mutaciones en zonas de\n")
        f.write(f"  exclusión (±{VENTANA_CONTEXTO} residuos de un X) se clasifican como\n")
        f.write("  SOSPECHOSA y no contribuyen a ningún indicador de alerta.\n")
        f.write("  Los artefactos de laboratorio (His-tags/Linkers) han sido filtrados.\n\n")

        f.write("==============================================\n")
        f.write("  Generado por: Telos-S — Analizador Genómico\n")
        f.write("==============================================\n")

    print(f"📄 Informe ejecutivo: {ruta_informe}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 analizador_final.py <reporte.csv>")
        print("\nEjemplo:")
        print("  python3 analizador_final.py output/s/report/reporte_spike_omicron.csv")
        sys.exit(1)

    analizar_cepa(sys.argv[1])