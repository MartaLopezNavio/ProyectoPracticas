import os
import glob
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def clean_label(label):
    if pd.isna(label):
        return "unknown"
    return str(label)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="Compara varios summary.csv de RealSense vs Depth Anything 3."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(PROJECT_ROOT / "resultados" / "depth_comparison"),
        help="Carpeta donde están los *_summary.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "resultados" / "depth_comparison" / "comparativa"),
        help="Carpeta donde guardar resultados comparativos",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pattern = os.path.join(args.input_dir, "*_summary.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No se han encontrado archivos con patrón: {pattern}")
        return

    print("Archivos encontrados:")
    for f in files:
        print(" -", f)

    dfs = []

    for file_path in files:
        df = pd.read_csv(file_path)

        # Nombre del archivo como identificador de prueba
        file_name = os.path.basename(file_path)
        df["source_file"] = file_name

        # Intenta sacar etiqueta desde el nombre si no existe otra
        if "test_label" not in df.columns:
            # Ejemplo: 20260511_120000_prueba_1_20m_summary.csv
            label = file_name.replace("_summary.csv", "")
            df["test_label"] = label

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Ordenar columnas importantes si existen
    preferred_cols = [
        "source_file",
        "test_label",
        "target",
        "method",
        "real_distance_m",
        "total_rows",
        "valid_count",
        "valid_rate",
        "mean_depth_m",
        "std_depth_m",
        "mean_frame_to_frame_diff_m",
        "mae_m",
        "rmse_m",
        "mape_percent",
    ]

    existing_cols = [c for c in preferred_cols if c in combined.columns]
    other_cols = [c for c in combined.columns if c not in existing_cols]
    combined = combined[existing_cols + other_cols]

    combined_path = os.path.join(args.output_dir, "combined_summary.csv")
    combined.to_csv(combined_path, index=False)

    print(f"\n[OK] Resumen combinado guardado en:")
    print(combined_path)

    # ============================================================
    # TABLAS PIVOT PARA COMPARAR
    # ============================================================

    metrics = [
        "valid_rate",
        "mean_depth_m",
        "std_depth_m",
        "mean_frame_to_frame_diff_m",
        "mae_m",
        "rmse_m",
        "mape_percent",
    ]

    for metric in metrics:
        if metric not in combined.columns:
            continue

        pivot = combined.pivot_table(
            index=["test_label", "real_distance_m", "target"],
            columns="method",
            values=metric,
            aggfunc="mean",
        ).reset_index()

        pivot_path = os.path.join(args.output_dir, f"pivot_{metric}.csv")
        pivot.to_csv(pivot_path, index=False)

        print(f"[OK] Tabla {metric}: {pivot_path}")

    # ============================================================
    # RANKING GLOBAL
    # ============================================================

    ranking_rows = []

    for method in combined["method"].dropna().unique():
        df_m = combined[combined["method"] == method]

        row = {
            "method": method,
            "mean_valid_rate": df_m["valid_rate"].mean() if "valid_rate" in df_m else None,
            "mean_std_depth_m": df_m["std_depth_m"].mean() if "std_depth_m" in df_m else None,
            "mean_frame_to_frame_diff_m": df_m["mean_frame_to_frame_diff_m"].mean()
            if "mean_frame_to_frame_diff_m" in df_m else None,
            "mean_mae_m": df_m["mae_m"].mean() if "mae_m" in df_m else None,
            "mean_rmse_m": df_m["rmse_m"].mean() if "rmse_m" in df_m else None,
            "mean_mape_percent": df_m["mape_percent"].mean() if "mape_percent" in df_m else None,
        }

        ranking_rows.append(row)

    ranking = pd.DataFrame(ranking_rows)

    ranking_path = os.path.join(args.output_dir, "ranking_global.csv")
    ranking.to_csv(ranking_path, index=False)

    print(f"[OK] Ranking global guardado en:")
    print(ranking_path)

    print("\n================ RANKING GLOBAL ================")
    print(ranking.to_string(index=False))

    # ============================================================
    # GRÁFICAS
    # ============================================================

    plot_metrics = [
        ("mae_m", "Error absoluto medio MAE (m)", "Menor es mejor"),
        ("rmse_m", "RMSE (m)", "Menor es mejor"),
        ("std_depth_m", "Desviación típica temporal (m)", "Menor es mejor"),
        ("mean_frame_to_frame_diff_m", "Cambio medio entre frames (m)", "Menor es mejor"),
        ("valid_rate", "Tasa de valores válidos", "Mayor es mejor"),
        ("mape_percent", "MAPE (%)", "Menor es mejor"),
    ]

    for metric, title, subtitle in plot_metrics:
        if metric not in combined.columns:
            continue

        df_plot = combined.dropna(subset=[metric]).copy()

        if df_plot.empty:
            continue

        # Una gráfica por target
        for target in sorted(df_plot["target"].dropna().unique()):
            df_t = df_plot[df_plot["target"] == target]

            if df_t.empty:
                continue

            labels = []
            values = []

            grouped = df_t.groupby(["test_label", "method"], dropna=False)[metric].mean()

            for (test_label, method), value in grouped.items():
                labels.append(f"{test_label}\n{method}")
                values.append(value)

            plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
            plt.bar(labels, values)
            plt.title(f"{title} - {target}\n{subtitle}")
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            out_path = os.path.join(args.output_dir, f"plot_{metric}_{target}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()

            print(f"[OK] Gráfica guardada: {out_path}")

    print("\nComparativa terminada.")


if __name__ == "__main__":
    main()
