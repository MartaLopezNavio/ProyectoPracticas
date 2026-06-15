import argparse
import csv
import math


def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Resumen estadístico de un CSV de profundidad")
    parser.add_argument("--csv", required=True, help="Ruta al CSV")
    args = parser.parse_args()

    depths = []
    class_counts = {"CERCA": 0, "BIEN_DISTANCIA": 0, "EN_RANGO": 0, "LEJOS": 0, "INVALID": 0, "UNCLASSIFIED": 0}

    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        d = safe_float(row.get("depth"))
        if d is not None and math.isfinite(d):
            depths.append(d)

        c = row.get("classification", "UNCLASSIFIED")
        if c not in class_counts:
            class_counts[c] = 0
        class_counts[c] += 1

    print(f"CSV: {args.csv}")
    print(f"Frames válidos: {len(depths)}")

    if depths:
        mean_depth = sum(depths) / len(depths)
        min_depth = min(depths)
        max_depth = max(depths)

        print(f"Media depth: {mean_depth:.4f}")
        print(f"Mínimo depth: {min_depth:.4f}")
        print(f"Máximo depth: {max_depth:.4f}")

    print("Conteo de clases:")
    for k, v in class_counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
