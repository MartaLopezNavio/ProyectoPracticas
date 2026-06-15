import argparse
import csv


def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def classify_depth(depth_value, near_threshold=None, far_threshold=None):
    if depth_value is None:
        return "INVALID"

    if near_threshold is None or far_threshold is None:
        return "UNCLASSIFIED"

    if depth_value < near_threshold:
        return "CERCA"
    elif depth_value > far_threshold:
        return "LEJOS"
    else:
        return "EN_RANGO"


def moving_average(values, window):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = [v for v in values[start:i + 1] if v is not None]
        if not chunk:
            out.append(None)
        else:
            out.append(sum(chunk) / len(chunk))
    return out


def main():
    parser = argparse.ArgumentParser(description="Aplicar media móvil a un CSV de profundidad")
    parser.add_argument("--csv", required=True, help="CSV de entrada")
    parser.add_argument("--output_csv", required=True, help="CSV de salida suavizado")
    parser.add_argument("--window", type=int, default=5, help="Ventana de media móvil")
    parser.add_argument("--near_threshold", type=float, default=None, help="Umbral de cerca")
    parser.add_argument("--far_threshold", type=float, default=None, help="Umbral de lejos")
    args = parser.parse_args()

    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    depths = [safe_float(r.get("depth")) for r in rows]
    smooth_depths = moving_average(depths, args.window)

    for row, d_smooth in zip(rows, smooth_depths):
        row["depth_smoothed"] = "" if d_smooth is None else f"{d_smooth:.6f}"
        row["classification_smoothed"] = classify_depth(
            d_smooth, args.near_threshold, args.far_threshold
        )

    fieldnames = list(rows[0].keys()) if rows else []
    if "depth_smoothed" not in fieldnames:
        fieldnames.append("depth_smoothed")
    if "classification_smoothed" not in fieldnames:
        fieldnames.append("classification_smoothed")

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV suavizado guardado en: {args.output_csv}")


if __name__ == "__main__":
    main()
