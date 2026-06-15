#!/usr/bin/env python3
"""Verificación estática básica del material entregable."""

import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "scripts_principales/depth_da3_server.py",
    "scripts_principales/realsense_pose_server.py",
    "scripts_principales/landmarks_unified.py",
    "scripts_principales/publish_mobile_pose_topics.py",
    "scripts_principales/ros2_to_android_bridge.py",
    "scripts_experimentacion/compare_realsense_da3_depth.py",
    "scripts_experimentacion/compare_depth_summaries.py",
    "entornos/environment_depth3.yml",
    "entornos/environment_openmmlab.yml",
]


def main():
    errors = []
    for relative in REQUIRED:
        if not (ROOT / relative).exists():
            errors.append(f"Falta: {relative}")

    for path in sorted((ROOT / "scripts_principales").glob("*.py")) + sorted(
        (ROOT / "scripts_experimentacion").glob("*.py")
    ):
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            errors.append(f"Error de sintaxis en {path.relative_to(ROOT)}: {exc}")

    for path in (ROOT / "entornos").rglob("*.yml"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("prefix:"):
                errors.append(f"Ruta absoluta no portable en {path.relative_to(ROOT)}")

    if errors:
        print("Verificación fallida:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Repositorio verificado: estructura mínima y sintaxis Python correctas.")


if __name__ == "__main__":
    main()
