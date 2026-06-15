# Entornos de ejecución

La arquitectura utiliza tres entornos separados para evitar conflictos entre dependencias:

- `depth3`: servidor local de Depth Anything 3, con Python 3.11.
- `openmmlab`: RTMPose, MMPose, OpenCV y RealSense, con Python 3.8.
- ROS2 Humble: nodos ejecutados con `/usr/bin/python3`, fuera de Conda.

Los archivos `environment_depth3.yml` y `environment_openmmlab.yml` son bases portables.
Después de crearlos deben instalarse las dependencias específicas siguiendo el README principal.

La carpeta `referencia/` contiene exportaciones completas del equipo original. Se conservan para
trazabilidad, pero no se recomiendan como primera opción porque incluyen versiones muy ligadas al
hardware y al momento concreto de la instalación. Se han eliminado las rutas absolutas `prefix:`.
