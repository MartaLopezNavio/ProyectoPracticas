# Prelocalización anatómica externa mediante pose humana y visión artificial

[![Validación estática](https://github.com/MartaLopezNavio/ProyectoPracticas/actions/workflows/validacion-estatica.yml/badge.svg)](https://github.com/MartaLopezNavio/ProyectoPracticas/actions/workflows/validacion-estatica.yml)

Material de implementación y experimentación asociado al Trabajo Final de Grado:

> **Prelocalización anatómica externa en entornos hospitalarios mediante estimación de pose humana y visión artificial** 
> Marta López Navio - Grado en Inteligencia Robótica - Universitat Jaume I - Curso 2025/2026

El sistema integra estimación de pose humana con RTMPose, cálculo geométrico de landmarks
anatómicos externos, estimación de profundidad con Depth Anything 3, publicación de datos en ROS2
y generación de acciones discretas para posicionar un MiniCERNBot frente a una persona.

## Alcance y advertencia

Este proyecto es un prototipo académico de robótica asistencial. Los puntos `thyroid` y `prostate`
son **referencias externas aproximadas** obtenidas a partir de keypoints corporales; no representan
una localización clínica exacta de órganos internos. El sistema no realiza mediciones radiológicas ni
ha sido validado como dispositivo médico.

La profundidad de Depth Anything 3 se utiliza en la arquitectura final como señal práctica de
proximidad y se interpreta mediante umbrales experimentales, suavizado temporal e histéresis.

## Arquitectura

```text
Intel RealSense RGB
        |
        v
RTMPose / MMPose -> keypoints -> landmarks externos
        |                              |
        +------------------------------+
                       |
                       v
              Depth Anything 3
                       |
                       v
        distancia + centrado horizontal
                       |
                       v
         lógica de decisión y seguridad
                       |
                       v
ROS2 -> TCP/WiFi -> Android -> Bluetooth -> MiniCERNBot
```

Acciones generadas: `ACERCAR`, `ALEJAR`, `PARAR`, `IZQUIERDA`, `DERECHA` y acciones
combinadas. Cuando la orientación no es frontal, faltan landmarks o no existe una profundidad
válida, el comportamiento seguro es detener el robot.

## Estructura del repositorio

```text
.
├── README.md
├── INSTRUCCIONES_ACTUALIZACION.md
├── CITATION.cff
├── LICENSE
├── scripts_principales/
│   ├── depth_da3_server.py
│   ├── realsense_pose_server.py
│   ├── landmarks_unified.py
│   ├── publish_mobile_pose_topics.py
│   └── ros2_to_android_bridge.py
├── scripts_experimentacion/
│   ├── compare_realsense_da3_depth.py
│   ├── compare_depth_summaries.py
│   ├── procesar_carpeta_landmarks.py
│   ├── procesar_carpeta_profundidad.py
│   ├── resumen_csv_profundidad.py
│   └── suavizar_csv_profundidad.py
├── resultados/
│   └── ejemplo_depth/
├── entornos/
│   ├── environment_depth3.yml
│   ├── environment_openmmlab.yml
│   ├── ros2_humble_notas.txt
│   └── referencia/
├── app_android/
│   ├── README.md
│   └── <proyecto_android_studio>/
├── videos/
├── documentacion/
│   └── Memoria_TFG_Marta_Lopez_Navio.pdf
├── herramientas/
│   └── verificar_repositorio.py
├── mobile_output/
└── legacy_practicas/          # opcional: fase anterior de prácticas
```

El repositorio original contenía `RTMPose/` y `ros2_ws/`. Esas carpetas pueden conservarse en
`legacy_practicas/` para mantener la trazabilidad de la fase de prácticas. La implementación final
descrita en la memoria está en `scripts_principales/`.

## Scripts principales

### `depth_da3_server.py`

Servidor HTTP local de Depth Anything 3. Carga el modelo una sola vez y expone:

- `GET /health`
- `POST /depth_at_point`
- `POST /depth_at_points`

La versión final utiliza principalmente `/depth_at_points`, de modo que una sola inferencia permite
consultar varios landmarks.

### `realsense_pose_server.py`

Proceso principal de percepción y decisión:

- captura RGB con Intel RealSense;
- ejecuta RTMPose mediante `LandmarksEngine`;
- calcula `neck_base`, `thyroid`, `pelvis` y `prostate`;
- valida la orientación frontal;
- consulta DA3 en `thyroid` y `prostate`;
- aplica mediana temporal, histéresis y bloqueo de posición;
- combina distancia y error horizontal;
- guarda la imagen y el JSON más recientes en `mobile_output/`.

Configuración principal incluida en el script:

- imagen: `640 x 480`, 15 FPS;
- rango funcional DA3: `0.8 <= depth <= 1.0`;
- tolerancia de centrado: `±50 px`;
- consulta a DA3 cada 3 frames;
- prioridad de objetivo: tiroides y, si no está disponible, próstata.

Los valores de DA3 anteriores son umbrales experimentales del montaje utilizado y no deben
interpretarse como una calibración clínica universal.

### `landmarks_unified.py`

Motor de pose y landmarks. Selecciona a la persona con mayor confianza media, filtra keypoints,
valida la frontalidad y calcula:

- `neck_base`: punto medio entre hombros;
- `thyroid`: interpolación entre base cervical y nariz;
- `pelvis`: punto medio entre caderas;
- `prostate`: desplazamiento vertical proporcional a la anchura de caderas.

### `publish_mobile_pose_topics.py`

Nodo ROS2 que lee `mobile_output/latest_frame.jpg` y
`mobile_output/latest_landmarks.json` y publica imagen, landmarks, orientación, profundidad,
estados y acción final.

### `ros2_to_android_bridge.py`

Nodo que se suscribe a `/pose_app/distance/action` y abre un servidor TCP en el puerto `9999`.
Envía a Android mensajes JSON con la última acción y una marca temporal.

## Requisitos

- Ubuntu 22.04.
- ROS2 Humble instalado en el sistema.
- Miniconda o Anaconda.
- Cámara Intel RealSense compatible con `pyrealsense2`.
- GPU NVIDIA recomendada.
- Android Studio para instalar la aplicación incluida en `app_android/`.
- Teléfono Android y ordenador en la misma red WiFi.
- MiniCERNBot enlazado por Bluetooth con el teléfono.

## Instalación

La separación de entornos es deliberada: OpenMMLab/RTMPose, Depth Anything 3 y ROS2 presentan
dependencias incompatibles en una única instalación.

### 1. Entorno Depth Anything 3

```bash
conda env create -f entornos/environment_depth3.yml
conda activate depth3
```

Descargue el repositorio oficial de DA3 fuera de este proyecto e instálelo en modo editable:

```bash
git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
python -m pip install "torch>=2" torchvision xformers
python -m pip install -e .
```

El servidor utiliza el modelo `depth-anything/DA3Metric-Large`, descargado mediante
`DepthAnything3.from_pretrained(...)`.

### 2. Entorno OpenMMLab / RTMPose

```bash
conda env create -f entornos/environment_openmmlab.yml
conda activate openmmlab
python -m pip install -U openmim
python -m mim install mmengine==0.10.7
python -m mim install mmcv==2.1.0
python -m pip install \
  mmpose==1.3.2 \
  mmdet==3.2.0 \
  opencv-python==4.13.0.92 \
  pyrealsense2==2.55.1.6486 \
  numpy==1.24.3 \
  flask==3.0.3 \
  pandas matplotlib
```

Las exportaciones completas del equipo original están en `entornos/referencia/`. Se conservan
como referencia, pero los ficheros base anteriores son más portables.

### 3. ROS2 Humble

Los nodos ROS2 se ejecutan con el Python del sistema, no con Conda:

```bash
source /opt/ros/humble/setup.bash
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=42
```

## Ejecución de la arquitectura final

Abra cuatro terminales desde la raíz del repositorio.

### Terminal 1 - servidor DA3

```bash
conda activate depth3
python scripts_principales/depth_da3_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --model depth-anything/DA3Metric-Large \
  --device cuda
```

Comprobación:

```bash
curl http://127.0.0.1:8765/health
```

### Terminal 2 - RealSense, RTMPose y decisión

```bash
conda activate openmmlab
python scripts_principales/realsense_pose_server.py
```

Pulse `Esc` en la ventana de OpenCV para detenerlo.

### Terminal 3 - publicación ROS2

```bash
conda deactivate
source /opt/ros/humble/setup.bash
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=42
/usr/bin/python3 scripts_principales/publish_mobile_pose_topics.py
```

### Terminal 4 - puente ROS2 a Android

```bash
conda deactivate
source /opt/ros/humble/setup.bash
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=42
/usr/bin/python3 scripts_principales/ros2_to_android_bridge.py
```

Antes de ejecutar el sistema completo, instale en el teléfono la aplicación incluida en
`app_android/`. Para ello, abra el proyecto con Android Studio, conecte el dispositivo Android y
ejecute la opción **Run** para compilar e instalar la aplicación.

Una vez instalada, la aplicación debe conectarse a la IP del ordenador y al puerto `9999`. Después
envía los comandos al MiniCERNBot por Bluetooth.

## Topics ROS2

Topics generales:

```text
/pose_app/image/compressed
/pose_app/thyroid
/pose_app/prostate
/pose_app/keypoints
/pose_app/all_landmarks
/pose_app/orientation
/pose_app/measurement_allowed
/pose_app/debug
```

Topics principales de distancia y decisión:

```text
/pose_app/distance/raw_depth
/pose_app/distance/smooth_depth
/pose_app/distance/instant_state
/pose_app/distance/state
/pose_app/distance/action
/pose_app/distance/target
/pose_app/distance/target_name
/pose_app/distance/measurement_allowed
/pose_app/distance/near_threshold
```

Además, se publican topics separados para `thyroid` y `prostate` bajo
`/pose_app/distance/<target>/...`.

El topic utilizado por el puente Android es:

```bash
ros2 topic echo /pose_app/distance/action
```

## Experimentación

### Comparación RealSense - DA3

Con el servidor DA3 activo:

```bash
conda activate openmmlab
python scripts_experimentacion/compare_realsense_da3_depth.py \
  --real-distance 1.20 \
  --label prueba_1_20m \
  --max-frames 300
```

Genera un CSV por frame y un resumen por método y landmark con:

- tasa de valores válidos;
- media y desviación temporal;
- cambio medio entre frames;
- MAE;
- RMSE;
- MAPE, cuando la distancia de referencia es distinta de cero.

### Comparación de resúmenes

```bash
python scripts_experimentacion/compare_depth_summaries.py
```

### Procesamiento offline

```bash
python scripts_experimentacion/procesar_carpeta_landmarks.py \
  --input-dir datos/imagenes \
  --output-dir resultados/landmarks_offline

python scripts_experimentacion/procesar_carpeta_profundidad.py \
  --images-dir datos/imagenes \
  --landmarks-dir resultados/landmarks_offline/landmarks_json \
  --output-csv resultados/profundidad_offline.csv
```

`suavizar_csv_profundidad.py` aplica una media móvil a series históricas de pruebas offline. En el
sistema final en tiempo real, el suavizado implementado en `realsense_pose_server.py` utiliza la
mediana temporal, más robusta frente a valores atípicos.

## Vídeos y material complementario

- [Demostración del movimiento autónomo del MiniCERNBot](https://drive.google.com/file/d/1jvd5K_I8cUh0hAfUboUSsmK0-VSazqaC/view?usp=drive_link)
- [Percepción y datos generados desde el robot](https://drive.google.com/file/d/1VeLyjA2Ej3m1JU9cYUsteCVZTvJ-D-AE/view?usp=drive_link)
- [Carpeta completa de material complementario](https://drive.google.com/drive/folders/1ofm7XQByVwtwYFp3-z8pua8xAuqWYnGu?usp=drive_link)
- [Memoria final del TFG](documentacion/Memoria_TFG_Marta_Lopez_Navio.pdf)

## Verificación del repositorio

La comprobación estática no ejecuta los modelos ni requiere GPU. Verifica la estructura mínima,
la sintaxis de los scripts y la ausencia de rutas absolutas en los YAML:

```bash
python herramientas/verificar_repositorio.py
```

## Limitaciones

- Landmarks externos aproximados, no clínicos.
- Validación experimental controlada y con pocas personas.
- Sensibilidad a oclusiones, ropa, iluminación y postura.
- Umbrales de profundidad dependientes del montaje.
- Acciones discretas; no incluye navegación global ni evitación de obstáculos.

## Dependencias externas

Este repositorio no redistribuye MMPose/RTMPose, Depth Anything 3 ni sus pesos. Deben obtenerse
desde sus repositorios oficiales:

- <https://github.com/open-mmlab/mmpose>
- <https://github.com/ByteDance-Seed/Depth-Anything-3>
- Repositorio de apoyo: <https://github.com/diegomarzaa/cirtesu_da3_mapping>

## Autoría

**Marta López Navio** 
Trabajo Final de Grado - Universitat Jaume I - 2026.

Consulte `CITATION.cff` para la forma de citación y `LICENSE` para las condiciones de uso.
