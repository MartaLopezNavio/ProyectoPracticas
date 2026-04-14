# RTMPose Mobile Vision + ROS2

Sistema de visión basado en **RTMPose** para detectar pose humana a partir de la cámara de un móvil, calcular landmarks anatómicos externos aproximados y publicar los resultados tanto por **HTTP** como por **ROS2**.

Este proyecto se ha probado con un entorno muy concreto. Debido a la compatibilidad entre PyTorch, CUDA, MMPose, MMCV y el resto de dependencias, **se recomienda reproducir el entorno lo más fielmente posible**. Las versiones principales usadas en este proyecto son: Python 3.8.20, PyTorch 2.1.0, torchvision 0.16.0, mmcv 2.1.0, mmengine 0.10.7, mmpose 1.3.2, OpenCV 4.13.0 y Flask 3.0.3.

---

## Características

- Recepción de imágenes desde una app móvil por HTTP.
- Procesamiento de pose humana con RTMPose.
- Cálculo de landmarks anatómicos externos aproximados:
  - base cervical
  - tiroides estimada
  - pelvis
  - próstata estimada
- Visualización de la imagen procesada en tiempo real mediante MJPEG.
- Exportación de landmarks en formato JSON.
- Publicación de imagen, landmarks y keypoints en ROS2.

---

## Arquitectura

El sistema se divide en dos partes principales.

### 1. Servidor de visión

Recibe frames desde el móvil, ejecuta RTMPose, calcula landmarks y genera:

- una imagen procesada,
- un JSON con landmarks y keypoints,
- un stream MJPEG para visualización.

### 2. Publicador ROS2

Lee la salida generada por el servidor y la publica en ROS2 como topics.

---

## Scripts principales

### `landmarks_unified.py`
Módulo principal de visión. Se encarga de:

- ejecutar RTMPose,
- seleccionar la mejor persona detectada,
- calcular landmarks derivados,
- estimar la orientación aproximada,
- dibujar el resultado sobre la imagen.

### `mobile_pose_server.py`
Servidor Flask que:

- recibe imágenes desde el móvil,
- procesa cada frame,
- guarda los últimos resultados en disco,
- expone endpoints HTTP para imagen, stream y landmarks.

### `publish_mobile_pose_topics.py`
Nodo ROS2 que:

- lee la imagen procesada y el JSON generado,
- publica los resultados en distintos topics.

### `pose_server_discovery.py`
Script auxiliar usado por el servidor para facilitar el descubrimiento del equipo en red.

---

## Requisitos

- Ubuntu 22.04
- Miniconda o Anaconda
- ROS2 Humble instalado en el sistema
- GPU NVIDIA opcional pero recomendada
- móvil Android y ordenador en la misma red Wi-Fi

---

## Versiones del entorno usadas

Este proyecto se ha ejecutado con las siguientes versiones principales:

- Python 3.8.20
- pip 25.0.1
- torch 2.1.0
- torchvision 0.16.0
- mmcv 2.1.0
- mmengine 0.10.7
- mmpose 1.3.2
- opencv-python 4.13.0.92
- numpy 1.24.3
- flask 3.0.3

Además, el entorno se ha utilizado con CUDA disponible en PyTorch y con soporte para CUDA 12.1.

En el equipo donde se preparó este entorno, `torch.cuda.is_available()` devolvía `True` y `torch.version.cuda` devolvía `12.1`. 

---

## Estructura mínima del proyecto

Los archivos mínimos necesarios para ejecutar el sistema son:

```text
mobile_pose_server.py
landmarks_unified.py
pose_server_discovery.py
publish_mobile_pose_topics.py
README.md
```
El sistema genera automáticamente una carpeta de salida:

`mobile_output/`

donde guarda:
- `latest_frame.jpg`
- `latest_landmarks.json`

## Creación del entorno

Para reproducir el entorno de visión, primero se crea la base del entorno con Conda y después se instalan manualmente los paquetes adicionales necesarios con `python -m pip` y `python -m mim`.

```bash 
conda create --name openmmlab --file openmmlab_explicit.txt
conda activate openmmlab
```

### Instalar pip dentro del entorno

```bash
conda install pip setuptools wheel -y
```
Para comprobar que `pip` está asociado al entorno activo, puede ejecutarse:

```bash
which python
which pip
python -m pip --version
```
El intérprete de Python y `pip` deberían apuntar al entorno `openmmlab`.

Importante: aunque el entorno esté activado, conviene usar siempre python -m pip en lugar de pip a secas, para asegurarse de que la instalación se hace dentro del entorno correcto.

### Instalar OpenMIM
```bash
python -m pip install -U openmim
```
### Instalar MMEngine y MMCV
```bash
python -m mim install mmengine==0.10.7
python -m mim install mmcv==2.1.0
```

### Instalar el resto de dependencias necesarias
```bash
python -m pip install mmpose==1.3.2 mmdet==3.2.0 flask==3.0.3 opencv-python==4.13.0.92 addict==2.4.0
```


## Dependencias clave del entorno

Aunque el entorno completo debe instalarse desde los archivos exportados, las dependencias principales de este proyecto incluyen: 
- Python 3.8.20
- pytorch 2.1.0
- pytorch-cuda 12.1
- torchvision 0.16.0
- mmcv 2.1.0
- mmengine 0.10.7
- mmpose 1.3.2
- opencv-python 4.13.0.92
- numpy 1.24.3
- flask 3.0.3 
Estas son las versiones críticas que conviene no cambiar para evitar incompatibilidades.

## Ejecución del sistema

Situarse en la carpeta del proyecto y abrir varias terminales.

### Terminal 1: servidor de visión

```bash
conda activate openmmlab
python3 mobile_pose_server.py
```

### Terminal 2: publicador ROS2

```bash 
source /opt/ros/humble/setup.bash
/usr/bin/python3 publish_mobile_pose_topics.py
```

### Terminal 3: comprobación de topics (opcional)
```bash
source /opt/ros/humble/setup.bash
ros2 topic list
ros2 topic echo /pose_app/thyroid
```
Por ejemplo, puede consultarse el topic `/pose_app/thyroid`.

## Uso con la aplicación móvil

Se adjunta una aplicación Android con detección automática de IP.
Para poder utilizarla:
1. Instalar Android Studio.
2. Abrir el proyecto de la aplicación.
3. Compilarla e instalarla en el móvil.
4. Asegurarse de que el móvil y el ordenador están conectados a la misma red Wi-Fi.

### Importante sobre la red
Se ha comprobado que compartir datos directamente del móvil al ordenador no siempre funciona correctamente en este caso. Lo recomendable es: 
- conectar ambos dispositivos a una misma red Wi-Fi.
- o usar un tercer dispositivo que comparta internet a ambos. 
Si se cierra accidentalmente la pantalla de búsqueda de IP, puede volver a abrirse usando el botón Buscar PC. Los demás botones pueden ignorarse si solo se quiere probar el sistema de visión, ya que están pensados para el control manual del robot. 

## Comprobación de conexión

Una vez estén los scripts ejecutándose, se puede comprobar desde el navegador del ordenador que el servidor está activo: 
```sh 
http://IP_DEL_PC:8090/health
```
Ejemplo: 
```sh
http://10.65.42.217:8090/health
```
Si el servidor está funcionando correctamente, debería devolver una respuesta indicando que el servicio está activo. 
Para ver la imagen procesada en el navegador: 
```sh
http://IP_DEL_PC:8090/stream.mjpg
```
Este stream permite comprobar visualmente que la inferencia se está ejecutando correctamente y que la imagen procesada está llegando al servidor. 

## Endpoints HTTP disponibles

```text
GET /health 
```
Comprueba si el servidor está activo.

```text
GET /stream.mjpg 
```
Muestra el stream MJPEG con la imagen procesada.

```text
GET /frame.jpg
```
Devuelve la última imagen procesada.

```text
GET /landmarks 
```
Devuelve el último JSON con landmarks y keypoints.

```text
POST /upload_frame
```
Recibe un frame JPEG desde la aplicación móvil.

## Topics ROS2 publicados
El nodo publish_mobile_pose_topics.py publica los siguientes topics:
- /pose_app/image/compressed
- /pose_app/thyroid
- /pose_app/prostate
- /pose_app/keypoints
- /pose_app/all_landmarks
- /pose_app/debug
- /pose_app/orientation
- /pose_app/measurement_allowed

## Notas importantes

### Orden de ejecución

Primero debe ejecutarse `mobile_pose_server.py`, ya que `publish_mobile_pose_topics.py` depende de los archivos generados por el servidor en `mobile_output/`.

### GPU

Actualmente el sistema está configurado para usar GPU mediante 
```sh
device="cuda:0"
```
Si se quiere ejecutar en CPU, debe modificarse el código correspondiente.

### ROS2

El script ROS2 se ejecuta con: 
```sh
/usr/bin/python3 publish_mobile_pose_topics.py
```
Eso significa que ROS2 debe estar correctamente instalado en el sistema, no solo dentro del entorno Conda. 

## Resolución de problemas frecuentes

### El móvil no encuentra el PC
- Comprobar que ambos dispositivos están en la misma Wi-Fi.
- Comprobar que el firewall no bloquea el puerto `8090`.
- Verificar que el servidor está levantado accediendo a `/health desde el navegador del PC.

