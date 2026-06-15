# Aplicación Android y MiniCERNBot

Esta carpeta debe contener el proyecto completo de Android Studio utilizado como puente entre el
ordenador y el MiniCERNBot.

## Instalación

Para ejecutar la aplicación:

1. Abra el proyecto incluido en esta carpeta con Android Studio.
2. Conecte el teléfono Android al ordenador o seleccione un dispositivo disponible.
3. Pulse **Run** para compilar e instalar la aplicación en el teléfono.

Una vez instalada, la aplicación puede ejecutarse directamente desde el dispositivo Android.

## Funcionamiento

La aplicación realiza las siguientes tareas:

1. Se conecta por TCP/WiFi al servidor abierto por `ros2_to_android_bridge.py` en el puerto `9999`.
2. Recibe mensajes JSON terminados en salto de línea, por ejemplo:

```json
{"action": "ACERCAR", "timestamp": 0.0}
```

3. Traduce la acción simbólica a un comando de movimiento y lo envía al robot mediante Bluetooth.

Acciones previstas: `ACERCAR`, `ALEJAR`, `PARAR`, `IZQUIERDA`, `DERECHA` y sus combinaciones.

Para el funcionamiento completo, el móvil debe estar conectado a la misma red WiFi que el
ordenador y enlazado por Bluetooth con el MiniCERNBot.
