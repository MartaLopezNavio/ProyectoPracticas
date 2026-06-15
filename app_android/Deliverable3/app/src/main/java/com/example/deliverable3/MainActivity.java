package com.example.deliverable3;

import android.Manifest;
import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import android.view.View;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.URL;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";

    private static final int REQUEST_CAMERA_PERMISSION = 1001;
    private static final int REQUEST_BLUETOOTH_PERMISSIONS = 2001;

    // ============================================================
    // ROS2 → ANDROID TCP BRIDGE
    // ============================================================

    private static final String ROS_BRIDGE_HOST = "10.203.116.217";
    private static final int ROS_BRIDGE_PORT = 9999;

    private volatile boolean rosBridgeRunning = false;
    private volatile boolean rosBridgeConnected = false;

    private Thread rosBridgeThread;
    private Socket rosBridgeSocket;

    private String lastRosAction = "";
    private long lastRosMessageMs = 0L;

    // Velocidad automática del MiniCERNBot.
    private boolean robotSpeedInitialized = false;
    private boolean combinedActionToggle = false;

    // Si con 1 no se mueve al arrancar, cambia a 2.
    private static final int AUTO_SPEED_UP_PULSES = 1;

    private final Handler rosWatchdogHandler = new Handler(Looper.getMainLooper());

    private final Runnable rosWatchdogRunnable = new Runnable() {
        @Override
        public void run() {
            long now = System.currentTimeMillis();

            if (rosBridgeRunning && rosBridgeConnected && lastRosMessageMs > 0) {
                long diff = now - lastRosMessageMs;

                if (diff > 1000) {
                    safeStopRobot();
                    lastRosAction = "PARAR";
                    status("Watchdog: PARAR por pérdida de señal ROS");
                    lastLandmarksText = "Watchdog: BT enviado 0\\n | velocidad reseteada";
                    refreshInfoText();
                }
            }

            rosWatchdogHandler.postDelayed(this, 500);
        }
    };

    // ============================================================
    // ANTIGUO SERVIDOR DE VISIÓN DEL MÓVIL
    // ============================================================

    private String poseServerBaseUrl = null;
    private String poseServerStreamUrl = null;
    private String poseServerUploadUrl = null;
    private String landmarksUrl = null;

    private static final long PREVIEW_UPLOAD_INTERVAL_MS = 150;

    private FrameLayout cameraPreviewFrameLayout;
    private CameraPreview mCameraPreview;
    private android.hardware.Camera mCamera;

    private WebView robotStreamView;
    private TextView webText;
    private TextView statusLabel;

    private BluetoothAdapter bluetooth;
    private ActivityResultLauncher<Intent> enableBtLauncher;
    private ActivityResultLauncher<Intent> serverDiscoveryLauncher;

    public static Boolean bluetoothActive = false;

    public Handler handlerNetworkExecutorResult;
    private NetworkExecutor networkExecutor;

    private final Handler visionPollHandler = new Handler(Looper.getMainLooper());
    private final Runnable visionPollRunnable = new Runnable() {
        @Override
        public void run() {
            fetchLandmarks();
            visionPollHandler.postDelayed(this, 700);
        }
    };

    private String lastLandmarksText = "Modo nuevo: esperando acciones ROS2";
    private boolean hasUploadedAtLeastOneFrame = false;

    private volatile boolean uploadInProgress = false;
    private long lastPreviewUploadMs = 0L;

    private boolean previewStreamingEnabled = false;

    private boolean serverDiscoveryOpen = false;

    private boolean pendingStartProcessedStream = false;
    private boolean processedStreamLoaded = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        cameraPreviewFrameLayout = findViewById(R.id.cameraView);
        webText = findViewById(R.id.webText);
        statusLabel = null;

        refreshInfoText();

        serverDiscoveryLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    serverDiscoveryOpen = false;

                    if (result.getResultCode() == Activity.RESULT_OK && result.getData() != null) {
                        String ip = result.getData().getStringExtra("SERVER_IP");

                        if (ip != null && !ip.isEmpty()) {
                            setServerIp(ip);
                            refreshInfoText();

                            pendingStartProcessedStream = true;
                            processedStreamLoaded = false;
                            showProcessedOverlay(false);
                        }

                    } else {
                        toast("No se seleccionó ningún servidor");
                        showProcessedOverlay(false);
                    }
                }
        );

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {

            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.CAMERA},
                    REQUEST_CAMERA_PERMISSION
            );

        } else {
            startCameraPreview();
        }

        View root = findViewById(R.id.main);

        if (root != null) {
            ViewCompat.setOnApplyWindowInsetsListener(root, (v, insets) -> {
                Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
                v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
                return insets;
            });
        }

        handlerNetworkExecutorResult = new Handler(Looper.getMainLooper()) {
            @Override
            public void handleMessage(@NonNull Message msg) {
                Object obj = msg.obj;

                if (!(obj instanceof String)) return;

                String cmd = (String) obj;
                Log.d(TAG, "Cmd: " + cmd);

                switch (cmd) {
                    case "FORWARD":
                        ensureRobotSpeedInitialized();
                        forward();
                        status("Forward (HTTP)");
                        break;

                    case "BACKWARD":
                        ensureRobotSpeedInitialized();
                        backward();
                        status("Backward (HTTP)");
                        break;

                    case "LEFT":
                        ensureRobotSpeedInitialized();
                        left();
                        status("Left lateral (HTTP)");
                        break;

                    case "RIGHT":
                        ensureRobotSpeedInitialized();
                        right();
                        status("Right lateral (HTTP)");
                        break;

                    case "TURNLEFT":
                        ensureRobotSpeedInitialized();
                        turnLeft();
                        status("Turn left (HTTP)");
                        break;

                    case "TURNRIGHT":
                        ensureRobotSpeedInitialized();
                        turnRight();
                        status("Turn right (HTTP)");
                        break;

                    case "SPEEDUP":
                        speedUp();
                        robotSpeedInitialized = true;
                        status("Speed + (HTTP)");
                        break;

                    case "SPEEDDOWN":
                        speedDown();
                        status("Speed - (HTTP)");
                        break;

                    case "CAMERA":
                        forceSinglePreviewUpload();
                        status("Capture + process");
                        break;

                    case "STOP":
                        stopAndResetSpeed("Stop HTTP");
                        status("Stop (HTTP)");
                        break;

                    default:
                        if (cmd.startsWith("MSG:")) {
                            String text = cmd.substring(4);
                            toast("Mensaje web: " + text);
                            status("MSG: " + text);
                        }
                        break;
                }

                refreshInfoText();
            }
        };

        networkExecutor = new NetworkExecutor(this, handlerNetworkExecutorResult);
        networkExecutor.start();

        bluetooth = BluetoothAdapter.getDefaultAdapter();

        if (bluetooth == null) {
            toast("Este dispositivo no soporta Bluetooth");
        }

        enableBtLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == Activity.RESULT_OK) {
                        bluetoothActive = true;
                        status("Bluetooth ENABLED");
                        openBluetoothDiscovery();

                    } else {
                        bluetoothActive = false;
                        status("Usuario canceló activar Bluetooth");
                    }

                    refreshInfoText();
                }
        );

        startRosBridgeClient();
    }

    // ============================================================
    // ROS2 TCP BRIDGE CLIENT
    // ============================================================

    private void startRosBridgeClient() {
        if (rosBridgeRunning) return;

        rosBridgeRunning = true;
        lastRosMessageMs = System.currentTimeMillis();

        rosBridgeThread = new Thread(() -> {
            while (rosBridgeRunning) {
                Socket socket = null;

                try {
                    runOnUiThread(() -> {
                        status("Conectando al PC ROS: " + ROS_BRIDGE_HOST + ":" + ROS_BRIDGE_PORT);
                        refreshInfoText();
                    });

                    socket = new Socket();
                    socket.connect(
                            new InetSocketAddress(ROS_BRIDGE_HOST, ROS_BRIDGE_PORT),
                            3000
                    );

                    rosBridgeSocket = socket;
                    rosBridgeConnected = true;
                    lastRosMessageMs = System.currentTimeMillis();

                    runOnUiThread(() -> {
                        status("Conectado al puente ROS2");
                        toast("Conectado al puente ROS2");
                        refreshInfoText();
                    });

                    BufferedReader reader = new BufferedReader(
                            new InputStreamReader(socket.getInputStream())
                    );

                    String line;

                    while (rosBridgeRunning && (line = reader.readLine()) != null) {
                        handleRosBridgeLine(line);
                    }

                } catch (Exception e) {
                    Log.e(TAG, "Error en conexión TCP con puente ROS", e);

                    rosBridgeConnected = false;

                    runOnUiThread(() -> {
                        status("ROS bridge desconectado. Reintentando...");
                        refreshInfoText();
                    });

                    safeStopRobot();

                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException ignored) {
                    }

                } finally {
                    rosBridgeConnected = false;

                    try {
                        if (socket != null) socket.close();
                    } catch (Exception ignored) {
                    }

                    rosBridgeSocket = null;
                }
            }
        });

        rosBridgeThread.start();
        rosWatchdogHandler.post(rosWatchdogRunnable);
    }

    private void stopRosBridgeClient() {
        rosBridgeRunning = false;
        rosBridgeConnected = false;

        rosWatchdogHandler.removeCallbacks(rosWatchdogRunnable);

        try {
            if (rosBridgeSocket != null) {
                rosBridgeSocket.close();
            }
        } catch (Exception ignored) {
        }

        rosBridgeSocket = null;
    }

    private void handleRosBridgeLine(String line) {
        try {
            JSONObject obj = new JSONObject(line);
            String action = obj.optString("action", "SIN_ACCION");

            lastRosMessageMs = System.currentTimeMillis();

            if (action == null || action.isEmpty()) {
                action = "SIN_ACCION";
            }

            final String finalAction = action;
            lastRosAction = finalAction;

            runOnUiThread(() -> handleRosAction(finalAction));

        } catch (Exception e) {
            Log.e(TAG, "Error leyendo línea ROS bridge: " + line, e);
        }
    }

    private void ensureRobotSpeedInitialized() {
        if (robotSpeedInitialized) {
            return;
        }

        if (!isBluetoothRobotConnected()) {
            return;
        }

        for (int i = 0; i < AUTO_SPEED_UP_PULSES; i++) {
            send("2\n");
        }

        robotSpeedInitialized = true;
        lastLandmarksText = "Velocidad inicializada automáticamente";
        refreshInfoText();
    }

    private void stopAndResetSpeed(String reason) {
        stop();

        robotSpeedInitialized = false;
        combinedActionToggle = false;

        lastLandmarksText = reason + " | BT enviado: 0\\n | velocidad reseteada";
        refreshInfoText();
    }

    private void handleRosAction(String action) {
        Log.d(TAG, "ROS action recibida: " + action);

        lastLandmarksText = "ROS recibido: " + action;

        if (!isBluetoothRobotConnected()) {
            status("ROS: " + action + " recibido, pero sin Bluetooth");
            refreshInfoText();
            return;
        }

        switch (action) {
            case "ACERCAR":
                ensureRobotSpeedInitialized();
                forward();
                status("AUTO ROS: ACERCAR -> w");
                break;

            case "ALEJAR":
                ensureRobotSpeedInitialized();
                backward();
                status("AUTO ROS: ALEJAR -> s");
                break;

            case "PARAR":
            case "SIN_ACCION":
            case "SIN_MEDIDA":
                stopAndResetSpeed("AUTO ROS: PARAR");
                status("AUTO ROS: PARAR -> 0, velocidad reseteada");
                break;

            case "IZQUIERDA":
                ensureRobotSpeedInitialized();
                left();
                status("AUTO ROS: IZQUIERDA LATERAL -> q");
                break;

            case "DERECHA":
                ensureRobotSpeedInitialized();
                right();
                status("AUTO ROS: DERECHA LATERAL -> e");
                break;

            case "ACERCAR_IZQUIERDA":
                ensureRobotSpeedInitialized();

                if (combinedActionToggle) {
                    forward();
                    status("AUTO ROS: ACERCAR_IZQUIERDA -> w");
                } else {
                    left();
                    status("AUTO ROS: ACERCAR_IZQUIERDA -> q");
                }

                combinedActionToggle = !combinedActionToggle;
                break;

            case "ACERCAR_DERECHA":
                ensureRobotSpeedInitialized();

                if (combinedActionToggle) {
                    forward();
                    status("AUTO ROS: ACERCAR_DERECHA -> w");
                } else {
                    right();
                    status("AUTO ROS: ACERCAR_DERECHA -> e");
                }

                combinedActionToggle = !combinedActionToggle;
                break;

            case "ALEJAR_IZQUIERDA":
                ensureRobotSpeedInitialized();

                if (combinedActionToggle) {
                    backward();
                    status("AUTO ROS: ALEJAR_IZQUIERDA -> s");
                } else {
                    left();
                    status("AUTO ROS: ALEJAR_IZQUIERDA -> q");
                }

                combinedActionToggle = !combinedActionToggle;
                break;

            case "ALEJAR_DERECHA":
                ensureRobotSpeedInitialized();

                if (combinedActionToggle) {
                    backward();
                    status("AUTO ROS: ALEJAR_DERECHA -> s");
                } else {
                    right();
                    status("AUTO ROS: ALEJAR_DERECHA -> e");
                }

                combinedActionToggle = !combinedActionToggle;
                break;

            default:
                stopAndResetSpeed("AUTO ROS: comando desconocido");
                status("AUTO ROS: comando desconocido -> 0, velocidad reseteada");
                break;
        }

        refreshInfoText();
    }

    private boolean isBluetoothRobotConnected() {
        try {
            BluetoothService service = BluetoothService.getInstance();
            return service != null && service.isConnected();

        } catch (Exception e) {
            return false;
        }
    }

    private void safeStopRobot() {
        try {
            BluetoothService service = BluetoothService.getInstance();

            if (service != null && service.isConnected()) {
                service.sendData("0\n".getBytes());
                Log.d(TAG, "BT enviado por safeStopRobot: 0\\n");
            }

            robotSpeedInitialized = false;
            combinedActionToggle = false;

        } catch (Exception e) {
            Log.e(TAG, "Error en safeStopRobot()", e);
        }
    }

    // ============================================================
    // ANTIGUO SERVIDOR DE POSE DEL MÓVIL
    // ============================================================

    private void setServerIp(String ip) {
        poseServerBaseUrl = "http://" + ip + ":8090";
        poseServerStreamUrl = poseServerBaseUrl + "/stream.mjpg";
        poseServerUploadUrl = poseServerBaseUrl + "/upload_frame";
        landmarksUrl = poseServerBaseUrl + "/landmarks";

        resetVisionConnectionState();
        pendingStartProcessedStream = true;
        processedStreamLoaded = false;
        showProcessedOverlay(false);

        Log.d(TAG, "Servidor configurado dinámicamente: " + poseServerBaseUrl);
    }

    private boolean hasServerConfigured() {
        return poseServerBaseUrl != null
                && poseServerStreamUrl != null
                && poseServerUploadUrl != null
                && landmarksUrl != null;
    }

    private void openServerDiscovery() {
        if (serverDiscoveryOpen) return;

        serverDiscoveryOpen = true;

        Intent intent = new Intent(MainActivity.this, ServerDiscoveryActivity.class);
        serverDiscoveryLauncher.launch(intent);
    }

    public void onClickFindServer(View view) {
        toast("Modo ROS2: no se usa búsqueda de servidor de pose");
        status("Usando puente ROS2 TCP: " + ROS_BRIDGE_HOST + ":" + ROS_BRIDGE_PORT);
        refreshInfoText();
    }

    private void ensureProcessedOverlay() {
        if (robotStreamView != null) return;

        robotStreamView = new WebView(this);

        WebSettings settings = robotStreamView.getSettings();
        settings.setJavaScriptEnabled(false);
        settings.setLoadWithOverviewMode(true);
        settings.setUseWideViewPort(true);
        settings.setCacheMode(WebSettings.LOAD_NO_CACHE);

        robotStreamView.setVerticalScrollBarEnabled(false);
        robotStreamView.setHorizontalScrollBarEnabled(false);
        robotStreamView.setBackgroundColor(Color.TRANSPARENT);

        robotStreamView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView view, String url) {
                showProcessedOverlay(true);
            }

            @Override
            public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                showProcessedOverlay(false);
            }
        });

        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
        );

        robotStreamView.setLayoutParams(params);
        robotStreamView.setVisibility(View.INVISIBLE);

        if (cameraPreviewFrameLayout != null) {
            cameraPreviewFrameLayout.addView(robotStreamView);
        }
    }

    private void showProcessedOverlay(boolean show) {
        if (robotStreamView == null) return;

        if (show) {
            robotStreamView.setVisibility(View.VISIBLE);
            robotStreamView.bringToFront();

        } else {
            robotStreamView.setVisibility(View.INVISIBLE);
        }
    }

    private void startCameraPreview() {
        if (cameraPreviewFrameLayout == null) return;

        if (mCamera == null) {
            mCamera = getCameraInstance();
        }

        if (mCamera == null) {
            Log.e(TAG, "No se pudo obtener la cámara");
            return;
        }

        cameraPreviewFrameLayout.removeAllViews();

        mCameraPreview = new CameraPreview(this, mCamera);
        mCameraPreview.setPreviewFrameListener(this::onPreviewFrameAvailable);
        cameraPreviewFrameLayout.addView(mCameraPreview);

        ensureProcessedOverlay();
    }

    private void onPreviewFrameAvailable(byte[] data, android.hardware.Camera camera) {
        if (!previewStreamingEnabled) return;
        if (uploadInProgress) return;

        long now = System.currentTimeMillis();

        if (now - lastPreviewUploadMs < PREVIEW_UPLOAD_INTERVAL_MS) return;

        try {
            android.hardware.Camera.Size size = camera.getParameters().getPreviewSize();

            if (size == null) return;

            byte[] jpegBytes = nv21ToJpeg(data, size.width, size.height, 55);

            if (jpegBytes == null) return;

            byte[] rotatedJpegBytes = rotateJpeg90(jpegBytes);

            if (rotatedJpegBytes == null) return;

            lastPreviewUploadMs = now;
            uploadPreviewFrame(rotatedJpegBytes);

        } catch (Exception e) {
            Log.e(TAG, "Error procesando preview frame", e);
        }
    }

    private byte[] rotateJpeg90(byte[] jpegBytes) {
        try {
            Bitmap originalBitmap = BitmapFactory.decodeByteArray(
                    jpegBytes,
                    0,
                    jpegBytes.length
            );

            if (originalBitmap == null) return null;

            Matrix matrix = new Matrix();
            matrix.postRotate(90);

            Bitmap rotatedBitmap = Bitmap.createBitmap(
                    originalBitmap,
                    0,
                    0,
                    originalBitmap.getWidth(),
                    originalBitmap.getHeight(),
                    matrix,
                    true
            );

            ByteArrayOutputStream out = new ByteArrayOutputStream();
            rotatedBitmap.compress(Bitmap.CompressFormat.JPEG, 55, out);

            return out.toByteArray();

        } catch (Exception e) {
            Log.e(TAG, "rotateJpeg90 failed", e);
            return null;
        }
    }

    private byte[] nv21ToJpeg(byte[] nv21, int width, int height, int quality) {
        try {
            YuvImage yuvImage = new YuvImage(
                    nv21,
                    ImageFormat.NV21,
                    width,
                    height,
                    null
            );

            ByteArrayOutputStream out = new ByteArrayOutputStream();

            boolean ok = yuvImage.compressToJpeg(
                    new Rect(0, 0, width, height),
                    quality,
                    out
            );

            if (!ok) return null;

            return out.toByteArray();

        } catch (Exception e) {
            Log.e(TAG, "nv21ToJpeg failed", e);
            return null;
        }
    }

    private void uploadPreviewFrame(byte[] imageBytes) {
        if (!hasServerConfigured()) return;

        uploadInProgress = true;

        new Thread(() -> {
            HttpURLConnection conn = null;

            try {
                URL url = new URL(poseServerUploadUrl);

                conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setDoOutput(true);
                conn.setConnectTimeout(3000);
                conn.setReadTimeout(6000);
                conn.setRequestProperty("Content-Type", "image/jpeg");
                conn.setFixedLengthStreamingMode(imageBytes.length);

                OutputStream os = conn.getOutputStream();
                os.write(imageBytes);
                os.flush();
                os.close();

                int code = conn.getResponseCode();

                Log.d(TAG, "uploadPreviewFrame() HTTP code=" + code);

                if (code == 200) {
                    hasUploadedAtLeastOneFrame = true;

                    runOnUiThread(() -> {
                        if (!processedStreamLoaded
                                || robotStreamView == null
                                || robotStreamView.getVisibility() != View.VISIBLE) {
                            startProcessedStream();

                        } else {
                            showProcessedOverlay(true);
                        }
                    });
                }

            } catch (Exception e) {
                Log.e(TAG, "Error subiendo preview al servidor de pose", e);

                runOnUiThread(() -> {
                    showProcessedOverlay(false);
                    poseServerBaseUrl = null;
                    poseServerStreamUrl = null;
                    poseServerUploadUrl = null;
                    landmarksUrl = null;
                    resetVisionConnectionState();
                    refreshInfoText();
                });

            } finally {
                if (conn != null) conn.disconnect();
                uploadInProgress = false;
            }
        }).start();
    }

    private void forceSinglePreviewUpload() {
        if (mCamera == null) return;
        if (!hasServerConfigured()) return;

        lastPreviewUploadMs = 0L;
        uploadInProgress = false;
    }

    private void startProcessedStream() {
        if (robotStreamView == null) return;
        if (!hasServerConfigured()) return;

        String streamUrlWithTs = poseServerStreamUrl + "?t=" + System.currentTimeMillis();

        robotStreamView.bringToFront();
        robotStreamView.clearCache(true);
        robotStreamView.loadUrl(streamUrlWithTs);

        processedStreamLoaded = true;
    }

    private android.hardware.Camera getCameraInstance() {
        android.hardware.Camera camera = null;

        try {
            camera = android.hardware.Camera.open(0);

        } catch (Exception e) {
            Log.d(TAG, "ERROR en getCameraInstance()", e);
        }

        return camera;
    }

    private void refreshInfoText() {
        if (webText == null) return;

        String rosText = rosBridgeConnected
                ? "Conectado"
                : "Desconectado / reintentando";

        String btText = isBluetoothRobotConnected()
                ? "Conectado"
                : "No conectado";

        String text =
                "Modo: PC ROS2 → Android → Bluetooth\n" +
                        "PC ROS: " + ROS_BRIDGE_HOST + ":" + ROS_BRIDGE_PORT + "\n" +
                        "Estado ROS bridge: " + rosText + "\n" +
                        "Bluetooth robot: " + btText + "\n" +
                        "Última acción ROS: " + lastRosAction + "\n" +
                        "Info: " + lastLandmarksText;

        webText.setText(text);
    }

    private void fetchLandmarks() {
        if (!hasUploadedAtLeastOneFrame) return;
        if (!hasServerConfigured()) return;

        new Thread(() -> {
            HttpURLConnection conn = null;

            try {
                URL url = new URL(landmarksUrl);

                conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.setConnectTimeout(1500);
                conn.setReadTimeout(1500);

                int code = conn.getResponseCode();

                if (code != 200) return;

                BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream())
                );

                StringBuilder sb = new StringBuilder();
                String line;

                while ((line = br.readLine()) != null) {
                    sb.append(line);
                }

                br.close();

                JSONObject obj = new JSONObject(sb.toString());
                String text = formatLandmarksText(obj);

                runOnUiThread(() -> {
                    lastLandmarksText = text;
                    refreshInfoText();
                });

            } catch (Exception e) {
                Log.e(TAG, "Error leyendo landmarks", e);

            } finally {
                if (conn != null) conn.disconnect();
            }
        }).start();
    }

    private String formatLandmarksText(JSONObject obj) {
        boolean valid = obj.optBoolean("valid", false);

        if (!valid) return "No detection";

        String thyroid = pointToText(obj, "thyroid");
        String prostate = pointToText(obj, "prostate");

        return "thyroid=" + thyroid + " | prostate=" + prostate;
    }

    private String pointToText(JSONObject obj, String key) {
        JSONObject p = obj.optJSONObject(key);

        if (p == null) return "null";

        double x = p.optDouble("x", Double.NaN);
        double y = p.optDouble("y", Double.NaN);

        return "(" + String.format("%.1f", x) + ", " + String.format("%.1f", y) + ")";
    }

    // ============================================================
    // BLUETOOTH
    // ============================================================

    public void onClickConnectButton(View view) {
        if (bluetooth == null) {
            toast("Este dispositivo no soporta Bluetooth");
            return;
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            boolean missingConnect =
                    checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT)
                            != PackageManager.PERMISSION_GRANTED;

            boolean missingScan =
                    checkSelfPermission(Manifest.permission.BLUETOOTH_SCAN)
                            != PackageManager.PERMISSION_GRANTED;

            if (missingConnect || missingScan) {
                requestPermissions(
                        new String[]{
                                Manifest.permission.BLUETOOTH_CONNECT,
                                Manifest.permission.BLUETOOTH_SCAN
                        },
                        REQUEST_BLUETOOTH_PERMISSIONS
                );
                return;
            }

        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                    != PackageManager.PERMISSION_GRANTED) {

                ActivityCompat.requestPermissions(
                        this,
                        new String[]{Manifest.permission.ACCESS_FINE_LOCATION},
                        REQUEST_BLUETOOTH_PERMISSIONS
                );
                return;
            }
        }

        openBluetoothDiscovery();
    }

    private void openBluetoothDiscovery() {
        if (bluetooth == null) {
            toast("Este dispositivo no soporta Bluetooth");
            return;
        }

        try {
            if (bluetooth.isEnabled()) {
                String name = "BT";

                if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S ||
                        checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT)
                                == PackageManager.PERMISSION_GRANTED) {
                    name = bluetooth.getName();
                }

                toast("Bluetooth ENABLED: " + name);
                status("Bluetooth ENABLED " + name);

                Intent intent = new Intent(MainActivity.this, Discovery.class);
                startActivity(intent);

            } else {
                Intent enableBt = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
                enableBtLauncher.launch(enableBt);
            }

        } catch (SecurityException e) {
            toast("Faltan permisos Bluetooth");
            status("SecurityException Bluetooth: " + e.getMessage());
            Log.e(TAG, "Error abriendo Bluetooth", e);
        }
    }

    public void onClickDisconnectButton(View v) {
        BluetoothService.getInstance().disconnect();

        robotSpeedInitialized = false;
        combinedActionToggle = false;

        status("Desconectado");
        toast("Desconectado del dispositivo");
        refreshInfoText();
    }

    public void onClickForwardButton(View v) {
        ensureRobotSpeedInitialized();
        status("Forward");
        forward();
        refreshInfoText();
    }

    public void onClickBackwardButton(View v) {
        ensureRobotSpeedInitialized();
        status("Backward");
        backward();
        refreshInfoText();
    }

    public void onClickLeftButton(View v) {
        ensureRobotSpeedInitialized();
        status("Left lateral");
        left();
        refreshInfoText();
    }

    public void onClickRightButton(View v) {
        ensureRobotSpeedInitialized();
        status("Right lateral");
        right();
        refreshInfoText();
    }

    public void onClickStopButton(View v) {
        status("Stop");
        stopAndResetSpeed("Stop manual");
        refreshInfoText();
    }

    public void onClickIncreaseSpeed(View v) {
        speedUp();
        robotSpeedInitialized = true;
        status("Speed +");
        refreshInfoText();
    }

    public void onClickDecreaseSpeed(View v) {
        speedDown();
        status("Speed -");
        refreshInfoText();
    }

    public void onClickTurnLeft(View v) {
        ensureRobotSpeedInitialized();
        turnLeft();
        status("Turn left");
        refreshInfoText();
    }

    public void onClickTurnRight(View v) {
        ensureRobotSpeedInitialized();
        turnRight();
        status("Turn right");
        refreshInfoText();
    }

    public void forward() {
        send("w\n");
    }

    public void backward() {
        send("s\n");
    }

    public void left() {
        send("q\n");
    }

    public void right() {
        send("e\n");
    }

    public void stop() {
        send("0\n");
    }

    public void turnLeft() {
        send("a\n");
    }

    public void turnRight() {
        send("d\n");
    }

    public void speedUp() {
        send("2\n");
    }

    public void speedDown() {
        send("1\n");
    }

    private void send(String text) {
        try {
            BluetoothService service = BluetoothService.getInstance();

            if (service != null && service.isConnected()) {
                service.sendData(text.getBytes());

                String cleanText = text.replace("\n", "\\n");
                lastLandmarksText = "BT enviado: " + cleanText;

                Log.d(TAG, "BT enviado: " + cleanText);

            } else {
                lastLandmarksText = "No hay conexión Bluetooth";
                toast("No hay conexión Bluetooth");
            }

        } catch (Exception e) {
            lastLandmarksText = "Error BT: " + e.getMessage();
            toast("Error enviando datos: " + e.getMessage());
            Log.e(TAG, "Error enviando datos por Bluetooth", e);
        }

        refreshInfoText();
    }

    private void status(String msg) {
        if (statusLabel != null) statusLabel.setText(msg);

        Log.d(TAG, "STATUS: " + msg);
    }

    private void toast(final String text) {
        runOnUiThread(() ->
                Toast.makeText(MainActivity.this, text, Toast.LENGTH_SHORT).show()
        );
    }

    public void captureCamera() {
        forceSinglePreviewUpload();
    }

    public void captureCamera(View view) {
        captureCamera();
    }

    private void resetVisionConnectionState() {
        hasUploadedAtLeastOneFrame = false;
        uploadInProgress = false;
        lastPreviewUploadMs = 0L;
        processedStreamLoaded = false;
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {

            if (mCamera == null) {
                startCameraPreview();
            }
        }

        refreshInfoText();

        visionPollHandler.post(visionPollRunnable);

        if (hasServerConfigured()) {
            new Handler(Looper.getMainLooper()).postDelayed(() -> {
                forceSinglePreviewUpload();

                if (pendingStartProcessedStream || !processedStreamLoaded) {
                    startProcessedStream();
                    pendingStartProcessedStream = false;
                }
            }, 1200);

        } else {
            showProcessedOverlay(false);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();

        visionPollHandler.removeCallbacks(visionPollRunnable);

        if (mCamera != null) {
            try {
                mCamera.setPreviewCallback(null);
                mCamera.release();

            } catch (Exception e) {
                Log.e(TAG, "Error liberando cámara en onPause", e);
            }

            mCamera = null;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        visionPollHandler.removeCallbacks(visionPollRunnable);
        rosWatchdogHandler.removeCallbacks(rosWatchdogRunnable);

        stopRosBridgeClient();

        if (networkExecutor != null) {
            networkExecutor.stopServer();
        }

        BluetoothService.getInstance().disconnect();
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            @NonNull String[] permissions,
            @NonNull int[] grantResults
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                startCameraPreview();

            } else {
                toast("Permiso de cámara denegado");
            }

            return;
        }

        if (requestCode == REQUEST_BLUETOOTH_PERMISSIONS) {
            boolean allGranted = true;

            if (grantResults.length == 0) {
                allGranted = false;

            } else {
                for (int result : grantResults) {
                    if (result != PackageManager.PERMISSION_GRANTED) {
                        allGranted = false;
                        break;
                    }
                }
            }

            if (allGranted) {
                toast("Permisos Bluetooth concedidos");
                status("Permisos Bluetooth concedidos");
                openBluetoothDiscovery();

            } else {
                toast("Permisos Bluetooth denegados");
                status("Permisos Bluetooth denegados");
            }

            refreshInfoText();
        }
    }
}