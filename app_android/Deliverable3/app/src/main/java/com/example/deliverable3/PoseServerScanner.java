package com.example.deliverable3;

import android.content.Context;
import android.net.wifi.WifiManager;
import android.text.format.Formatter;
import android.util.Log;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class PoseServerScanner {

    public interface ScanCallback {
        void onServerFound(String ip);
        void onFinished(List<String> foundIps);
        void onError(String message);
    }

    private final Context context;
    private final ExecutorService executor;
    private volatile boolean isCancelled = false;

    public PoseServerScanner(Context context) {
        this.context = context.getApplicationContext();
        this.executor = Executors.newFixedThreadPool(32);
    }

    public void scan(ScanCallback callback) {
        new Thread(() -> {
            try {
                String localIp = getLocalIpAddress();
                if (localIp == null || !localIp.contains(".")) {
                    callback.onError("No se pudo obtener la IP local");
                    return;
                }

                String subnet = localIp.substring(0, localIp.lastIndexOf(".") + 1);
                List<String> foundIps = new ArrayList<>();

                final int totalHosts = 254;
                final int[] completed = {0};
                final Object lock = new Object();

                for (int i = 1; i <= 254; i++) {
                    if (isCancelled) {
                        callback.onFinished(foundIps);
                        return;
                    }

                    final String ip = subnet + i;

                    executor.submit(() -> {
                        try {
                            if (!isCancelled && isPoseServer(ip)) {
                                synchronized (foundIps) {
                                    if (!foundIps.contains(ip)) {
                                        foundIps.add(ip);
                                    }
                                }
                                callback.onServerFound(ip);
                            }
                        } finally {
                            synchronized (lock) {
                                completed[0]++;
                                if (completed[0] == totalHosts) {
                                    callback.onFinished(foundIps);
                                }
                            }
                        }
                    });
                }

            } catch (Exception e) {
                callback.onError(e.getMessage() != null ? e.getMessage() : "Error desconocido");
            }
        }).start();
    }

    private String getLocalIpAddress() {
        try {
            WifiManager wm = (WifiManager) context.getSystemService(Context.WIFI_SERVICE);
            if (wm == null || wm.getConnectionInfo() == null) {
                return null;
            }
            int ipInt = wm.getConnectionInfo().getIpAddress();
            return Formatter.formatIpAddress(ipInt);
        } catch (Exception e) {
            Log.e("PoseServerScanner", "Error obteniendo IP local", e);
            return null;
        }
    }

    private boolean isPoseServer(String ip) {
        HttpURLConnection connection = null;
        try {
            URL url = new URL("http://" + ip + ":8090/health");
            connection = (HttpURLConnection) url.openConnection();
            connection.setConnectTimeout(1000);
            connection.setReadTimeout(1000);
            connection.setRequestMethod("GET");

            int code = connection.getResponseCode();
            if (code != 200) {
                return false;
            }

            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(connection.getInputStream())
            );

            StringBuilder response = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                response.append(line);
            }
            reader.close();

            JSONObject json = new JSONObject(response.toString());
            return json.optBoolean("ok", false)
                    && "mobile_pose_server".equals(json.optString("service", ""));
        } catch (Exception e) {
            return false;
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    public void cancel() {
        isCancelled = true;
    }

    public void shutdown() {
        isCancelled = true;
        executor.shutdownNow();
    }
}