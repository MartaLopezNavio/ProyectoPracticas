package com.example.deliverable3;

import android.content.Context;
import android.net.wifi.WifiManager;
import android.text.format.Formatter;
import android.util.Log;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.util.HashSet;
import java.util.Set;

public class UdpPoseServerDiscovery {

    public interface DiscoveryCallback {
        void onServerFound(String ip, int port);
        void onError(String message);
    }

    private static final String TAG = "UdpPoseServerDiscovery";
    private static final int DISCOVERY_PORT = 8091;
    private static final String DISCOVERY_MESSAGE = "WHO_IS_POSE_SERVER";

    private final Context context;

    public UdpPoseServerDiscovery(Context context) {
        this.context = context.getApplicationContext();
    }

    public void discover(DiscoveryCallback callback) {
        new Thread(() -> {
            DatagramSocket socket = null;
            try {
                socket = new DatagramSocket();
                socket.setBroadcast(true);
                socket.setSoTimeout(1000);

                byte[] sendData = DISCOVERY_MESSAGE.getBytes();

                // 1) Broadcast global
                sendPacket(socket, sendData, "255.255.255.255");

                // 2) Broadcast de subred
                String subnetBroadcast = getSubnetBroadcastAddress();
                if (subnetBroadcast != null) {
                    sendPacket(socket, sendData, subnetBroadcast);
                }

                Log.d(TAG, "Broadcast enviado");

                long endTime = System.currentTimeMillis() + 4000;
                Set<String> foundIps = new HashSet<>();

                while (System.currentTimeMillis() < endTime) {
                    try {
                        byte[] recvBuf = new byte[1024];
                        DatagramPacket receivePacket = new DatagramPacket(recvBuf, recvBuf.length);
                        socket.receive(receivePacket);

                        String response = new String(
                                receivePacket.getData(),
                                0,
                                receivePacket.getLength()
                        ).trim();

                        Log.d(TAG, "Respuesta recibida: " + response);

                        if (response.startsWith("POSE_SERVER:")) {
                            String[] parts = response.split(":");
                            if (parts.length == 3) {
                                String ip = parts[1];
                                int port = Integer.parseInt(parts[2]);

                                if (!foundIps.contains(ip)) {
                                    foundIps.add(ip);
                                    callback.onServerFound(ip, port);
                                    return;
                                }
                            }
                        }
                    } catch (Exception ignored) {
                        // sigue escuchando hasta que acabe el tiempo
                    }
                }

                callback.onError("No se encontró servidor por UDP");

            } catch (Exception e) {
                callback.onError(e.getMessage() != null ? e.getMessage() : "No se encontró servidor");
            } finally {
                if (socket != null) {
                    socket.close();
                }
            }
        }).start();
    }

    private void sendPacket(DatagramSocket socket, byte[] data, String ip) {
        try {
            DatagramPacket packet = new DatagramPacket(
                    data,
                    data.length,
                    InetAddress.getByName(ip),
                    DISCOVERY_PORT
            );
            socket.send(packet);
            Log.d(TAG, "Broadcast enviado a " + ip);
        } catch (Exception e) {
            Log.e(TAG, "Error enviando a " + ip, e);
        }
    }

    private String getSubnetBroadcastAddress() {
        try {
            WifiManager wm = (WifiManager) context.getSystemService(Context.WIFI_SERVICE);
            if (wm == null || wm.getConnectionInfo() == null) return null;

            int ipInt = wm.getConnectionInfo().getIpAddress();
            String localIp = Formatter.formatIpAddress(ipInt);

            if (localIp == null || !localIp.contains(".")) return null;

            String prefix = localIp.substring(0, localIp.lastIndexOf(".") + 1);
            return prefix + "255";
        } catch (Exception e) {
            Log.e(TAG, "No se pudo calcular broadcast de subred", e);
            return null;
        }
    }
}