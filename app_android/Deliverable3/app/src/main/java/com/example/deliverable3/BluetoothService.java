package com.example.deliverable3;

import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothSocket;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.UUID;

public class BluetoothService {
    private static BluetoothService instance;
    private BluetoothSocket btSocket;
    private OutputStream outputStream;
    private InputStream inputStream;

    private BluetoothService() {}

    public static BluetoothService getInstance() {
        if (instance == null) instance = new BluetoothService();
        return instance;
    }

    public boolean connect(BluetoothDevice device) {
        try {
            btSocket = device.createRfcommSocketToServiceRecord(
                    UUID.fromString("00001101-0000-1000-8000-00805F9B34FB"));
            btSocket.connect();
            outputStream = btSocket.getOutputStream();
            inputStream = btSocket.getInputStream();
            Log.d("BluetoothService", "Conectado a " + device.getName());
            return true;
        } catch (IOException e) {
            Log.e("BluetoothService", "Error al conectar", e);
            disconnect();
            return false;
        }
    }

    public void sendData(byte[] data) throws IOException {
        if (outputStream != null) outputStream.write(data);
    }

    public boolean isConnected() {
        return btSocket != null && btSocket.isConnected();
    }

    public void disconnect() {
        try {
            if (btSocket != null) {
                btSocket.close();
                btSocket = null;
            }
        } catch (IOException e) {
            Log.e("BluetoothService", "Error al desconectar", e);
        }
    }
}

