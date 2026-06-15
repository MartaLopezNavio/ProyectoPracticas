package com.example.deliverable3;

import android.Manifest;
import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ListView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import java.util.ArrayList;
import java.util.Set;

public class Discovery extends AppCompatActivity {

    private static final String TAG = "Discovery";

    private static final int REQUEST_ALL_BT_PERMISSIONS = 1003;

    private ArrayList<BluetoothDevice> deviceList;

    private Button Botoncerrar;
    private Button BotonStartDis;

    private BroadcastReceiver discoveryResult;
    private BluetoothAdapter bluetooth;

    private ArrayAdapter<String> deviceAdapter;
    private ListView deviceListView;

    private boolean receiverRegistered = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.discovery);

        deviceList = new ArrayList<>();

        Botoncerrar = findViewById(R.id.closeButton);
        BotonStartDis = findViewById(R.id.startDiscovery);

        bluetooth = BluetoothAdapter.getDefaultAdapter();

        deviceListView = findViewById(R.id.devicelistView);
        deviceAdapter = new ArrayAdapter<>(
                this,
                android.R.layout.simple_list_item_1,
                new ArrayList<>()
        );

        deviceListView.setAdapter(deviceAdapter);

        discoveryResult = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                String action = intent.getAction();

                if (BluetoothDevice.ACTION_FOUND.equals(action)) {
                    BluetoothDevice remoteDevice = intent.getParcelableExtra(
                            BluetoothDevice.EXTRA_DEVICE
                    );

                    int rssi = intent.getShortExtra(
                            BluetoothDevice.EXTRA_RSSI,
                            Short.MIN_VALUE
                    );

                    if (remoteDevice == null) {
                        return;
                    }

                    addDeviceToList(remoteDevice, "scan", rssi);

                } else if (BluetoothAdapter.ACTION_DISCOVERY_STARTED.equals(action)) {
                    BotonStartDis.setText("Discovering...");
                    Toast.makeText(
                            context,
                            "Discovery started",
                            Toast.LENGTH_SHORT
                    ).show();

                    Log.d(TAG, "Discovery started");

                } else if (BluetoothAdapter.ACTION_DISCOVERY_FINISHED.equals(action)) {
                    BotonStartDis.setText("Start Discovery");
                    Toast.makeText(
                            context,
                            "Discovery finished",
                            Toast.LENGTH_SHORT
                    ).show();

                    Log.d(TAG, "Discovery finished");
                }
            }
        };

        deviceListView.setOnItemClickListener((parent, view, position, id) -> {
            if (position < 0 || position >= deviceList.size()) {
                return;
            }

            BluetoothDevice selectedDevice = deviceList.get(position);

            if (!hasConnectPermission()) {
                requestDiscoveryPermission();
                return;
            }

            try {
                String name = selectedDevice.getName();
                String address = selectedDevice.getAddress();

                Toast.makeText(
                        this,
                        "Conectando a " + name,
                        Toast.LENGTH_SHORT
                ).show();

                if (bluetooth != null && bluetooth.isDiscovering()) {
                    bluetooth.cancelDiscovery();
                }

                boolean connected = BluetoothService.getInstance().connect(selectedDevice);

                if (connected) {
                    Intent result = new Intent();
                    result.putExtra("DEVICE_ADDRESS", address);
                    setResult(Activity.RESULT_OK, result);

                    Toast.makeText(
                            this,
                            "Conectado a " + name,
                            Toast.LENGTH_SHORT
                    ).show();

                    finish();

                } else {
                    Toast.makeText(
                            this,
                            "Error al conectar",
                            Toast.LENGTH_SHORT
                    ).show();
                }

            } catch (SecurityException e) {
                Log.e(TAG, "Faltan permisos al conectar", e);
                Toast.makeText(
                        this,
                        "Faltan permisos Bluetooth",
                        Toast.LENGTH_SHORT
                ).show();
            }
        });

        if (bluetooth == null) {
            Toast.makeText(
                    this,
                    "Este dispositivo no soporta Bluetooth",
                    Toast.LENGTH_LONG
            ).show();
            return;
        }

        if (!bluetooth.isEnabled()) {
            Toast.makeText(
                    this,
                    "Activa el Bluetooth antes de buscar",
                    Toast.LENGTH_LONG
            ).show();
        }

        // Añade dispositivos ya emparejados al abrir la pantalla.
        // Esto es útil porque muchos robots Bluetooth ya aparecen aquí
        // aunque el escaneo no los encuentre.
        loadBondedDevices();
    }

    private void loadBondedDevices() {
        if (bluetooth == null) return;

        if (!hasConnectPermission()) {
            return;
        }

        try {
            Set<BluetoothDevice> bondedDevices = bluetooth.getBondedDevices();

            if (bondedDevices == null) return;

            for (BluetoothDevice device : bondedDevices) {
                addDeviceToList(device, "paired", 0);
            }

        } catch (SecurityException e) {
            Log.e(TAG, "Faltan permisos para leer bonded devices", e);
        }
    }

    private void addDeviceToList(BluetoothDevice device, String source, int rssi) {
        if (device == null) return;

        if (!hasConnectPermission()) {
            return;
        }

        try {
            String name = device.getName();
            String address = device.getAddress();

            if (address == null || address.isEmpty()) {
                return;
            }

            if (name == null || name.isEmpty()) {
                name = "Unknown";
            }

            for (BluetoothDevice existing : deviceList) {
                if (existing != null && address.equals(existing.getAddress())) {
                    return;
                }
            }

            String deviceInfo;

            if ("paired".equals(source)) {
                deviceInfo = name + " (" + address + ") [paired]";
            } else {
                deviceInfo = name + " (" + address + ") RSSI=" + rssi;
            }

            deviceList.add(device);
            deviceAdapter.add(deviceInfo);
            deviceAdapter.notifyDataSetChanged();

            Log.d(TAG, "Device added: " + deviceInfo);

        } catch (SecurityException e) {
            Log.e(TAG, "Faltan permisos para leer dispositivo", e);
        }
    }

    private void startDiscovery() {
        if (bluetooth == null) {
            Toast.makeText(
                    this,
                    "Este dispositivo no soporta Bluetooth",
                    Toast.LENGTH_SHORT
            ).show();
            return;
        }

        if (!bluetooth.isEnabled()) {
            Toast.makeText(
                    this,
                    "Bluetooth está desactivado",
                    Toast.LENGTH_SHORT
            ).show();
            return;
        }

        if (!hasDiscoveryPermission()) {
            requestDiscoveryPermission();
            return;
        }

        deviceList.clear();
        deviceAdapter.clear();
        deviceAdapter.notifyDataSetChanged();

        loadBondedDevices();

        registerDiscoveryReceiverIfNeeded();

        try {
            if (bluetooth.isDiscovering()) {
                bluetooth.cancelDiscovery();
            }

            Log.d(TAG, "BluetoothAdapter: " + bluetooth);
            Log.d(TAG, "isEnabled: " + bluetooth.isEnabled());
            Log.d(TAG, "isDiscovering: " + bluetooth.isDiscovering());
            Log.d(TAG, "Scan Mode: " + bluetooth.getScanMode());

            boolean ok = bluetooth.startDiscovery();

            Log.d(TAG, "startDiscovery result: " + ok);

            if (!ok) {
                Toast.makeText(
                        this,
                        "Could not start discovery",
                        Toast.LENGTH_SHORT
                ).show();

                BotonStartDis.setText("Start Discovery");

            } else {
                BotonStartDis.setText("Discovering...");
                Toast.makeText(
                        this,
                        "Buscando dispositivos...",
                        Toast.LENGTH_SHORT
                ).show();
            }

        } catch (SecurityException e) {
            Log.e(TAG, "Faltan permisos para startDiscovery", e);
            Toast.makeText(
                    this,
                    "Faltan permisos Bluetooth",
                    Toast.LENGTH_SHORT
            ).show();
        }
    }

    private void registerDiscoveryReceiverIfNeeded() {
        if (receiverRegistered) return;

        IntentFilter filter = new IntentFilter();
        filter.addAction(BluetoothDevice.ACTION_FOUND);
        filter.addAction(BluetoothAdapter.ACTION_DISCOVERY_STARTED);
        filter.addAction(BluetoothAdapter.ACTION_DISCOVERY_FINISHED);

        registerReceiver(discoveryResult, filter);
        receiverRegistered = true;
    }

    private boolean hasDiscoveryPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            return ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.BLUETOOTH_SCAN
            ) == PackageManager.PERMISSION_GRANTED
                    &&
                    ActivityCompat.checkSelfPermission(
                            this,
                            Manifest.permission.BLUETOOTH_CONNECT
                    ) == PackageManager.PERMISSION_GRANTED;
        } else {
            return ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED;
        }
    }

    private boolean hasConnectPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            return ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.BLUETOOTH_CONNECT
            ) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestDiscoveryPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            ActivityCompat.requestPermissions(
                    this,
                    new String[]{
                            Manifest.permission.BLUETOOTH_SCAN,
                            Manifest.permission.BLUETOOTH_CONNECT
                    },
                    REQUEST_ALL_BT_PERMISSIONS
            );
        } else {
            ActivityCompat.requestPermissions(
                    this,
                    new String[]{
                            Manifest.permission.ACCESS_FINE_LOCATION
                    },
                    REQUEST_ALL_BT_PERMISSIONS
            );
        }
    }

    public void onClickClose(View view) {
        finish();
    }

    public void onClickStart(View view) {
        startDiscovery();
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            @NonNull String[] permissions,
            @NonNull int[] grantResults
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == REQUEST_ALL_BT_PERMISSIONS) {
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
                Toast.makeText(
                        this,
                        "Permisos Bluetooth concedidos",
                        Toast.LENGTH_SHORT
                ).show();

                startDiscovery();

            } else {
                Toast.makeText(
                        this,
                        "Permisos Bluetooth denegados",
                        Toast.LENGTH_LONG
                ).show();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        try {
            if (bluetooth != null && hasDiscoveryPermission() && bluetooth.isDiscovering()) {
                bluetooth.cancelDiscovery();
            }
        } catch (Exception ignored) {
        }

        try {
            if (receiverRegistered) {
                unregisterReceiver(discoveryResult);
                receiverRegistered = false;
            }
        } catch (Exception ignored) {
        }
    }
}