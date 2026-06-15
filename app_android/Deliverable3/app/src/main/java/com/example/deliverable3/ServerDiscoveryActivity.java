package com.example.deliverable3;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ListView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.util.ArrayList;

public class ServerDiscoveryActivity extends AppCompatActivity {

    private ListView listView;
    private Button scanButton;
    private Button closeButton;

    private ArrayAdapter<String> adapter;
    private final ArrayList<String> serverList = new ArrayList<>();

    private UdpPoseServerDiscovery udpDiscovery;
    private boolean resultSent = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_server_discovery);

        listView = findViewById(R.id.serverListView);
        scanButton = findViewById(R.id.scanServerButton);
        closeButton = findViewById(R.id.closeServerDiscoveryButton);

        adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, serverList);
        listView.setAdapter(adapter);

        udpDiscovery = new UdpPoseServerDiscovery(this);

        scanButton.setOnClickListener(v -> startDiscovery());
        closeButton.setOnClickListener(v -> {
            setResult(Activity.RESULT_CANCELED);
            finish();
        });

        listView.setOnItemClickListener((parent, view, position, id) -> {
            String selectedIp = serverList.get(position);
            returnIp(selectedIp);
        });

        startDiscovery();
    }

    private void startDiscovery() {
        resultSent = false;
        serverList.clear();
        adapter.notifyDataSetChanged();
        scanButton.setEnabled(false);

        Toast.makeText(this, "Buscando servidor...", Toast.LENGTH_SHORT).show();

        udpDiscovery.discover(new UdpPoseServerDiscovery.DiscoveryCallback() {
            @Override
            public void onServerFound(String ip, int port) {
                runOnUiThread(() -> {
                    if (!serverList.contains(ip)) {
                        serverList.add(ip);
                        adapter.notifyDataSetChanged();
                    }

                    scanButton.setEnabled(true);
                    returnIp(ip);
                });
            }

            @Override
            public void onError(String message) {
                runOnUiThread(() -> {
                    scanButton.setEnabled(true);
                    Toast.makeText(
                            ServerDiscoveryActivity.this,
                            "No se encontró servidor",
                            Toast.LENGTH_SHORT
                    ).show();
                });
            }
        });
    }

    private void returnIp(String ip) {
        if (resultSent) return;
        resultSent = true;

        Intent result = new Intent();
        result.putExtra("SERVER_IP", ip);
        setResult(Activity.RESULT_OK, result);
        finish();
    }
}