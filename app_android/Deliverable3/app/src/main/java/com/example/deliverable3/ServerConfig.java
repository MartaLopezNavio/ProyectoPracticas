package com.example.deliverable3;

import android.content.Context;
import android.content.SharedPreferences;

public class ServerConfig {

    private static final String PREFS_NAME = "app_prefs";
    private static final String KEY_SERVER_IP = "server_ip";
    private static final int SERVER_PORT = 8090;

    public static void saveServerIp(Context context, String ip) {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        prefs.edit().putString(KEY_SERVER_IP, ip).apply();
    }

    public static String getServerIp(Context context) {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        return prefs.getString(KEY_SERVER_IP, null);
    }

    public static void clearServerIp(Context context) {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        prefs.edit().remove(KEY_SERVER_IP).apply();
    }

    public static String getBaseUrl(Context context) {
        String ip = getServerIp(context);
        if (ip == null || ip.isEmpty()) {
            return null;
        }
        return "http://" + ip + ":" + SERVER_PORT;
    }

    public static String getStreamUrl(Context context) {
        String base = getBaseUrl(context);
        return base != null ? base + "/stream.mjpg" : null;
    }

    public static String getLandmarksUrl(Context context) {
        String base = getBaseUrl(context);
        return base != null ? base + "/landmarks" : null;
    }

    public static String getUploadUrl(Context context) {
        String base = getBaseUrl(context);
        return base != null ? base + "/upload_frame" : null;
    }

    public static String getHealthUrl(Context context) {
        String base = getBaseUrl(context);
        return base != null ? base + "/health" : null;
    }
}