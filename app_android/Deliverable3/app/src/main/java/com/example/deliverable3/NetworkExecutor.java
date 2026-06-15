package com.example.deliverable3;

import android.content.Context;
import android.os.Handler;
import android.os.Message;
import android.util.Log;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.net.HttpURLConnection;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.URL;
import java.net.URLDecoder;
import java.util.StringTokenizer;

public class NetworkExecutor extends Thread {

    private static final String TAG = "NetworkExecutor";
    private static final int HTTP_SERVER_PORT = 8080;

    // IP del PC donde corre mobile_pose_server.py
    private static final String VISION_BASE_URL = "http://10.242.207.217:8090";

    private final Handler handler;
    private final Context context;
    private final String fileStr;
    private ServerSocket unSocket;
    private volatile boolean running = true;

    public final int CODE_OK = 200;
    public final int CODE_BADREQUEST = 400;
    public final int CODE_FORBIDDEN = 403;
    public final int CODE_NOTFOUND = 404;
    public final int CODE_INTERNALSERVERERROR = 500;
    public final int CODE_NOTIMPLEMENTED = 501;

    public NetworkExecutor(Context ctx, Handler handlerNetworkExecutorResult) {
        this.context = ctx.getApplicationContext();
        this.handler = handlerNetworkExecutorResult;
        this.fileStr = readResourceTextFile();
    }

    @Override
    public void run() {
        Socket scliente = null;

        try {
            unSocket = new ServerSocket(HTTP_SERVER_PORT);
            Log.d(TAG, "Servidor HTTP escuchando en puerto " + HTTP_SERVER_PORT);

            while (running) {
                try {
                    scliente = unSocket.accept();

                    System.setProperty("line.separator", "\r\n");

                    BufferedReader in = new BufferedReader(
                            new InputStreamReader(scliente.getInputStream())
                    );
                    PrintStream out = new PrintStream(
                            new BufferedOutputStream(scliente.getOutputStream())
                    );

                    String requestLine = in.readLine();
                    if (requestLine == null) {
                        scliente.close();
                        continue;
                    }

                    Log.d(TAG, "Petición cruda: " + requestLine);

                    StringTokenizer st = new StringTokenizer(requestLine);
                    String commandString = st.nextToken().toUpperCase();
                    String urlObjectString = "/";
                    if (st.hasMoreTokens()) {
                        urlObjectString = st.nextToken();
                    }

                    Log.v(TAG, "commandString=" + commandString);
                    Log.v(TAG, "urlObjectString=" + urlObjectString);

                    String line;
                    while ((line = in.readLine()) != null && line.length() > 0) {
                        // consumir cabeceras
                    }

                    if (commandString.equals("GET")) {

                        if (urlObjectString.toUpperCase().startsWith("/INDEX.HTML") ||
                                urlObjectString.toUpperCase().equals("/INDEX.HTM") ||
                                urlObjectString.equals("/")) {

                            sendIndexPage(out);

                        } else if (urlObjectString.toUpperCase().startsWith("/FORWARD")) {

                            showDisplayMessage("FORWARD");
                            sendIndexPage(out);

                        } else if (urlObjectString.toUpperCase().startsWith("/BACKWARD")) {

                            showDisplayMessage("BACKWARD");
                            sendIndexPage(out);

                        } else if (urlObjectString.toUpperCase().startsWith("/LEFT")) {

                            showDisplayMessage("LEFT");
                            sendIndexPage(out);

                        } else if (urlObjectString.toUpperCase().startsWith("/RIGHT")) {

                            showDisplayMessage("RIGHT");
                            sendIndexPage(out);

                        } else if (urlObjectString.toUpperCase().startsWith("/TURNLEFT")) {

                            showDisplayMessage("TURNLEFT");
                            sendIndexPage(out);

                        } else if (urlObjectString.toUpperCase().startsWith("/TURNRIGHT")) {

                            showDisplayMessage("TURNRIGHT");
                            sendIndexPage(out);

                        } else if (urlObjectString.toUpperCase().startsWith("/SPEEDUP")) {

                            showDisplayMessage("SPEEDUP");
                            sendIndexPage(out);

                        } else if (urlObjectString.toUpperCase().startsWith("/SPEEDDOWN")) {

                            showDisplayMessage("SPEEDDOWN");
                            sendIndexPage(out);

                        } else if (urlObjectString.toUpperCase().startsWith("/STOP")) {

                            showDisplayMessage("STOP");
                            sendIndexPage(out);

                        } else if (urlObjectString.toUpperCase().startsWith("/MESSAGE")) {

                            String messageText = extractTextParam(urlObjectString);
                            showDisplayMessage("MSG:" + messageText);
                            sendIndexPage(out);

                        } else if (urlObjectString.toUpperCase().startsWith("/CAMERA.JPG") ||
                                urlObjectString.toUpperCase().startsWith("/CAMERA.")) {

                            proxyBinary(out, VISION_BASE_URL + "/frame.jpg", "image/jpeg");

                        } else if (urlObjectString.toUpperCase().startsWith("/LANDMARKS")) {

                            proxyText(out, VISION_BASE_URL + "/landmarks", "application/json");

                        } else {
                            sendTextResponse(out, CODE_NOTFOUND, "text/plain",
                                    "UNKNOWN RESOURCE: " + urlObjectString);
                        }

                    } else {
                        sendTextResponse(out, CODE_NOTIMPLEMENTED, "text/plain",
                                "ONLY GET IMPLEMENTED");
                    }

                    out.flush();
                    scliente.close();

                } catch (Exception e) {
                    Log.e(TAG, "Error manejando cliente", e);
                    if (scliente != null) {
                        try {
                            scliente.close();
                        } catch (Exception ignored) {}
                    }
                }
            }

        } catch (Exception e) {
            Log.e(TAG, "Error en servidor HTTP", e);
        } finally {
            try {
                if (unSocket != null && !unSocket.isClosed()) {
                    unSocket.close();
                }
            } catch (Exception ignored) {}
        }
    }

    private void sendIndexPage(PrintStream out) {
        String headerStr = getHTTP_Header(CODE_OK, "text/html", fileStr.length());
        out.print(headerStr);
        out.println(fileStr);
        out.flush();
    }

    public String readResourceTextFile() {
        StringBuilder fileBuilder = new StringBuilder();
        try {
            InputStream is = context.getResources().openRawResource(R.raw.index);
            BufferedReader br = new BufferedReader(new InputStreamReader(is));
            String readLine;

            while ((readLine = br.readLine()) != null) {
                fileBuilder.append(readLine).append("\r\n");
            }

            br.close();
            is.close();
        } catch (Exception e) {
            Log.e(TAG, "Error leyendo index.html de res/raw", e);
        }
        return fileBuilder.toString();
    }

    private String getHTTP_HeaderStatus(int headerStatusCode) {
        String result = "";
        switch (headerStatusCode) {
            case CODE_OK:
                result = "200 OK";
                break;
            case CODE_BADREQUEST:
                result = "400 Bad Request";
                break;
            case CODE_FORBIDDEN:
                result = "403 Forbidden";
                break;
            case CODE_NOTFOUND:
                result = "404 Not Found";
                break;
            case CODE_INTERNALSERVERERROR:
                result = "500 Internal Server Error";
                break;
            case CODE_NOTIMPLEMENTED:
                result = "501 Not Implemented";
                break;
        }
        return "HTTP/1.0 " + result;
    }

    private String getHTTP_HeaderContentLength(int headerFileLength) {
        return "Content-Length: " + headerFileLength + "\r\n";
    }

    private String getHTTP_HeaderContentType(String headerContentType) {
        return "Content-Type: " + headerContentType + "\r\n";
    }

    private String getHTTP_Header(int headerStatusCode, String headerContentType, int headerFileLength) {
        return getHTTP_HeaderStatus(headerStatusCode) +
                "\r\n" +
                getHTTP_HeaderContentLength(headerFileLength) +
                getHTTP_HeaderContentType(headerContentType) +
                "\r\n";
    }

    private void sendTextResponse(PrintStream out, int statusCode, String mimeType, String body) {
        int length = body.length();
        String header = getHTTP_Header(statusCode, mimeType, length);
        out.print(header);
        out.print(body);
        out.flush();
    }

    private void showDisplayMessage(String msg) {
        sendCommandToMain(msg);
    }

    private void sendCommandToMain(String cmd) {
        if (handler == null) return;
        Message msg = handler.obtainMessage();
        msg.obj = cmd;
        handler.sendMessage(msg);
    }

    private String extractTextParam(String urlObjectString) {
        try {
            int qIndex = urlObjectString.indexOf('?');
            if (qIndex == -1 || qIndex == urlObjectString.length() - 1) {
                return "";
            }
            String query = urlObjectString.substring(qIndex + 1);
            String[] params = query.split("&");
            for (String p : params) {
                int eqIndex = p.indexOf('=');
                if (eqIndex > 0) {
                    String name = p.substring(0, eqIndex);
                    String value = p.substring(eqIndex + 1);
                    if (name.equals("text")) {
                        return URLDecoder.decode(value, "UTF-8");
                    }
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Error extrayendo parámetro text", e);
        }
        return "";
    }

    private void proxyBinary(PrintStream out, String targetUrl, String mimeType) {
        HttpURLConnection conn = null;
        InputStream is = null;

        try {
            URL url = new URL(targetUrl);
            conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(2000);
            conn.setReadTimeout(4000);

            int code = conn.getResponseCode();
            if (code != 200) {
                sendTextResponse(out, CODE_NOTFOUND, "text/plain", "UPSTREAM NOT AVAILABLE");
                return;
            }

            is = conn.getInputStream();
            byte[] data = readAllBytes(is);

            String headerStr = getHTTP_Header(CODE_OK, mimeType, data.length);
            out.print(headerStr);
            out.flush();
            out.write(data);
            out.flush();

        } catch (Exception e) {
            Log.e(TAG, "Error en proxyBinary()", e);
            sendTextResponse(out, CODE_INTERNALSERVERERROR, "text/plain", "PROXY ERROR");
        } finally {
            try {
                if (is != null) is.close();
            } catch (Exception ignored) {}
            if (conn != null) conn.disconnect();
        }
    }

    private void proxyText(PrintStream out, String targetUrl, String mimeType) {
        HttpURLConnection conn = null;
        BufferedReader br = null;

        try {
            URL url = new URL(targetUrl);
            conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(2000);
            conn.setReadTimeout(4000);

            int code = conn.getResponseCode();
            if (code != 200) {
                sendTextResponse(out, CODE_NOTFOUND, "text/plain", "UPSTREAM NOT AVAILABLE");
                return;
            }

            br = new BufferedReader(new InputStreamReader(conn.getInputStream()));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = br.readLine()) != null) {
                sb.append(line);
            }

            String body = sb.toString();
            String headerStr = getHTTP_Header(CODE_OK, mimeType, body.length());
            out.print(headerStr);
            out.print(body);
            out.flush();

        } catch (Exception e) {
            Log.e(TAG, "Error en proxyText()", e);
            sendTextResponse(out, CODE_INTERNALSERVERERROR, "text/plain", "PROXY ERROR");
        } finally {
            try {
                if (br != null) br.close();
            } catch (Exception ignored) {}
            if (conn != null) conn.disconnect();
        }
    }

    private byte[] readAllBytes(InputStream is) throws Exception {
        byte[] buffer = new byte[4096];
        int n;
        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();

        while ((n = is.read(buffer)) > 0) {
            baos.write(buffer, 0, n);
        }

        return baos.toByteArray();
    }

    public void stopServer() {
        running = false;
        try {
            if (unSocket != null && !unSocket.isClosed()) {
                unSocket.close();
            }
        } catch (Exception ignored) {}
        interrupt();
    }
}