package com.example.deliverable3;

import android.content.Context;
import android.hardware.Camera;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.IOException;
import java.util.List;

public class CameraPreview extends SurfaceView implements SurfaceHolder.Callback {

    public interface PreviewFrameListener {
        void onPreviewFrame(byte[] data, Camera camera);
    }

    private SurfaceHolder mHolder;
    private Camera mCamera;
    private Camera.Size mPreviewSize;
    private List<Camera.Size> mSupportedPreviewSizes;
    private PreviewFrameListener previewFrameListener;

    public CameraPreview(Context context, Camera camera) {
        super(context);
        init(camera);
    }

    public CameraPreview(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public void setPreviewFrameListener(PreviewFrameListener listener) {
        this.previewFrameListener = listener;
    }

    private void init(Camera camera) {
        mCamera = camera;
        if (mCamera != null) {
            Camera.Parameters params = mCamera.getParameters();

            List<String> focusModes = params.getSupportedFocusModes();
            if (focusModes != null) {
                if (focusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE)) {
                    params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE);
                } else if (focusModes.contains(Camera.Parameters.FOCUS_MODE_AUTO)) {
                    params.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
                }
            }

            mSupportedPreviewSizes = params.getSupportedPreviewSizes();
            mPreviewSize = getReasonablePreviewSize(mSupportedPreviewSizes);

            if (mPreviewSize != null) {
                params.setPreviewSize(mPreviewSize.width, mPreviewSize.height);
            }

            params.setPreviewFormat(android.graphics.ImageFormat.NV21);
            mCamera.setParameters(params);
        }

        mHolder = getHolder();
        mHolder.addCallback(this);
        mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        try {
            if (mCamera != null) {
                mCamera.setDisplayOrientation(90);
                mCamera.setPreviewDisplay(holder);

                mCamera.setPreviewCallback((data, camera) -> {
                    if (previewFrameListener != null) {
                        previewFrameListener.onPreviewFrame(data, camera);
                    }
                });

                mCamera.startPreview();

                try {
                    mCamera.autoFocus(null);
                } catch (Exception e) {
                    Log.d("CameraPreview", "Error en autoFocus inicial: " + e.getMessage());
                }
            }
        } catch (IOException e) {
            Log.d("CameraPreview", "Error setting camera preview: " + e.getMessage());
        }
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        if (mCamera != null) {
            try {
                mCamera.setPreviewCallback(null);
            } catch (Exception ignored) {}
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        if (mHolder.getSurface() == null || mCamera == null) {
            return;
        }

        try {
            mCamera.stopPreview();
        } catch (Exception ignored) {}

        try {
            Camera.Parameters params = mCamera.getParameters();

            if (mPreviewSize != null) {
                params.setPreviewSize(mPreviewSize.width, mPreviewSize.height);
            }

            params.setPreviewFormat(android.graphics.ImageFormat.NV21);
            params.set("orientation", "portrait");

            List<String> focusModes = params.getSupportedFocusModes();
            if (focusModes != null) {
                if (focusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE)) {
                    params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE);
                } else if (focusModes.contains(Camera.Parameters.FOCUS_MODE_AUTO)) {
                    params.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
                }
            }

            mCamera.setParameters(params);
            mCamera.setDisplayOrientation(90);
            mCamera.setPreviewDisplay(mHolder);

            mCamera.setPreviewCallback((data, camera) -> {
                if (previewFrameListener != null) {
                    previewFrameListener.onPreviewFrame(data, camera);
                }
            });

            mCamera.startPreview();

            try {
                mCamera.autoFocus(null);
            } catch (Exception e) {
                Log.d("CameraPreview", "Error en autoFocus tras surfaceChanged: " + e.getMessage());
            }

        } catch (Exception e) {
            Log.d("CameraPreview", "Error starting camera preview: " + e.getMessage());
        }
    }

    private Camera.Size getReasonablePreviewSize(List<Camera.Size> sizes) {
        if (sizes == null || sizes.isEmpty()) {
            return null;
        }

        Camera.Size best = null;

        for (Camera.Size s : sizes) {
            if (s.width <= 640 && s.height <= 480) {
                if (best == null || (s.width * s.height > best.width * best.height)) {
                    best = s;
                }
            }
        }

        if (best != null) return best;

        return sizes.get(0);
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        if (mPreviewSize == null) {
            super.onMeasure(widthMeasureSpec, heightMeasureSpec);
            return;
        }

        int width = MeasureSpec.getSize(widthMeasureSpec);
        int height = MeasureSpec.getSize(heightMeasureSpec);

        float cameraRatio = (float) mPreviewSize.width / (float) mPreviewSize.height;
        int idealHeight = (int) (width / cameraRatio);

        if (idealHeight > height) {
            int idealWidth = (int) (height * cameraRatio);
            setMeasuredDimension(idealWidth, height);
        } else {
            setMeasuredDimension(width, idealHeight);
        }
    }
}