package com.example.model.onnx.api.android;

import android.util.Log;
import com.example.model.onnx.core.OnnxDeployer;

public class Logger implements OnnxDeployer.Logger {

    @Override
    public void debug(String TAG, String text) {
        Log.d(TAG, text);
    }

    @Override
    public void info(String TAG, String text) {
        Log.i(TAG, text);
    }

    @Override
    public void error(String TAG, String text) {
        Log.e(TAG, text);
    }

}
