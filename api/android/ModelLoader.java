package com.example.model.api.android;

import android.content.Context;
import android.util.Log;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;

public class ModelLoader {

    private static final String TAG = "MyLogcat-ImageHelper";

    public static byte[] getModelData(Context context, String modelName) {
        try (InputStream is = context.getAssets().open(modelName);
             BufferedInputStream bis = new BufferedInputStream(is)) {
            byte[] modelData = new byte[is.available()];
            int length = bis.read(modelData);
            Log.d(TAG, String.format("load model [%s] success, data length: %d bytes", modelName, length));
            return modelData;
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize ONNX Runtime: " + e);
            throw new RuntimeException();
        }
    }

}
