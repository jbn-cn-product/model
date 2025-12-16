package com.example.model.base;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Collections;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public abstract class OnnxDeployer<T> implements AutoCloseable {

    private static final String TAG = "MyLogcat-OnnxDeployer";

    // 模型环境
    private final OrtEnvironment env;
    protected OrtSession session;

    // 输入缓冲区
    private final long[] inputShape;
    private final ByteBuffer inputByteBuffer;
    private final FloatBuffer inputFloatBuffer;

    // 需要子类实现的方法
    protected abstract float[] preprocess(Bitmap originalBitmap);
    protected abstract T postprocess(OrtSession.Result sessionResult);

    protected OnnxDeployer(Context context, String modelName, int inputWidth, int inputHeight) {
        // 申请native内存
        int inputSize = 1 * 3 * inputHeight * inputWidth;
        inputByteBuffer = ByteBuffer.allocateDirect(inputSize * 4);
        inputByteBuffer.order(ByteOrder.nativeOrder());
        inputFloatBuffer = inputByteBuffer.asFloatBuffer();
        inputShape = new long[]{1, 3, inputHeight, inputWidth};
        // 创建模型环境
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = createSessionOptions();
        // 创建模型会话，打开模型文件，建立字节流
        try (InputStream is = context.getAssets().open(modelName);
             BufferedInputStream bis = new BufferedInputStream(is)) {
            byte[] modelData = new byte[is.available()];
            bis.read(modelData);
            session = env.createSession(modelData, sessionOptions);
        } catch (IOException | OrtException e) {
            Log.e(TAG, "Failed to initialize ONNX Runtime: ", e);
            throw new RuntimeException();
        }
    }

    // 配置会话选项
    protected OrtSession.SessionOptions createSessionOptions() {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        // 可扩展优化选项
        return options;
    }

    // 释放资源
    @Override
    public void close() {
        try {
            session.close();
        } catch (OrtException e) {
            Log.e(TAG, "Failed to close ONNX session", e);
        }
        env.close();
    }

    // 运行模型
    private OrtSession.Result runSession() throws OrtException {
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputByteBuffer, inputShape, OnnxJavaType.FLOAT)) {
            return session.run(Collections.singletonMap(session.getInputInfo().keySet().iterator().next(), inputTensor));
        }
    }

    // 运行
    protected T inference(Bitmap originalBitmap) {
        float[] inputData = preprocess(originalBitmap);
        // 更新输入缓冲区数据
        inputByteBuffer.rewind();
        inputFloatBuffer.rewind();
        inputFloatBuffer.put(inputData);
        try (OrtSession.Result result = runSession()) {
            return postprocess(result);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }

}
