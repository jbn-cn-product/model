package com.example.model.base;

import android.graphics.Bitmap;
import com.example.model.api.Logger;
import com.example.model.api.ModelLoader;
import java.io.IOException;
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

    // 同步锁，避免推理时调用close
    private final Object lock = new Object();

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

    // 接口
    protected Logger logger;

    protected OnnxDeployer(ModelLoader modelLoader, Logger logger, String modelName, int inputWidth, int inputHeight) {
        this.logger = logger;
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
        try {
            byte[] modelData = modelLoader.getModelData(modelName);
            session = env.createSession(modelData, sessionOptions);
        } catch (IOException | OrtException e) {
            logger.error(TAG, "Failed to initialize ONNX Runtime: " + e);
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
        synchronized (lock) {
            try {
                session.close();
            } catch (OrtException e) {
                logger.error(TAG, "Failed to close ONNX session" + e);
            }
            env.close();
        }
    }

    // 运行模型
    private OrtSession.Result runSession() throws OrtException {
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputByteBuffer, inputShape, OnnxJavaType.FLOAT)) {
            return session.run(Collections.singletonMap(session.getInputInfo().keySet().iterator().next(), inputTensor));
        }
    }

    // 运行
    protected T inference(Bitmap originalBitmap) {
        synchronized (lock) {
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

}
