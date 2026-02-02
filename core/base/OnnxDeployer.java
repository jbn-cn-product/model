package com.example.model.core.base;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Collections;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public abstract class OnnxDeployer<ResultType> implements AutoCloseable {

    public interface Logger {
        void debug(String TAG, String text);
        void info(String TAG, String text);
        void error(String TAG, String text);
    }

    public static class Model {
        public byte[] data;
        public int width;
        public int height;
        public float meanValue;
        public float stdValue;
        public Model(byte[] data, int width, int height, float meanValue, float stdValue) {
            this.data = data;
            this.width = width;
            this.height = height;
            this.meanValue = meanValue;
            this.stdValue = stdValue;
        }
    }

    private static final String TAG = "MyLogcat-OnnxDeployer";

    // 同步锁，避免推理时调用close
    private final Object lock = new Object();

    // 模型配置
    private final Model model;

    // 模型环境
    private final OrtEnvironment env;
    protected OrtSession session;

    // 输入缓冲区
    private final long[] inputShape;
    private final ByteBuffer inputByteBuffer;
    private final FloatBuffer inputFloatBuffer;

    // 接口
    protected Logger logger;

    // 需要实现的方法
    protected abstract ResultType postprocess(OrtSession.Result sessionResult);

    protected OnnxDeployer(Logger logger, Model model) {
        this.logger = logger;
        this.model = model;
        // 申请native内存
        int inputSize = 1 * 3 * model.height * model.width;
        inputByteBuffer = ByteBuffer.allocateDirect(inputSize * 4);
        inputByteBuffer.order(ByteOrder.nativeOrder());
        inputFloatBuffer = inputByteBuffer.asFloatBuffer();
        inputShape = new long[]{1, 3, model.height, model.width};
        // 创建模型环境
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = createSessionOptions();
        try {
            session = env.createSession(model.data, sessionOptions);
        } catch (OrtException e) {
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

    // 归一化
    protected float[] normalize(byte[] rgbData) {
        int size = model.width * model.height;
        float[] inputData = new float[size * 3];
        for (int i = 0; i < size; i++) {
            int pixelIndex = i * 3;
            float r = (rgbData[pixelIndex] & 0xFF) / 255.0f;
            float g = (rgbData[pixelIndex + 1] & 0xFF) / 255.0f;
            float b = (rgbData[pixelIndex + 2] & 0xFF) / 255.0f;
            inputData[i] = (r - model.meanValue) / model.stdValue;
            inputData[i + size] = (g - model.meanValue) / model.stdValue;
            inputData[i + 2 * size] = (b - model.meanValue) / model.stdValue;
        }
        return inputData;
    }

    // 运行模型
    private OrtSession.Result runSession() throws OrtException {
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputByteBuffer, inputShape, OnnxJavaType.FLOAT)) {
            return session.run(Collections.singletonMap(session.getInputInfo().keySet().iterator().next(), inputTensor));
        }
    }

    // 运行
    protected ResultType inference(byte[] rgbData) {
        synchronized (lock) {
            float[] inputData = normalize(rgbData);
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
