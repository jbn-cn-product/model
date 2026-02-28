package com.example.model.onnx.core;

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

    public static class ModelConfig {
        private record StdConfig (float mean, float std) {}
        private record RetinaFaceConfig (float mean_r, float mean_g, float mean_b) {}
        private byte[] data;
        private int width;
        private int height;
        private StdConfig stdConfig;
        private RetinaFaceConfig rfConfig;
        private void init(byte[] data, int width, int height) {
            this.data = data;
            this.width = width;
            this.height = height;
        }
        public ModelConfig(byte[] data, int width, int height, float mean, float std) {
            init(data, width, height);
            this.stdConfig = new StdConfig(mean, std);
        }
        public ModelConfig(byte[] data, int width, int height, float mean_r, float mean_g, float mean_b) {
            init(data, width, height);
            this.rfConfig = new RetinaFaceConfig(mean_r, mean_g, mean_b);
        }
    }

    private static final String TAG = "MyLogcat-OnnxDeployer";

    // 同步锁，避免推理时调用close
    private final Object lock = new Object();

    // 模型配置
    private final ModelConfig modelConfig;

    // 模型环境
    private final OrtEnvironment env;
    protected OrtSession session;

    // 输入缓冲区
    private final long[] inputShape;
    private final ByteBuffer inputByteBuffer;
    private final FloatBuffer inputFloatBuffer;

    // 日志接口
    protected Logger logger;

    // 需要实现的方法
    protected abstract ResultType postprocess(OrtSession.Result sessionResult) throws OrtException;

    protected OnnxDeployer(Logger logger, ModelConfig modelConfig) {
        this.logger = logger;
        this.modelConfig = modelConfig;
        // 申请native内存
        int inputSize = 1 * 3 * modelConfig.height * modelConfig.width;
        inputByteBuffer = ByteBuffer.allocateDirect(inputSize * 4);
        inputByteBuffer.order(ByteOrder.nativeOrder());
        inputFloatBuffer = inputByteBuffer.asFloatBuffer();
        inputShape = new long[]{1, 3, modelConfig.height, modelConfig.width};
        // 创建模型环境
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = createSessionOptions();
        try {
            session = env.createSession(modelConfig.data, sessionOptions);
        } catch (OrtException e) {
            logger.error(TAG, "初始化onnx runtime环境失败: " + e);
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
                logger.error(TAG, "关闭onnx session时错误" + e);
            }
            env.close();
        }
    }

    // 归一化
    private float[] normalize(byte[] rgbData) {
        int size = modelConfig.width * modelConfig.height;
        float[] inputData = new float[size * 3];
        for (int i = 0; i < size; i++) {
            int pixelIndex = i * 3;
            float r = rgbData[pixelIndex] & 0xFF;
            float g = rgbData[pixelIndex + 1] & 0xFF;
            float b = rgbData[pixelIndex + 2] & 0xFF;
            if (modelConfig.stdConfig != null) {
                inputData[i] = (r / 255.0f - modelConfig.stdConfig.mean) / modelConfig.stdConfig.std;
                inputData[i + size] = (g / 255.0f - modelConfig.stdConfig.mean) / modelConfig.stdConfig.std;
                inputData[i + size * 2] = (b / 255.0f - modelConfig.stdConfig.mean) / modelConfig.stdConfig.std;
            } else {
                inputData[i] = b - modelConfig.rfConfig.mean_b;
                inputData[i + size] = g - modelConfig.rfConfig.mean_g;
                inputData[i + size * 2] = r - modelConfig.rfConfig.mean_r;
            }
        }
        return inputData;
    }

    // 模型推理的全过程
    protected ResultType inference(byte[] rgbData) throws OrtException {
        synchronized (lock) {
            float[] inputData = normalize(rgbData);
            // 更新输入缓冲区数据
            inputByteBuffer.rewind();
            inputFloatBuffer.rewind();
            inputFloatBuffer.put(inputData);
            try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputByteBuffer, inputShape, OnnxJavaType.FLOAT)) {
                OrtSession.Result result = session.run(Collections.singletonMap(session.getInputInfo().keySet().iterator().next(), inputTensor));
                return postprocess(result);
            }
        }
    }

}
