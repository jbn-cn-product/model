package com.example.model.core.recognizer;

import com.example.model.core.OnnxDeployer;
import ai.onnxruntime.*;
import java.nio.FloatBuffer;

public class FaceRecognizer extends OnnxDeployer<FaceRecognizer.Result> {

    public static class Result {
        public float[] features;
    }

    private static final String TAG = "MyLogcat-FaceRecognizer";

    // 模型配置
    public static final String MODEL_NAME = "face_rec.onnx";
    public static final int MODEL_WIDTH = 112;
    public static final int MODEL_HEIGHT = 112;
    private static final float MEAN_VALUE = 0.5f;
    private static final float STD_VALUE = 0.5f;

    public FaceRecognizer(Logger logger, byte[] modelData) {
        super(logger, new ModelConfig(modelData, MODEL_WIDTH, MODEL_HEIGHT, MEAN_VALUE, STD_VALUE));
    }

    // 运行
    public float[] run(byte[] rgbData) {
        try {
            Result result = super.inference(rgbData);
            return result.features;
        } catch (OrtException e) {
            return null;
        }
    }

    // 后处理
    @Override
    protected Result postprocess(OrtSession.Result sessionResult) {
        try (OnnxTensor tensor = (OnnxTensor) sessionResult.get(0)) {
            FloatBuffer buffer = tensor.getFloatBuffer();
            float[] outputs = new float[buffer.remaining()];
            buffer.get(outputs);
            float norm = 0.0f;
            for (float value : outputs) {
                norm += value * value;
            }
            norm = (float) Math.sqrt(norm + 1e-8f); // 添加小值防止除零
            Result result = new Result();
            result.features = new float[outputs.length];
            for (int i = 0; i < outputs.length; i++) {
                result.features[i] = outputs[i] / norm;
            }
            return result;
        }
    }

    // 计算两个特征向量的余弦相似度
    private static float calculateSimilarity(float[] features1, float[] features2) {
        if (features1.length != features2.length) {
            return 0.0f;
        }
        float dotProduct = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        for (int i = 0; i < features1.length; i++) {
            dotProduct += features1[i] * features2[i];
            norm1 += features1[i] * features1[i];
            norm2 += features2[i] * features2[i];
        }
        norm1 = (float) Math.sqrt(norm1);
        norm2 = (float) Math.sqrt(norm2);
        if (norm1 == 0f || norm2 == 0f) {
            return 0.0f;
        }
        return dotProduct / (norm1 * norm2);
    }

}