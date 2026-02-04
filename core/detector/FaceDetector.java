package com.example.model.core.detector;

import com.example.model.core.OnnxDeployer;
import com.example.model.structure.Common.Box;
import com.example.model.structure.Common.Point;
import com.example.model.structure.Face;
import com.example.model.utils.DataHelper;
import java.util.ArrayList;
import java.util.List;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtSession;

public class FaceDetector extends OnnxDeployer<List<FaceDetector.Result>> {

    public static class Result {
        public Box box;
        public float confidence;
        public Face.Landmarks faceLandmarks;
        public Face.Angles faceAngles;
    }

    private static final String TAG = "MyLogcat-FaceDetector";

    // 模型配置
    public static final String MODEL_NAME = "face_det.onnx";
    public static final int MODEL_WIDTH = 640;
    public static final int MODEL_HEIGHT = 640;
    private static final float CONF_THRESHOLD = 0.6f;
    private static final float IOU_THRESHOLD = 0.5f;

    // RetinaFace参数
    private static final int[][] MIN_SIZES = {{16, 32}, {64, 128}, {256, 512}};
    private static final int[] STEPS = {8, 16, 32};
    private static final float[] VARIANCE = {0.1f, 0.2f};
    private static final float MEAN_R = 123.0f;
    private static final float MEAN_G = 117.0f;
    private static final float MEAN_B = 104.0f;
    private final List<float[]> anchors = new ArrayList<>();

    public FaceDetector(Logger logger, byte[] modelData) {
        super(logger, new Model(modelData, MODEL_WIDTH, MODEL_HEIGHT, 0.0f, 1.0f));
        for (int i = 0; i < STEPS.length; i++) {
            for (int j = 0; j < (int) Math.ceil((float) MODEL_HEIGHT / STEPS[i]); j++) {
                for (int k = 0; k < (int) Math.ceil((float) MODEL_WIDTH / STEPS[i]); k++) {
                    for (int minSize : MIN_SIZES[i]) {
                        anchors.add(new float[]{
                                (k + 0.5f) * STEPS[i] / MODEL_WIDTH,
                                (j + 0.5f) * STEPS[i] / MODEL_HEIGHT,
                                minSize / (float) MODEL_WIDTH,
                                minSize / (float) MODEL_HEIGHT
                        });
                    }
                }
            }
        }
    }

    // 运行
    public List<Result> run(byte[] rgbData) {
        long startTime = System.currentTimeMillis();
        List<Result> results = super.inference(rgbData);
        if (!results.isEmpty()) {
            logger.debug(TAG, String.format("检测到%d张人脸, 用时%dms", results.size(), System.currentTimeMillis() - startTime));
        }
        return results;
    }

    // RetinaFace的归一化
    @Override
    protected float[] normalize(byte[] rgbData) {
        int size = MODEL_WIDTH * MODEL_HEIGHT;
        float[] inputData = new float[size * 3];
        for (int i = 0; i < size; i++) {
            int pixelIndex = i * 3;
            float r = rgbData[pixelIndex] & 0xFF;
            float g = rgbData[pixelIndex + 1] & 0xFF;
            float b = rgbData[pixelIndex + 2] & 0xFF;
            inputData[i] = b - MEAN_B;
            inputData[i + size] = g - MEAN_G;
            inputData[i + 2 * size] = r - MEAN_R;
        }
        return inputData;
    }

    // 后处理
    @Override
    protected List<Result> postprocess(OrtSession.Result sessionResult) {
        try (OnnxTensor boxTensor = (OnnxTensor) sessionResult.get(0);
             OnnxTensor confidenceTensor = (OnnxTensor) sessionResult.get(1);
             OnnxTensor landmarksTensor = (OnnxTensor) sessionResult.get(2)) {
            float[] boxData = new float[boxTensor.getFloatBuffer().remaining()];
            boxTensor.getFloatBuffer().get(boxData);
            float[] confidenceData = new float[confidenceTensor.getFloatBuffer().remaining()];
            confidenceTensor.getFloatBuffer().get(confidenceData);
            float[] landmarksData = new float[landmarksTensor.getFloatBuffer().remaining()];
            landmarksTensor.getFloatBuffer().get(landmarksData);
            return decodeResult(boxData, confidenceData, landmarksData);
        } catch (Exception e) {
            logger.error(TAG, "update output tensor failed: " + e.getMessage());
            return null;
        }
    }

    // 解码检测结果
    private List<Result> decodeResult(float[] boxData, float[] confidenceData, float[] landmarksData) {
        List<Result> results = new ArrayList<>();
        for (int i = 0; i < anchors.size(); i++) {
            float faceScore = confidenceData[i * 2 + 1];
            if (faceScore < CONF_THRESHOLD) {
                continue;
            }
            float ax = anchors.get(i)[0];
            float ay = anchors.get(i)[1];
            float aw = anchors.get(i)[2];
            float ah = anchors.get(i)[3];
            float cx = ax + aw * boxData[i * 4] * VARIANCE[0];
            float cy = ay + ah * boxData[i * 4 + 1] * VARIANCE[0];
            float w = aw * (float) Math.exp(boxData[i * 4 + 2] * VARIANCE[1]);
            float h = ah * (float) Math.exp(boxData[i * 4 + 3] * VARIANCE[1]);
            int offset = i * 10;
            Point[] points = new Point[5];
            for (int j = 0; j < 10; j += 2) {
                float lx = (ax + landmarksData[offset + j] * VARIANCE[0] * aw) * MODEL_WIDTH;
                float ly = (ay + landmarksData[offset + j + 1] * VARIANCE[0] * ah) * MODEL_HEIGHT;
                points[j / 2] = new Point(lx, ly);
            }
            Result result = new Result();
            result.confidence = faceScore;
            result.box = new Box(new Point((cx - w / 2) * MODEL_WIDTH, (cy - h / 2) * MODEL_HEIGHT), w * MODEL_WIDTH, h * MODEL_HEIGHT);
            result.faceLandmarks = new Face.Landmarks(points[0], points[1], points[2], points[3], points[4]);
            result.faceAngles = DataHelper.calculateAngles(result.faceLandmarks);
            results.add(result);
        }
        results.sort((a, b) -> Float.compare(b.confidence, a.confidence));
        List<Box> boxList = new ArrayList<>();
        for (Result result : results) {
            boxList.add(result.box);
        }
        DataHelper.nms(results, boxList, IOU_THRESHOLD);
        return results;
    }

}
