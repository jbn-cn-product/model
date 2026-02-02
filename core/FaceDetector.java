package com.example.model.core;

import com.example.model.core.base.OnnxDeployer;
import com.example.model.structure.Common;
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
    private static final float CONF_THRESHOLD = 0.9f;
    private static final float IOU_THRESHOLD = 0.7f;

    // RetinaFace参数
    private static final int[][] MIN_SIZES = {{16, 32}, {64, 128}, {256, 512}};
    private static final int[] STEPS = {8, 16, 32};
    private static final float[] VARIANCE = {0.1f, 0.2f};
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

    public List<Result> run(byte[] rgbData) {
        long startTime = System.currentTimeMillis();
        List<Result> results = super.inference(rgbData);
        if (!results.isEmpty()) {
            logger.debug(TAG, String.format("run detection in %d ms, %d faces", System.currentTimeMillis() - startTime, results.size()));
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
            inputData[i] = b - 104.0f;
            inputData[i + size] = g - 117.0f;
            inputData[i + 2 * size] = r - 123.0f;
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
            Result result = new Result();
            result.confidence = faceScore;
            float[] anchor = anchors.get(i);
            float cx = anchor[0] + boxData[i*4] * VARIANCE[0] * anchor[2];
            float cy = anchor[1] + boxData[i*4+1] * VARIANCE[0] * anchor[3];
            float w = anchor[2] * (float) Math.exp(boxData[i*4+2] * VARIANCE[1]);
            float h = anchor[3] * (float) Math.exp(boxData[i*4+3] * VARIANCE[1]);
            float lx1 = (anchor[0] + landmarksData[i*10 + 0] * VARIANCE[0] * anchor[2]) * MODEL_WIDTH;
            float lx2 = (anchor[0] + landmarksData[i*10 + 2] * VARIANCE[0] * anchor[2]) * MODEL_WIDTH;
            float lx3 = (anchor[0] + landmarksData[i*10 + 4] * VARIANCE[0] * anchor[2]) * MODEL_WIDTH;
            float lx4 = (anchor[0] + landmarksData[i*10 + 6] * VARIANCE[0] * anchor[2]) * MODEL_WIDTH;
            float lx5 = (anchor[0] + landmarksData[i*10 + 8] * VARIANCE[0] * anchor[2]) * MODEL_WIDTH;
            float ly1 = (anchor[1] + landmarksData[i*10 + 1] * VARIANCE[0] * anchor[3]) * MODEL_HEIGHT;
            float ly2 = (anchor[1] + landmarksData[i*10 + 3] * VARIANCE[0] * anchor[3]) * MODEL_HEIGHT;
            float ly3 = (anchor[1] + landmarksData[i*10 + 5] * VARIANCE[0] * anchor[3]) * MODEL_HEIGHT;
            float ly4 = (anchor[1] + landmarksData[i*10 + 7] * VARIANCE[0] * anchor[3]) * MODEL_HEIGHT;
            float ly5 = (anchor[1] + landmarksData[i*10 + 9] * VARIANCE[0] * anchor[3]) * MODEL_HEIGHT;
            int x1 = (int) ((cx - w/2) * MODEL_WIDTH);
            int y1 = (int) ((cy - h/2) * MODEL_HEIGHT);
            int x2 = (int) ((cx + w/2) * MODEL_WIDTH);
            int y2 = (int) ((cy + h/2) * MODEL_HEIGHT);
            result.box = new Box(new Point(x1, y1), x2, y2);
            result.faceLandmarks = new Face.Landmarks(
                    new Point((int) lx1, (int) ly1),
                    new Point((int) lx2, (int) ly2),
                    new Point((int) lx3, (int) ly3),
                    new Point((int) lx4, (int) ly4),
                    new Point((int) lx5, (int) ly5)
            );
            result.faceAngles = DataHelper.calculateAngles(result.faceLandmarks);
            results.add(result);
        }
        // NMS
        results.sort((a, b) -> Float.compare(b.confidence, a.confidence));
        List<Common.Box> boxList = new ArrayList<>();
        for (Result result : results) {
            boxList.add(result.box);
        }
        DataHelper.nms(results, boxList, IOU_THRESHOLD);
        return results;
    }

}
