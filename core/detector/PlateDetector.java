package com.example.model.core.detector;

import com.example.model.core.OnnxDeployer;
import com.example.model.structure.Common.Box;
import com.example.model.structure.Common.Point;
import com.example.model.utils.DataHelper;
import java.util.ArrayList;
import java.util.List;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtSession;

public class PlateDetector extends OnnxDeployer<List<PlateDetector.Result>> {

    public static class Result {
        public Box box;
        public float confidence;
    }

    private static final String TAG = "MyLogcat-PlateDetector";

    // 模型配置
    public static final String MODEL_NAME = "car_det.onnx";
    public static final int MODEL_WIDTH = 640;
    public static final int MODEL_HEIGHT = 640;
    private static final float CONF_THRESHOLD = 0.6f;
    private static final float IOU_THRESHOLD = 0.5f;

    public PlateDetector(Logger logger, byte[] modelData) {
        super(logger, new Model(modelData, MODEL_WIDTH, MODEL_HEIGHT, 0.5f, 0.5f));
    }

    // 运行
    public List<Result> run(byte[] rgbData) {
        long startTime = System.currentTimeMillis();
        List<Result> results = super.inference(rgbData);
        if (!results.isEmpty()) {
            logger.debug(TAG, String.format("run detection in %d ms, %d plates", System.currentTimeMillis() - startTime, results.size()));
        }
        return results;
    }

    // 后处理
    @Override
    protected List<Result> postprocess(OrtSession.Result sessionResult) {
        try (OnnxTensor outputTensor = (OnnxTensor) sessionResult.get(0)) {
            float[] data = new float[outputTensor.getFloatBuffer().remaining()];
            outputTensor.getFloatBuffer().get(data);
            return decodeResults(data);
        }
    }

    // 解码检测结果
    private List<Result> decodeResults(float[] data) {
        List<Result> results = new ArrayList<>();
        int numClasses = 2;
        int stride = 4 + 1 + 8 + numClasses;
        for (int i = 0; i < data.length; i += stride) {
            float objConf = data[i + 4];
            if (objConf < CONF_THRESHOLD) {
                continue;
            }
            float maxScore = 0;
            for (int c = 0; c < numClasses; c++) {
                float score = objConf * data[i + 13 + c];
                if (score > maxScore) {
                    maxScore = score;
                }
            }
            if (maxScore < CONF_THRESHOLD) {
                continue;
            }
            float cx = data[i];
            float cy = data[i + 1];
            float w = data[i + 2];
            float h = data[i + 3];
            Result result = new Result();
            result.confidence = maxScore;
            result.box = new Box(new Point(cx - w / 2, cy - h / 2), w, h);
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
