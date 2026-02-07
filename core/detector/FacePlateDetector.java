package com.example.model.core.detector;

import com.example.model.core.OnnxDeployer;
import com.example.model.structure.Common.Box;
import com.example.model.structure.Common.Point;
import com.example.model.structure.Face;
import com.example.model.structure.Plate;
import com.example.model.utils.DataHelper;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

public class FacePlateDetector extends OnnxDeployer<List<FacePlateDetector.Result>> {

    public static class Result {
        public Box box;                         // 检测框
        public float confidence;                // 置信度
        public int classId;                     // 类别 0-车牌 1-人脸
        public Face.Angles faceAngles;          // 人脸姿态
        public Face.Landmarks faceLandmarks;    // 人脸关键点
        public Plate.Vertexes plateVertexes;    // 车牌顶点
    }

    private static final String TAG = "MyLogcat-FacePlateDetector";

    // 模型配置
    public static final String MODEL_NAME = "car_face_det.onnx";
    public static final int MODEL_WIDTH = 640;
    public static final int MODEL_HEIGHT = 640;
    private static final float CONF_THRESHOLD = 0.6f;
    private static final float IOU_THRESHOLD = 0.5f;
    private static final float MEAN_VALUE = 0.0f;
    private static final float STD_VALUE = 1.5f;

    public FacePlateDetector(Logger logger, byte[] modelData) {
        super(logger, new ModelConfig(modelData, MODEL_WIDTH, MODEL_HEIGHT, MEAN_VALUE, STD_VALUE));
    }

    // 运行
    public List<Result> run(byte[] rgbData) {
        long startTime = System.currentTimeMillis();
        try {
            List<Result> results = super.inference(rgbData);
            if (!results.isEmpty()) {
                logger.debug(TAG, String.format("检测到%d个目标, 用时%dms", System.currentTimeMillis() - startTime, results.size()));
            }
            return results;
        } catch (OrtException e) {
            return new ArrayList<>();
        }
    }

    // 后处理
    @Override
    protected List<Result> postprocess(OrtSession.Result sessionResult) throws OrtException {
        try (OnnxTensor tensor = (OnnxTensor) sessionResult.get(0)) {
            TensorInfo tensorInfo = (TensorInfo) super.session.getOutputInfo().values().iterator().next().getInfo();
            FloatBuffer buffer = tensor.getFloatBuffer();
            long[] outputShape = tensorInfo.getShape();
            float[] outputArray = new float[buffer.remaining()];
            buffer.get(outputArray);
            int featuresPerDetection = (int) outputShape[2];
            int totalDetections = (int) outputShape[0] * (int) outputShape[1];
            float[][] outputs = new float[totalDetections][featuresPerDetection];
            for (int i = 0; i < totalDetections; i++) {
                System.arraycopy(outputArray, i * featuresPerDetection, outputs[i], 0, featuresPerDetection);
            }
            return decodeResult(outputs);
        }
    }

    // 解码检测结果
    private List<Result> decodeResult(float[][] outputs) {
        List<Result> results = new ArrayList<>();
        // 解析输出结构
        for (float[] output : outputs) {
            float confidence = output[4];
            if (confidence < CONF_THRESHOLD) { // 先忽略大多数低置信度结果
                continue;
            }
            Result result = new Result();
            // 置信度
            result.confidence = confidence;
            float x = output[0], y = output[1], w = output[2], h = output[3];
            // 检测框
            result.box = new Box(new Point(x - w / 2, y - h / 2), w, h); // 原始数据的xy坐标是检测框的中心点，需要修正为左上角
            // 类别
            float plateScore = output[15];
            float faceScore = output[16];
            result.classId = plateScore > faceScore ? 0 : 1;
            // 关键点
            List<Point> pointList = new ArrayList<>(5);
            for (int i = 5; i < 14; i += 2) {
                pointList.add(new Point(output[i], output[i + 1]));
            }
            if (result.classId == 0) {
                result.plateVertexes = new Plate.Vertexes();
                List<Point> vertexes = new ArrayList<>(pointList);
                vertexes.remove(2); // 第3个关键点在车牌中无意义
                // 排序后判断坐标方位
                vertexes.sort((a, b) -> Float.compare(a.x, b.x));
                result.plateVertexes.lt = vertexes.get(0).y < vertexes.get(1).y ? vertexes.get(0) : vertexes.get(1);
                result.plateVertexes.rt = vertexes.get(2).y < vertexes.get(3).y ? vertexes.get(2) : vertexes.get(3);
                result.plateVertexes.rb = vertexes.get(2).y >= vertexes.get(3).y ? vertexes.get(2) : vertexes.get(3);
                result.plateVertexes.lb = vertexes.get(0).y >= vertexes.get(1).y ? vertexes.get(0) : vertexes.get(1);
            } else if (result.classId == 1) {
                result.faceLandmarks = new Face.Landmarks(pointList.get(0), pointList.get(1), pointList.get(2), pointList.get(3), pointList.get(4));
                result.faceAngles = DataHelper.calculateFaceAngles(result.faceLandmarks);
            }
            results.add(result);
        }
        // NMS
        results.sort((a, b) -> Float.compare(b.confidence, a.confidence));
        List<Box> boxList = new ArrayList<>();
        for (Result result : results) {
            boxList.add(result.box);
        }
        DataHelper.nms(results, boxList, IOU_THRESHOLD);
        return results;
    }

}
