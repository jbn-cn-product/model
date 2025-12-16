package com.example.model;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.util.Log;
import com.example.model.base.OnnxDeployer;
import com.example.model.util.ModelImageHelper;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

public class FacePlateDetector extends OnnxDeployer<List<FacePlateDetector.Result>> {

    public static class Result {
        public int[] bbox;
        public float threshold;
        public List<float[]> landmarks;
        public int classId;
    }

    private static final String TAG = "MyLogcat-FacePlateDetector";

    // 文件
    private static final String MODEL_NAME = "car_face_det.onnx";

    // 模型参数
    private static final int INPUT_WIDTH = 640;
    private static final int INPUT_HEIGHT = 640;
    private static final float CONF_THRESHOLD = 0.3f;   // 置信度阈值
    private static final float IOU_THRESHOLD = 0.5f;    // 重合阈值

    // 输入图像尺寸
    private int previewWidth;
    private int previewHeight;

    public FacePlateDetector(Context context) {
        super(context, MODEL_NAME, INPUT_WIDTH, INPUT_HEIGHT);
    }

    // 运行
    public List<Result> inference(Bitmap originalBitmap) {
        this.previewWidth = originalBitmap.getWidth();
        this.previewHeight = originalBitmap.getHeight();
        long startTime = System.currentTimeMillis();
        List<Result> results = super.inference(originalBitmap);
        Log.d(TAG, String.format("run detection in %d ms, %d objects", System.currentTimeMillis() - startTime, results.size()));
        return results;
    }

    // 预处理
    @Override
    protected float[] preprocess(Bitmap originalBitmap) {
        // 保持比例居中缩放
        Bitmap resizedBitmap = Bitmap.createBitmap(INPUT_WIDTH, INPUT_HEIGHT, Bitmap.Config.RGB_565);
        // Bitmap自带的缩放API产生的图像会变形，导致关键点错位，需要保持原始图像比例
        float scale = Math.min((float) INPUT_WIDTH / originalBitmap.getWidth(), (float) INPUT_HEIGHT / originalBitmap.getHeight());
        int newWidth = Math.round(originalBitmap.getWidth() * scale);
        int newHeight = Math.round(originalBitmap.getHeight() * scale);
        Canvas canvas = new Canvas(resizedBitmap);
        Matrix matrix = new Matrix();
        matrix.postScale(scale, scale);
        float left = (INPUT_WIDTH - newWidth) / 2.0f;
        float top = (INPUT_HEIGHT - newHeight) / 2.0f;
        matrix.postTranslate(left, top);
        canvas.drawBitmap(originalBitmap, matrix, null);
        // 简单归一化
        float[] inputData = ModelImageHelper.normalizeBitmap(resizedBitmap, INPUT_WIDTH, INPUT_HEIGHT, 0.0f, 1.0f);
        resizedBitmap.recycle();
        return inputData;
    }

    // 后处理
    @Override
    protected List<Result> postprocess(OrtSession.Result sessionResult) {
        // 处理输出张量
        long[] outputShape;
        FloatBuffer buffer;
        try (OnnxTensor tensor = (OnnxTensor) sessionResult.get(0)) {
            TensorInfo tensorInfo = (TensorInfo) super.session.getOutputInfo().values().iterator().next().getInfo();
            buffer = tensor.getFloatBuffer();
            outputShape = tensorInfo.getShape();
        } catch (OrtException e) {
            Log.e(TAG, "update output tensor failed");
            return Collections.emptyList();
        }
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

    // 解码检测结果
    private List<Result> decodeResult(float[][] outputs) {
        List<Result> results = new ArrayList<>();
        List<float[]> bboxList = new ArrayList<>();
        List<Float> thresholdList = new ArrayList<>();
        List<Integer> classIdList = new ArrayList<>();
        List<float[]> landmarksList = new ArrayList<>();
        float scale = Math.min((float) INPUT_WIDTH / previewWidth, (float) INPUT_HEIGHT / previewHeight);
        int padX = (INPUT_WIDTH - (int)(previewWidth * scale)) / 2;
        int padY = (INPUT_HEIGHT - (int)(previewHeight * scale)) / 2;
        // 解析输出结构
        for (float[] output : outputs) {
            float confidence = output[4];
            float plateScore = output[15];
            float faceScore = output[16];
            float threshold = confidence * Math.max(plateScore, faceScore); // 输出置信度*最高类别分数=最终置信度
            if (threshold < CONF_THRESHOLD) {
                continue;
            }
            thresholdList.add(threshold);
            classIdList.add(plateScore > faceScore ? 0 : 1);
            // 输出的检测框和特征点坐标缩放到原图比例
            float x = output[0], y = output[1], w = output[2], h = output[3];
            float[] bbox = new float[4];
            bbox[0] = Math.max(0, Math.min(previewWidth, (x - w / 2 - padX) / scale));
            bbox[1] = Math.max(0, Math.min(previewHeight, (y - h / 2 - padY) / scale));
            bbox[2] = Math.max(0, Math.min(previewWidth, (x + w / 2 - padX) / scale));
            bbox[3] = Math.max(0, Math.min(previewHeight, (y + h / 2 - padY) / scale));
            bboxList.add(bbox);
            float[] landmarks = new float[10];
            for (int i = 0; i < 10; i += 2) {
                landmarks[i] = Math.max(0, Math.min(previewWidth, (output[5 + i] - padX) / scale));
                landmarks[i + 1] = Math.max(0, Math.min(previewHeight, (output[6 + i] - padY) / scale));
            }
            landmarksList.add(landmarks);
        }
        // NMS过滤多个重复检测框
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < thresholdList.size(); i++) {
            order.add(i);
        }
        order.sort((a, b) -> Float.compare(thresholdList.get(b), thresholdList.get(a)));
        List<Integer> keepIndexes = new ArrayList<>();
        while (!order.isEmpty()) {
            int current = order.remove(0);
            keepIndexes.add(current);
            Iterator<Integer> it = order.iterator();
            while (it.hasNext()) {
                float[] bbox1 = bboxList.get(current);
                float[] bbox2 = bboxList.get(it.next());
                float x1 = Math.max(bbox1[0], bbox2[0]);
                float x2 = Math.min(bbox1[2], bbox2[2]);
                float y1 = Math.max(bbox1[1], bbox2[1]);
                float y2 = Math.min(bbox1[3], bbox2[3]);
                float interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
                if (interArea > 0) {
                    float iou = interArea / ((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - interArea);
                    if (iou > IOU_THRESHOLD) {
                        it.remove();
                    }
                }
            }
        }
        // 创建结果
        for (int keepIndex : keepIndexes) {
            float[] bbox = bboxList.get(keepIndex);
            float[] landmarks = landmarksList.get(keepIndex);
            List<float[]> landmarkPoints = new ArrayList<>();
            for (int i = 0; i < 5; i++) {
                landmarkPoints.add(new float[]{landmarks[i * 2], landmarks[i * 2 + 1]});
            }
            Result result = new Result();
            result.bbox = new int[]{(int) bbox[0], (int) bbox[1], (int) bbox[2], (int) bbox[3]};
            result.threshold = thresholdList.get(keepIndex);
            result.landmarks = landmarkPoints;
            result.classId = classIdList.get(keepIndex);
            results.add(result);
        }
        return results;
    }

}
