package com.example.model.core;

import com.example.model.core.base.OnnxDeployer;
import ai.onnxruntime.*;

public class PlateRecognizer extends OnnxDeployer<PlateRecognizer.Result> {

    public static class Result {
        public String number;
        public String color;
    }

    private static final String TAG = "MyLogcat-PlateRecognizer";

    // 模型配置
    public static final String MODEL_NAME = "car_rec.onnx";
    public static final int MODEL_WIDTH = 168;
    public static final int MODEL_HEIGHT = 48;
    private static final float MEAN_VALUE = 0.588f;
    private static final float STD_VALUE = 0.193f;

    // 字符集
    private static final String[] COLOR_LIST = {"黑色", "蓝色", "绿色", "白色", "黄色"};
    private static final String NAME_LIST = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品";

    public PlateRecognizer(Logger logger, byte[] modelData) {
        super(logger, new Model(modelData, MODEL_WIDTH, MODEL_HEIGHT, MEAN_VALUE, STD_VALUE));
    }

    // 运行
    public String run(byte[] rgbData) {
        Result result = super.inference(rgbData);
        String number = result.number;
        logger.info(TAG, "plate number recognized: " + number);
        return number;
    }

    // 后处理
    @Override
    protected Result postprocess(OrtSession.Result sessionResult) {
        Result result = new Result();
        try (OnnxTensor plateTensor = (OnnxTensor) sessionResult.get(0);
             OnnxTensor colorTensor = (OnnxTensor) sessionResult.get(1)) {
            result.number = decodePlate((float[][][]) plateTensor.getValue());
            result.color = decodeColor(((float[][]) colorTensor.getValue())[0]);
            return result;
        } catch (OrtException e) {
            logger.error(TAG, "update output tensor failed: " + e.getMessage());
            return null;
        }
    }

    // 解码车牌号码
    private static String decodePlate(float[][][] plateOutput) {
        StringBuilder result = new StringBuilder();
        float[][] floatArrays = plateOutput[0];
        int pre = 0;
        for (float[] floatArray : floatArrays) {
            int maxIndex = 0;
            float maxValue = floatArray[0];
            for (int j = 0; j < floatArrays[0].length; j++) {
                if (floatArray[j] > maxValue) {
                    maxValue = floatArray[j];
                    maxIndex = j;
                }
            }
            if (maxIndex > 0 && maxIndex <= NAME_LIST.length() && maxIndex != pre) {
                result.append(NAME_LIST.charAt(maxIndex));
            }
            pre = maxIndex;
        }
        return result.toString();
    }

    // 解码车牌颜色
    private static String decodeColor(float[] colorOutput) {
        int maxIndex = 0;
        float maxValue = colorOutput[0];
        for (int i = 1; i < colorOutput.length; i++) {
            if (colorOutput[i] > maxValue) {
                maxValue = colorOutput[i];
                maxIndex = i;
            }
        }
        return COLOR_LIST[maxIndex];
    }

}