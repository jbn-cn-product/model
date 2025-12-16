package com.example.model;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import com.example.model.base.OnnxDeployer;
import com.example.model.util.ModelImageHelper;
import ai.onnxruntime.*;

public class PlateRecognizer extends OnnxDeployer<PlateRecognizer.Result> {

    public static class Result {
        public String number;
        public String color;
    }

    private static final String TAG = "MyLogcat-PlateRecognizer";

    // 文件
    private static final String MODEL_NAME = "car_rec.onnx";

    // 模型参数
    private static final int INPUT_WIDTH = 168;
    private static final int INPUT_HEIGHT = 48;
    private static final float MEAN_VALUE = 0.588f; // 均值
    private static final float STD_VALUE = 0.193f;  // 标准差

    // 字符集
    private static final String[] COLOR_LIST = {"黑色", "蓝色", "绿色", "白色", "黄色"};
    private static final String NAME_LIST = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品";

    // 检测框，用于预处理时裁剪图像
    private int[] bbox;

    public PlateRecognizer(Context context) {
        super(context, MODEL_NAME, INPUT_WIDTH, INPUT_HEIGHT);
    }

    // 运行
    public String inference(Bitmap originalBitmap, int[] bbox) {
        this.bbox = bbox;
        Result result = super.inference(originalBitmap);
        String number = result.number;
        Log.i(TAG, "plate number recognized: " + number);
        return number;
    }

    // 预处理
    @Override
    protected float[] preprocess(Bitmap originalBitmap) {
        // 裁减
        Bitmap cutBitmap = ModelImageHelper.cutBitmapByBox(originalBitmap, bbox, 0);
        // 拉伸缩放
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(cutBitmap, INPUT_WIDTH, INPUT_HEIGHT, true);
        cutBitmap.recycle();
        // 标准归一化
        float[] inputData = ModelImageHelper.normalizeBitmap(resizedBitmap, INPUT_WIDTH, INPUT_HEIGHT, MEAN_VALUE, STD_VALUE);
        resizedBitmap.recycle();
        return inputData;
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
            Log.e(TAG, "update output tensor failed");
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