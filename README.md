# Android ONNXRuntime 部署说明

## 一、代码结构

```text
model
├── api
│   └── android
│       ├── ImageProcesser.java - 图像处理
│       ├── Logger.java         - 日志输出
│       └── ModelLoader.java    - 模型加载
├── core
│   ├── base
│   │   └── OnnxDeployer.java   - 基类
│   ├── FacePlateDetector.java  - 车牌+人脸检测
│   ├── FaceRecognizer.java     - 人脸识别
│   └── PlateRecognizer.java    - 车牌识别
└── README.md
```

## 二、使用步骤

### 1.将本仓库初始化为你的Android项目的submodule

```sh
cd project/app/src/main/java/com/example
git submodule add git@github.com:jbn-cn-product/model-deployment-onnx.git model
git submodule update --init
```

### 2.配置gradle依赖

```kts
dependencies {
    implementation(libs.onnxruntime.android)
}
```

```toml
[versions]
onnxruntime = "1.23.1"

[libraries]
onnxruntime-android = { group = "com.microsoft.onnxruntime", name = "onnxruntime-android", version.ref = "onnxruntime" }
```

或者包名+版本号合并声明

```kts
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.1")
}
```

### 3.模型文件放置于assets目录

模型文件路径(Linux格式): smb://192.168.2.28/产研中心/算法/faceplate/model-onnx/

```text
app/src/main
├── AndroidManifest.xml
└── assets
    ├── car_face_det.onnx
    ├── car_rec.onnx
    └── face_rec.onnx
```

### 4.部署

```java
// 以android端人脸识别为例
import com.example.model.api.android.ImageProcesser;
import com.example.model.api.android.Logger;
import com.example.model.api.android.ModelLoader;
import com.example.model.core.FaceRecognizer;
public class Recognizer implements AutoCloseable {
    // 模型实例
    private final FaceRecognizer faceRecognizer;

    // 初始化
    public Recognizer(Context context) {
        // 用api.android.Logger类作为日志输出的实现
        Logger logger = new Logger();
        // 用api.android.ModelLoader加载模型
        byte[] modelData = ModelLoader.getModelData(context, "face_rec.onnx");
        // 创建实例
        faceRecognizer = new FaceRecognizer(logger, modelData);
    }

    // 释放资源
    @Override
    public void close() {
        faceRecognizer.close();
    }

    // 检测
    private void run(Bitmap bitmap) {
        // 缩放到目标尺寸
        Bitmap resizedBitmap = ImageProcesser.resizeBitmap(bitmap, FaceRecognizer.MODEL_WIDTH, FaceRecognizer.MODEL_HEIGHT, false);
        // 转换成RGB数组
        byte[] rgbData = ImageProcesser.convertBitmapToRGB(resizedBitmap);
        resizedBitmap.recycle(); // 记得及时释放bitmap资源
        // 运行
        float[] features = faceRecognizer.run(rgbData);
        Log.d(TAG, Arrays.toString(features));
    }
}
```

## 三、注意事项

- 模型推理时间较长，要在独立的线程中运行run方法，避免阻塞主线程
