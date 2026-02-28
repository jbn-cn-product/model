# Android ONNXRuntime 部署说明

## 一、代码结构

```text
model
├── core
│   ├── base
│   │   └── OnnxDeployer.java
│   ├── FacePlateDetector.java
│   ├── FaceRecognizer.java
│   └── PlateRecognizer.java
├── api
│   └── android
│       └── Logger.java
└── utils
    └── android
        ├── ImageProcesser.java
        └── ModelLoader.java
```

## 二、模块设计思路

- 模型定义了一个Logger接口，部署过程需要输出日志以供调试，不同运行环境下打印日志的方法也不同，例如纯java可以用System.out、Android用Log类，api/android为Android环境的默认实现

- 由于编写本库的目的是实现脱离Android环境依赖进行onnx模型推理部署，因此规定了core里的代码不允许导入android包，但这样只能将模型的图像输入格式定义成byte[]，而非Android的Bitmap类，而图像需要先经过缩放、裁剪等操作处理成模型接受的状态，再转换为字节数组，另外模型文件也需要处理成byte[]，因此创建utils/android，提供了一套辅助方法，不同的Android项目可共享使用。若是在电脑上进行纯java部署，就需要调用方自行编写相关功能

- 以下为本仓库三种不同模型要求的图像输入规格:

| 类名 | 模型文件 | 功能 | 分辨率 | 内容要求 |
| --- | --- | :---: | :---: | --- |
| FacePlateDetector | car_face_det.onnx | 车牌+人脸多目标检测 | 640x640 | 需要保持图像原始比例，以获取正常的关键点分布(可以在转换尺寸后填充黑色区域) |
| FaceRecognizer | face_rec.onnx | 人脸识别 | 112x112 | 画面内容只能有人脸，不能出现周围环境，否则会导致特征向量匹配度很低 |
| PlateRecognizer | car_rec.onnx | 车牌识别 | 168x48 | 同人脸识别，以牌照框为整张图片的边界，来提高车牌号识别准确度 |

## 三、使用步骤(以Android为例)

### 1.将本仓库初始化为你的项目的子模块

```sh
cd project/app/src/main/java/com/example
git submodule add git@github.com:jbn-cn-product/model-deployment-onnx.git model
git submodule update --init
```

### 2.配置gradle依赖

在build.gradle.kts中声明包名

```kts
dependencies {
    implementation(libs.onnxruntime.android)
}
```

在libs.versions.toml中配置版本

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

目前用的包是适配android的，但API完全通用

### 3.模型文件放置于assets目录

模型文件路径(Linux格式): smb://192.168.2.28/产研中心/算法/faceplate/model-onnx/

```text
app/src/main
└── assets
    ├── car_face_det.onnx
    ├── car_rec.onnx
    └── face_rec.onnx
```

### 4.部署

以下是简单示例，详细使用过程可参考执法仪应用[Detector.java](https://github.com/jbn-cn-product/vision-bodycam/blob/bodycam2/app/src/main/java/com/example/bodycam2/Detector.java)

```java
import com.example.model.onnx.api.android.Logger;
import com.example.model.onnx.core.FaceRecognizer;
import com.example.model.onnx.utils.android.ImageProcesser;
import com.example.model.onnx.utils.android.ModelLoader;
public class Recognizer implements AutoCloseable {

    // 人脸识别
    private final FaceRecognizer faceRecognizer;

    // 初始化
    public Recognizer(Context context) {
        Logger logger = new Logger();
        byte[] modelData = ModelLoader.getModelData(context, "face_rec.onnx");
        faceRecognizer = new FaceRecognizer(logger, modelData);
    }

    // 释放资源
    @Override
    public void close() {
        faceRecognizer.close();
    }

    // 检测
    public void run(Bitmap bitmap) {
        // 缩放到目标尺寸
        Bitmap resizedBitmap = ImageProcesser.resizeBitmap(bitmap, FaceRecognizer.MODEL_WIDTH, FaceRecognizer.MODEL_HEIGHT, false);
        // 转换成RGB数组
        byte[] rgbData = ImageProcesser.convertBitmapToRGB(resizedBitmap);
        resizedBitmap.recycle(); // 及时释放bitmap资源
        // 运行
        float[] features = faceRecognizer.run(rgbData);
        Log.d(TAG, Arrays.toString(features));
    }

}
```

## 四、注意事项

- 模型推理时间较长，要在独立的线程中运行run方法，避免阻塞主线程
