# onnx模型库

## 依赖
- ort运行时(c_lib/libonnxruntime.so)

## 案例
- 营业执照(pp模型)
```
cargo run --example license -- ./models/license_det.yolo.onnx ./models/license_biz_det.pp.onnx ./models/license_rec.pp.onnx ./models/license_dict.txt ../images/license.biz.0.jpg
```
- 烟证(pp模型)
```
cargo run --example license -- ./models/license_det.yolo.onnx ./models/license_cig_det.pp.onnx ./models/license_rec.pp.onnx ./models/license_dict.txt ../images/license.cig.0.jpg
```
- 烟柜(yolo8, 多头)
```
cargo run --example cig -- ./models/cig_sd_138.yolo8.onnx ../images/cig.0.jpg
```
- 基准
  - yolo26端到端 ```cargo run --release --example yolo26_e2e -- ./models/cig_sd_138.640.yolo26s.onnx ../images/cig_2026_01_21```
  - yolo8多头 ```cargo run --release --example yolo8 -- ./models/cig_sd_138.640.yolo8.onnx ../images/cig_2026_01_21```