# rknn模型库

## 依赖
- rknn运行时(c_lib/librknnrt.so)

## 案例
- 基准
  - 山东烟柜(端到端fp16)
    - 640 ```./bench ./cig_sd_138.640.yolo26s.fp16.rk3588.rknn ./cig_2026_01_21/```
    - 1280 ```./bench ./cig_sd_138.1280.yolo26s.fp16.rk3588.rknn ./cig_2026_01_21/```