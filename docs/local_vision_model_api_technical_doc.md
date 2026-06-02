# 本地视觉模型与接口技术文档

日期：2026-06-02

## 1. 目标

本地视觉服务用于配合 RK3568/RK3588 网关完成 Windows 画面识别和鼠标点击决策。

当前系统分成两部分：

1. Windows 本地视觉接口服务
   - 运行目录：`D:\window_yolo_train`
   - 服务端口：`5002`
   - 负责窗口检测、窗口 OCR、桌面图标定位。

2. RK3568/RK3588 网关程序
   - 板子截图后把 JPEG 转成 base64。
   - 调用 Windows 本地接口。
   - 根据返回的坐标和 OCR 内容执行 HID 鼠标/键盘动作。

当前视觉逻辑不再依赖旧的多类别窗口 label。窗口检测统一返回 `label: "0"`，具体状态由 OCR 文字判断。

## 2. 当前目录结构

本地视觉服务目录：

```text
D:\window_yolo_train
├── .venv311\
├── window_detect_all_api.py
├── start_vision_api.ps1
├── test_vision_api.ps1
├── train_window.py
├── window.yaml
├── yolov8n.pt
├── best.pt
├── runs\
│   ├── window_yolov8n_640\
│   │   └── weights\
│   │       ├── best.pt
│   │       └── last.pt
│   └── window_yolov8n_640_add9\
│       ├── args.yaml
│       ├── results.csv
│       └── weights\
│           ├── best.pt
│           └── last.pt
├── images\
│   ├── train\
│   └── val\
├── labels\
│   ├── train\
│   └── val\
└── em\
    └── em\
        ├── feats.npy
        └── labels.json
```

当前服务使用的主要文件：

```text
窗口模型：
D:\window_yolo_train\runs\window_yolov8n_640_add9\weights\best.pt

图标模型：
D:\window_yolo_train\best.pt

接口脚本：
D:\window_yolo_train\window_detect_all_api.py

启动脚本：
D:\window_yolo_train\start_vision_api.ps1
```

## 3. Python 环境

当前接口使用本地虚拟环境：

```text
D:\window_yolo_train\.venv311
```

已确认主要依赖版本：

```text
flask       3.1.3
ultralytics 8.4.60
paddleocr   2.7.3
opencv      4.6.0
numpy       1.26.4
torch       2.12.0+cpu
```

当前是 CPU 推理，没有使用 GPU。

## 4. 模型说明

### 4.1 窗口检测模型

窗口模型是 YOLO 检测模型。

当前使用：

```text
D:\window_yolo_train\runs\window_yolov8n_640_add9\weights\best.pt
```

模型类别：

```yaml
names:
  0: window
```

也就是所有窗口都检测为一个类别 `window`，接口返回时统一映射成：

```json
{
  "label": "0"
}
```

窗口状态不靠 label 区分，而是靠 OCR 文本区分。

### 4.2 图标定位模型

当前目录没有 `icon_match_local.py`，所以图标定位没有使用 embedding 检索模式。

当前实际模式：

```text
yolo_ocr_fallback
```

健康检查返回示例：

```json
{
  "icon_device": "yolo_ocr_fallback",
  "icon_index_n": 0,
  "icon_mode": "yolo_ocr_fallback",
  "icon_model": "D:\\window_yolo_train\\best.pt",
  "icon_model_names": {
    "0": "icon"
  },
  "model": "D:\\window_yolo_train\\runs\\window_yolov8n_640_add9\\weights\\best.pt",
  "model_names": {
    "0": "window"
  },
  "ocr": "paddleocr",
  "ok": true,
  "response_label": "0"
}
```

图标定位流程：

1. 用图标 YOLO 模型检测桌面图标区域。
2. 对图标附近文字区域做 OCR。
3. OCR 文字包含目标软件名称时，返回图标中心点。

当前目标软件名称通常是：

```text
人体成分分析仪
```

## 5. 训练数据

当前数据集配置文件：

```text
D:\window_yolo_train\window.yaml
```

内容：

```yaml
path: D:/window_yolo_train
train: images/train
val: images/val

names:
  0: window
```

当前目录统计：

```text
images/train: 81
images/val:   18
labels/train: 72
labels/val:   18
```

注意：当前 `images/train` 比 `labels/train` 多 9 个文件。复现训练前应检查是否有图片缺少同名 label。

检查命令：

```powershell
$ROOT = "D:\window_yolo_train"

$imgs = Get-ChildItem "$ROOT\images\train" -Filter *.jpg
$labs = Get-ChildItem "$ROOT\labels\train" -Filter *.txt

$labNames = @{}
$labs | ForEach-Object { $labNames[$_.BaseName] = $true }

$imgs | Where-Object { -not $labNames.ContainsKey($_.BaseName) } |
  Select-Object Name
```

YOLO 标签格式：

```text
class_id x_center y_center width height
```

坐标均为归一化坐标，取值范围 `0.0` 到 `1.0`。

当前所有窗口统一使用类别：

```text
0
```

## 6. 训练过程

初始训练脚本：

```text
D:\window_yolo_train\train_window.py
```

内容核心：

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=r"D:\window_yolo_train\window.yaml",
    imgsz=640,
    epochs=80,
    batch=8,
    workers=0,
    project=r"D:\window_yolo_train\runs",
    name="window_yolov8n_640",
    exist_ok=True,
)
```

后续补充 9 张图片后的训练目录：

```text
D:\window_yolo_train\runs\window_yolov8n_640_add9
```

训练参数来自：

```text
D:\window_yolo_train\runs\window_yolov8n_640_add9\args.yaml
```

关键参数：

```yaml
model: D:\window_yolo_train\runs\window_yolov8n_640\weights\best.pt
data: D:\window_yolo_train\window.yaml
epochs: 40
batch: 4
imgsz: 640
device: cpu
project: D:\window_yolo_train\runs
name: window_yolov8n_640_add9
```

最后一轮验证指标来自 `results.csv`：

```text
precision(B): 1.00000
recall(B):    0.96370
mAP50(B):     0.99500
mAP50-95(B):  0.96568
```

## 7. 接口服务

接口脚本：

```text
D:\window_yolo_train\window_detect_all_api.py
```

监听地址：

```text
0.0.0.0:5002
```

环境变量：

```text
VISION_API_HOST       默认 0.0.0.0
VISION_API_PORT       默认 5002
WINDOW_MODEL_PATH     可覆盖窗口模型路径
ICON_MODEL_PATH       可覆盖图标模型路径
```

### 7.1 开启服务

PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File "D:\window_yolo_train\start_vision_api.ps1"
```

启动脚本会做几件事：

1. 检查 `.venv311\Scripts\python.exe` 是否存在。
2. 检查 `window_detect_all_api.py` 是否存在。
3. 杀掉当前占用 `5002` 端口的旧进程。
4. 清空 `api_server.log` 和 `api_server.err.log`。
5. 用隐藏窗口启动 Flask 服务。
6. 等待 8 秒后请求 `/health`。

### 7.2 健康检查

本机：

```powershell
curl.exe -sS http://127.0.0.1:5002/health
```

板子：

```bash
curl -sS http://192.168.110.5:5002/health
```

其中 `192.168.110.5` 是当前 Windows 服务电脑 IP。换电脑或换网络后需要修改为新 IP。

### 7.3 关闭服务

PowerShell：

```powershell
Get-NetTCPConnection -LocalPort 5002 -ErrorAction SilentlyContinue |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

### 7.4 查看日志

```powershell
Get-Content "D:\window_yolo_train\api_server.log" -Tail 80
Get-Content "D:\window_yolo_train\api_server.err.log" -Tail 80
```

## 8. 接口协议

### 8.1 健康检查

请求：

```http
GET /health
```

返回：

```json
{
  "ok": true,
  "model": "D:\\window_yolo_train\\runs\\window_yolov8n_640_add9\\weights\\best.pt",
  "model_names": {
    "0": "window"
  },
  "response_label": "0",
  "icon_model": "D:\\window_yolo_train\\best.pt",
  "icon_model_names": {
    "0": "icon"
  },
  "icon_device": "yolo_ocr_fallback",
  "icon_index_n": 0,
  "icon_mode": "yolo_ocr_fallback",
  "ocr": "paddleocr"
}
```

### 8.2 窗口检测

请求：

```http
POST /window/detect
Content-Type: application/json
```

请求体：

```json
{
  "image_base64": "JPEG图片base64字符串"
}
```

可选参数：

```json
{
  "conf": 0.25,
  "iou": 0.95,
  "max_det": 300,
  "ocr_min_score": 0.0
}
```

返回：

```json
{
  "windows": [
    {
      "label": "0",
      "box": [222, 200, 918, 823],
      "ocr": [
        {
          "text": "人体成分分析仪",
          "center": [572, 254]
        },
        {
          "text": "新建患者",
          "center": [290, 322]
        }
      ]
    }
  ]
}
```

字段说明：

```text
windows       检测到的窗口数组
label         当前统一为 "0"
box           窗口框坐标 [x1, y1, x2, y2]
ocr           该窗口内部识别到的文字数组
ocr.text      OCR 文字
ocr.center    OCR 文字中心点，绝对屏幕坐标
```

兼容别名：

```text
POST /detect_window
```

### 8.3 图标定位

请求：

```http
POST /icon/locate
Content-Type: application/json
```

请求体：

```json
{
  "image_base64": "JPEG图片base64字符串",
  "software": "人体成分分析仪"
}
```

返回：

```json
{
  "center": [38, 534]
}
```

未找到：

```json
{
  "center": null
}
```

错误：

```json
{
  "error": "missing software"
}
```

兼容别名：

```text
POST /locate_icon
```

## 9. 本机测试流程

测试脚本：

```text
D:\window_yolo_train\test_vision_api.ps1
```

示例：

```powershell
powershell -ExecutionPolicy Bypass -File "D:\window_yolo_train\test_vision_api.ps1" `
  -Image "D:\Documents\New project\vision_probe\current.jpg" `
  -BaseUrl "http://127.0.0.1:5002" `
  -Software "人体成分分析仪"
```

这个脚本会：

1. 读取图片。
2. 转成 base64。
3. 调用 `/window/detect`。
4. 调用 `/icon/locate`。
5. 保存响应文件：

```text
D:\window_yolo_train\api_test_window_response.json
D:\window_yolo_train\api_test_icon_response.json
```

## 10. 板子截图和调用流程

### 10.1 RK3568 当前采集方式

RK3568 当前使用 HDMI 转 USB 采集卡。

当前稳定节点：

```text
/dev/video9
```

当前稳定格式：

```text
MJPG 1920x1080 30fps
```

稳定截图方式：

```bash
mkdir -p /tmp/vision_probe
rm -f /tmp/vision_probe/.current_*.jpg

gst-launch-1.0 -q -e \
  v4l2src device=/dev/video9 io-mode=2 num-buffers=30 ! \
  'image/jpeg,width=1920,height=1080,framerate=30/1' ! \
  multifilesink location=/tmp/vision_probe/.current_%02d.jpg

cp /tmp/vision_probe/.current_29.jpg /tmp/vision_probe/current.jpg
ls -lh /tmp/vision_probe/current.jpg
```

使用第 29 帧是为了避开首帧黑屏或采集卡刚启动时画面不稳定。

### 10.2 RK3588 HDMI RX 采集方式

RK3588 使用板载 HDMI RX。

稳定节点：

```text
/dev/video40
```

常用截图方式：

```bash
mkdir -p /tmp/vision_probe

IMG=/tmp/vision_probe/current.jpg

gst-launch-1.0 -q -e v4l2src device=/dev/video40 num-buffers=1 ! \
  'video/x-raw,format=BGR,width=1920,height=1080,framerate=60/1' ! \
  videoconvert ! jpegenc quality=90 ! filesink location="$IMG"

ls -lh "$IMG"
```

### 10.3 板子手动调用窗口接口

```bash
mkdir -p /tmp/vision_probe

IMG=/tmp/vision_probe/current.jpg
REQ=/tmp/vision_probe/window_request.json
RESP=/tmp/vision_probe/window_response.json

python3 - <<PY
import base64, json
img = "$IMG"
req = "$REQ"
with open(img, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")
with open(req, "w", encoding="utf-8") as f:
    json.dump({"image_base64": b64}, f, ensure_ascii=False)
print(req)
PY

curl -sS -X POST "http://192.168.110.5:5002/window/detect" \
  -H "Content-Type: application/json" \
  --data-binary @"$REQ" | tee "$RESP"
```

### 10.4 板子手动调用图标接口

```bash
REQ=/tmp/vision_probe/icon_request.json
RESP=/tmp/vision_probe/icon_response.json

python3 - <<PY
import base64, json
img = "/tmp/vision_probe/current.jpg"
req = "$REQ"
with open(img, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")
with open(req, "w", encoding="utf-8") as f:
    json.dump(
        {
            "image_base64": b64,
            "software": "人体成分分析仪",
        },
        f,
        ensure_ascii=False,
    )
print(req)
PY

curl -sS -X POST "http://192.168.110.5:5002/icon/locate" \
  -H "Content-Type: application/json" \
  --data-binary @"$REQ" | tee "$RESP"
```

## 11. 网关程序里的状态判断

当前网关不再相信窗口 label 的具体类别，只把模型检测到的窗口作为 OCR 容器。

主要规则：

1. 没有检测到窗口
   - 调用 `/icon/locate`
   - 找到桌面图标坐标后双击打开软件。

2. 检测到登录窗口
   - 判断 OCR 是否包含：

```text
用户登录
用户名
密码
登录
```

   - 点击 OCR 返回的 `登录` 坐标。

3. 检测到主界面且未选择患者
   - 判断 OCR 是否包含：

```text
未选择患者
就绪
新建患者
```

   - 点击 `新建患者`。

4. 检测到新建患者窗口
   - 判断 OCR 是否包含：

```text
患者号
姓名
性别 / 年龄
开单科室
确认
```

   - 触发原来的 HID 表单录入。
   - 最后点击 `确认`。

5. 患者录入完成后
   - 判断 OCR 是否包含：

```text
患者号
就绪
开始检查
```

   - 点击 `开始检查`。

6. 检查完成
   - 判断 OCR 是否包含：

```text
检查完成
数据分析
```

   - 点击 `数据分析`。

7. PDF 报告确认
   - 判断 OCR 是否包含：

```text
是否生成 PDF 报告
```

   - 在所有包含 `是` 的 OCR 字段里，选择纵坐标更大的那个点点击。

8. 报告生成完成
   - 判断 OCR 是否包含：

```text
检查报告已生成
确定
```

   - 点击 `确定`。
   - 再检测回主界面后点击 `新建患者`，结束本次任务。

## 12. 服务联调命令

### 12.1 Windows 开视觉接口

```powershell
powershell -ExecutionPolicy Bypass -File "D:\window_yolo_train\start_vision_api.ps1"
```

### 12.2 板子测视觉接口连通

```bash
curl -sS http://192.168.110.5:5002/health
```

### 12.3 板子重启网关

RK3568：

```bash
sudo systemctl restart rk3568-gateway.service
journalctl -u rk3568-gateway.service -f
```

RK3588：

```bash
sudo systemctl restart rk3588-gateway.service
journalctl -u rk3588-gateway.service -f
```

### 12.4 模拟扫码

```bash
curl -sS -X POST http://127.0.0.1:8080/scan \
  -H "Content-Type: application/json" \
  -d '{"code":"P2605260007"}'
```

## 13. 常见问题

### 13.1 Windows 服务启动后板子访问不到

检查：

```powershell
curl.exe -sS http://127.0.0.1:5002/health
ipconfig
```

板子上：

```bash
ping 192.168.110.5
curl -sS http://192.168.110.5:5002/health
```

如果本机能访问，板子不能访问，通常是：

```text
Windows 防火墙
IP 地址变了
板子和电脑不在同一网段
服务没有绑定 0.0.0.0
```

### 13.2 `/icon/locate` 返回 `center: null`

可能原因：

```text
桌面图标被遮挡
目标软件名称 OCR 没识别出来
请求里 software 字段为空
图标模型未加载
截图不是当前桌面
```

检查：

```powershell
curl.exe -sS http://127.0.0.1:5002/health
```

确认返回里有：

```json
{
  "icon_model_names": {
    "0": "icon"
  },
  "icon_mode": "yolo_ocr_fallback"
}
```

### 13.3 `/window/detect` 返回空窗口

可能原因：

```text
截图黑屏
采集卡节点不对
窗口被最小化
窗口位置或样式超出训练集覆盖范围
模型阈值过高
```

可以临时降低阈值：

```json
{
  "image_base64": "...",
  "conf": 0.15
}
```

### 13.4 OCR 识别错字

当前逻辑不要求 OCR 完全正确，但关键字必须能识别出来。

关键字包括：

```text
登录
未选择患者
就绪
新建患者
患者号
开单科室
开始检查
检查完成
数据分析
是否生成 PDF 报告
检查报告已生成
确定
```

如果某个关键词长期识别不稳定，优先调整业务逻辑做模糊匹配，而不是只改模型。

## 14. 后续改进建议

1. 修复训练集图片和 label 数量不一致问题。
2. 把 `window_detect_all_api.py` 里的模型路径写入配置文件，而不是只靠默认路径。
3. 增加接口日志，记录每次请求图片大小、检测窗口数、耗时。
4. 增加 `/version` 接口，返回模型版本、接口版本、训练日期。
5. 增加批量测试脚本，用固定 20 张图片统计窗口检测成功率和 OCR 关键词命中率。
6. 如果产品最终不需要视觉，ESP32-P4 版本只保留 HID、MSC、上传，不迁移这套视觉接口。

