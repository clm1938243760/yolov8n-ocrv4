# yolov8n-ocrv4

本仓库保存 Windows 本地视觉接口服务，用于配合 RK3568/RK3588 网关识别 Windows 画面并返回点击坐标。

当前用途：

- 检测桌面/软件窗口。
- 对检测到的窗口区域做 OCR。
- 定位桌面软件图标。
- 给板子返回窗口框、OCR 文字坐标、图标中心坐标。

## 当前模型

窗口检测模型：

```text
runs/window_yolov8n_640_add9/weights/best.pt
```

图标检测模型：

```text
best.pt
```

窗口模型只有一个类别：

```yaml
0: window
```

业务状态不再靠多类别 label 判断，而是靠 OCR 文本判断，例如 `登录`、`未选择患者`、`就绪`、`新建患者`、`检查完成`、`是否生成 PDF 报告`。

## 启动接口

```powershell
powershell -ExecutionPolicy Bypass -File "D:\window_yolo_train\start_vision_api.ps1"
```

默认端口：

```text
5002
```

健康检查：

```powershell
curl.exe -sS http://127.0.0.1:5002/health
```

关闭接口：

```powershell
Get-NetTCPConnection -LocalPort 5002 -ErrorAction SilentlyContinue |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

## 接口

窗口检测：

```http
POST /window/detect
Content-Type: application/json
```

请求：

```json
{
  "image_base64": "JPEG图片base64字符串"
}
```

图标定位：

```http
POST /icon/locate
Content-Type: application/json
```

请求：

```json
{
  "image_base64": "JPEG图片base64字符串",
  "software": "人体成分分析仪"
}
```

## 文档

详细过程见：

```text
docs/local_vision_model_api_technical_doc.md
```

