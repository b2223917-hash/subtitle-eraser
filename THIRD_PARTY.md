# 留英文 0.3.0 · 第三方组件与许可

本项目代码采用 MIT 许可证。**桌面界面使用 PySide6 / Qt 6.10.2，按 LGPLv3 使用和分发**；MIT 许可不替代第三方组件的许可。便携版保留动态库，可替换兼容版本并重新构建，不限制为调试库修改而进行的逆向工程。

## 组件来源

| 组件 | 版本 / 用途 | 许可与官方来源 |
| --- | --- | --- |
| PySide6、Shiboken6、Qt Base | 6.10.2；界面与窗口 | [Qt for Python 许可](https://doc.qt.io/qtforpython-6/licenses.html)，LGPLv3；随附完整 GPL/LGPL 文本 |
| RapidOCR | 1.4.4；可选语言识别 | [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR)，Apache-2.0 |
| PaddleOCR 模型 | 随 RapidOCR wheel 分发 | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)，Apache-2.0 |
| OpenCV / opencv-python | 4.13.0 / 4.13.0.92；轮廓与修补 | [OpenCV](https://github.com/opencv/opencv) Apache-2.0；[Python 包](https://github.com/opencv/opencv-python) MIT，含各自第三方通知 |
| FFmpeg（OpenCV 后端） | OpenCV 官方 Windows 构建，FFmpeg n4.4.6 | [OpenCV 源分支](https://github.com/opencv/opencv_3rdparty/tree/62bd53c24feadd92af0146462317fcd4d64fd842)，LGPL；保留 OpenCV wheel 内完整第三方许可 |
| Shapely / GEOS | 2.2.0 / 3.14.1；OCR 几何处理 | [Shapely](https://github.com/shapely/shapely) BSD-3-Clause；[GEOS](https://libgeos.org/) LGPL-2.1 |
| ONNX Runtime | 1.31.0；本地 OCR 推理 | [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime)，MIT |
| windows-capture | 2.0.1；窗口捕获 | [官方项目](https://github.com/NiiightmareXD/windows-capture)，MIT |
| Noto Sans SC | 开源中文回退字体 | [Google Fonts 固定版本](https://github.com/google/fonts/tree/a85815a42757630ce188fdad368c2dfc444d4773/ofl/notosanssc)，SIL OFL-1.1；字体与 OFL 同目录 |
| Python、NumPy、Pillow、psutil、PyYAML、pyclipper、FlatBuffers、Protobuf 等 | 见运行依赖锁 | 便携版 licenses/ 按依赖保留原始文本 |
| PyInstaller bootloader | 6.22.3；便携启动 | [PyInstaller 许可](https://pyinstaller.org/en/stable/license.html)，GPL 带允许应用分发的 bootloader 例外 |

Qt 的 PDF、SVG、QML、Quick、Virtual Keyboard 及相应可选插件未用于本工具，发布构建排除它们。应用只使用保留的 Qt Base 动态库及原生平台插件。没有分发 Windows 字体。

## 对应源码与替换库

本次发布同时提供 SubtitleEraser-0.3.0-ThirdPartySources.zip，包含以下未经修改的官方源码：

- Qt Base 6.10.2 和 PySide/Shiboken 6.10.2 完整源码包。
- GEOS 3.14.1 源码包。
- 与 OpenCV FFmpeg DLL 对应的官方源分支：FFmpeg、libvpx、aom、OpenH264 接口头文件、OpenCV 包装代码和构建脚本。

文件 URL、大小与 SHA-256 见 [源码清单](licenses/source-manifest.json)，下载与验证工具为 tools/fetch_license_sources.py。**发布 Windows 二进制时，同一 Release 附上该源码附件和本项目源码**，便于接收者直接取得完整材料；不要只上传孤立 EXE。

替换方法：保留原包副本，在 _internal/PySide6/、_internal/cv2/ 或 Shapely 动态库目录替换同架构、兼容 ABI 的库及必要插件；也可在项目源码环境安装修改后的依赖，然后按 build.cmd 重建。应用没有库签名锁定或阻止替换的检查。Qt/PySide、GEOS 和 FFmpeg 的上游源码包附有各自构建说明。许可证文本见 licenses/，相关权利按原文执行。[Qt 官方 LGPL 说明](https://www.qt.io/development/open-source-lgpl-obligations)

## 独立研究环境

ProPainter 仅为可选离线研究，源码与权重不在本项目包或桌面二进制内。其 [S-Lab License 1.0](https://github.com/sczhou/ProPainter/blob/main/LICENSE) 允许规定条件下的非商业使用，商业使用需联系权利人；本项目的 MIT 许可不改变该条件。

imageio-ffmpeg 用于源码中的验证视频编码，未打入桌面 EXE；其额外下载或随 wheel 的 FFmpeg 按自身许可使用。研究环境中的 PyTorch、CUDA 和下载模型同样按各自条款使用。
