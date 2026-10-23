# 留英文 0.3.0 · GitHub 发布指南

这是 0.3.0 的完整本地发布材料，尚未上传到 GitHub。项目代码采用 MIT 许可证，第三方组件保留各自的许可证。

## 先分清两个用途

- 给普通用户：发送 `SubtitleEraser-0.3.0-Windows-x64.zip`。对方完整解压后双击 `SubtitleEraser.exe`，无需安装 Python、开发工具或额外模型。
- 给 GitHub 仓库：使用总包里的 `project/` 源码目录。运行包、第三方源码包和校验文件放到 GitHub Releases 附件。

不要把整个发布总包当作源码提交，也不要只发送孤立的 EXE。运行包需要保留 `_internal` 文件夹。

## 总包里的文件

| 文件或目录 | 用途 |
| --- | --- |
| `project/` | 可独立提交的源码、README、MIT 许可证、贡献说明、依赖锁、测试与 Windows 构建工作流 |
| `project/docs/QUICK_START.md` | 面向普通用户的使用说明 |
| `project/docs/RELEASE_NOTES-0.3.0.md` | 可复制到 GitHub Release 的发布说明 |
| `project/docs/REPOSITORY_SETUP.md` | 建仓时可复制的名称、简介和主题词 |
| `release-assets/SubtitleEraser-0.3.0-Windows-x64.zip` | 解压即用的 Windows x64 便携版 |
| `release-assets/SubtitleEraser-0.3.0-Source.zip` | 对应项目源码快照 |
| `release-assets/SubtitleEraser-0.3.0-ThirdPartySources.zip` | 随二进制提供的 Qt/PySide、GEOS、FFmpeg 对应源码和许可材料 |
| `release-assets/SHA256SUMS.txt` | 上述三个 ZIP 的 SHA-256 校验值 |

## 发布步骤

1. 准备 GitHub 账号，并新建公开仓库，名称建议为 `subtitle-eraser`。简介和主题词可复制 `REPOSITORY_SETUP.md`。
2. 把 `project/` 内的文件提交到仓库根目录，包含 `.github`、`.gitignore`、`.gitattributes` 等隐藏项。可先把 `project/` 复制到独立目录，再使用 GitHub Desktop 建立、提交并发布仓库；不要在原 Unity 工程的 Git 历史中操作。
3. 打开仓库的 Actions，查看 Windows 构建结果。本地已执行的验证不等于你的 GitHub 仓库首次构建已经通过。
4. 在 Releases 创建新发布，标签为 `v0.3.0`，标题建议为 `留英文 v0.3.0 · Windows 便携版`。发布正文复制 `RELEASE_NOTES-0.3.0.md`，可按需要修改措辞。
5. 上传 `release-assets/` 中的三个 ZIP 和 `SHA256SUMS.txt`，检查附件齐全后发布。让普通用户下载文件名含 `Windows-x64` 的 ZIP；GitHub 自动生成的 `Source code` 附件不能直接运行。

GitHub 的 Release 附件适合分发软件包；源码仓库普通单文件超过 100 MiB 会被阻止。不要把约 150 MB 的 Windows 包直接提交到源码仓库。[GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[GitHub Release 说明](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)。

发布 Windows 包时，一并保留本项目源码和第三方对应源码附件，详见 `THIRD_PARTY.md`。本项目的 MIT 许可不会替代 Qt、GEOS 等组件的许可。

## 还需要你提供什么

必需的只有 GitHub 账号、仓库名称，以及由你执行的最终上传/发布。本材料不包含账号令牌，也没有预设远程仓库。

代码签名、安装向导和演示视频可以以后补充；当前便携版使用不依赖这些，不需要购买云推理服务或 API。

## 后续重新构建

以下步骤在独立源码目录、Windows x64 环境执行：

```powershell
.\setup.cmd
.\.venv\Scripts\python.exe -m pytest -q --ignore=tests/test_packaging.py
.\build.cmd dist-release build-release
$env:SUBTITLE_ERASER_DIST_ROOT = 'dist-release'
.\.venv\Scripts\python.exe -m pytest tests/test_packaging.py -q
.\.venv\Scripts\python.exe tools\fetch_license_sources.py
.\.venv\Scripts\python.exe tools\make_release.py --dist-root dist-release
```

发布工具只打包明确列入的源码目录，不包括播放器配置、真实截图、日志、旧构建、研究模型或上层工程。第三方源包验证 SHA-256，不执行下载内容。升级依赖时应同步更新对应源码清单和许可材料。

当前已知效果与兼容性范围见 `docs/VALIDATION.md`。正式打包不代表已完成所有播放器、独占全屏、HDR 或受保护视频的兼容性验证。
