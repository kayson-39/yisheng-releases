# 译声实时翻译（Yisheng Live Translator）

看日语直播时的实时中文字幕工具。从 Windows 当前播放设备采集声音，在本机识别日语，并实时显示简体中文翻译——默认全程本地运行，不需要 API 密钥，也不上传音频和字幕。

> 当前版本：**0.10.11**（Windows x64） · 许可：MIT

---

## 下载安装

到 [Releases](https://github.com/kayson-39/yisheng-releases/releases/latest) 下载最新安装器：

| 文件 | 说明 |
| --- | --- |
| `yisheng-setup-0.10.11.exe` | 单文件安装器，约 1.24 GB，双击即装 |

安装后从开始菜单打开「译声实时翻译」。安装包体积大是因为内置了离线识别模型与运行环境；**不含**约 4.25 GB 的 Sakura 翻译大模型（见下）。

---

## 功能

- **系统声音采集**：直接听你正在播放的直播/视频声音，不需要麦克风、不需要虚拟声卡。
- **本地日语识别**：基于 sherpa-onnx 的离线 ASR，音频不离开本机。
- **本地实时翻译**：默认「本地 · 轻量翻译」，无需联网、无需密钥；也可接入兼容的流式翻译服务或 Ollama。
- **可插拔大模型**：支持 Sakura 等 GGUF 翻译模型，可让程序代下载，也可指定已有文件。
- **滚动字幕与历史**：字幕浮层可回看、复制、导出 SRT；历史按场次保存，可改名、换保存目录。
- **草稿改写**：日语识别结果被修正后只补译最新待译版本，作废的草稿不会覆盖已出译文。
- **过载保护**：识别与翻译使用有界队列，积压时明确提示而不是无限卡死；超时可手动补译。

---

## 第一次使用

1. 安装后从开始菜单（或桌面快捷方式）打开「译声实时翻译」。
2. 播放直播，在译声界面点击 **「开始聆听」**。首次加载模型会比之后慢。
3. 听不到声音时，到设置里确认选中的是**正在播放直播的输出设备**，而不是麦克风。
4. 结束时点停止。关闭时若提示尚未保存，先取消退出、重试保存或导出。

### 关于翻译模型

安装包不含约 4.25 GB 的 Sakura 大模型。第一次使用可以选择：

- 在设置里**下载推荐模型**（需要网络和额外磁盘空间）；
- 指定你**已有的 GGUF** 文件（升级时可继续复用）；
- 直接使用随包提供的**「本地 · 轻量翻译」**——不下载任何东西也能用。

不想要大模型时不必改用云服务。

---

## 系统要求

- Windows 10/11 x64
- 安装约需 2.5 GB 可用空间；若要下载 Sakura 模型另需约 4.25 GB
- 内存建议 8 GB 以上；使用大模型翻译时建议有独立显卡或较强 CPU
- 实时性受模型、后台程序、内存与显存影响，不保证所有机器都能流畅实时

---

## 隐私

- **默认本地模式**：音频与字幕都不会发送给任何翻译服务。
- 只有你**主动选择**在线翻译或在线备用服务时，相关文本才会发给所选服务。
- 用户配置、历史库、日志保存在 `%LOCALAPPDATA%\YishengTranslator\UserData`，与程序目录分离，卸载不会删除。
- 请勿把自己的配置文件、历史库或日志打包转发，其中可能含有个人内容。

### 检查与修复

若提示模型或引擎缺失，请从完整发行文件重新安装，不要单独从不明来源补 DLL。可在程序目录运行离线自检（不录音）：

```powershell
& '.\app\runtime\py312\python.exe' -B -X utf8 '.\app\scripts\launcher.py' --check
```

---

## 源码结构

```
jpsub/
  main.py            程序入口
  core/              识别、翻译、音频采集、分段、导出、历史
    asr.py           sherpa-onnx 语音识别
    translate.py     翻译调度、流式处理、重试与超时
    local_mt.py      本地轻量翻译
    llama_engine.py  GGUF / Sakura 模型推理
    audio.py         系统播放设备采集与重采样
    segment.py       断句与草稿修订
  ui/                PySide6 界面（主窗、字幕浮层、设置、历史、主题）
packaging/           安装器构建脚本（Inno Setup）与发布文档
tools/               发布打包、验收与自检脚本
tests/               单元与回放测试
```

## 从源码运行

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m jpsub.main
```

需要 Python 3.12。构建安装器见 [`packaging/installer.md`](packaging/installer.md)，版本变更与修复明细见 [`DEVELOPMENT.md`](DEVELOPMENT.md)。

---

## 许可与第三方

- 项目代码：[MIT](LICENSE)
- 第三方组件与素材署名见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
- 背景素材署名见 `jpsub/assets/backgrounds/credits.json`

本软件用于个人观看直播时的辅助理解，识别与翻译结果可能存在错误，请勿用于需要准确性的正式场合。
