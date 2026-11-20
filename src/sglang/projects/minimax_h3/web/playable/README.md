# Invoker-Render browser client

The browser client provides interactive proxy capture, first-image selection, streaming playback, paired recording, and timing diagnostics. It connects to the existing native model service through a CPU media adapter and a browser gateway; it does not load model weights itself.

本客户端提供交互式 Proxy、首图选择、流式回放、配对录制与时序诊断。模型推理在原生服务中执行，网页不重复加载模型。

## Model deployment / 模型模式

Follow the [deployment guide](../../../../../../docs/INFERENCE.md#browser-client) to start the native service, the `web-adapter` command, and the `web` gateway. The model is served as `invoker-render`. Use the output/proxy geometry declared by the matching source artifact.

1. Open the loopback browser gateway and select a scene.
2. Upload or select a first image matching that scene.
3. Start streaming observation; input and generated video share frame identities.
4. Stop and drain before changing the scene or resetting its state.
5. Download the run record for source, frame, and timing information.

选择场景与匹配首图后开始观测；切换场景前先停止并排空。界面中的 CPU fixture 不调用模型，不可用来证明模型质量或 GPU 性能。

## CPU fixture / 协议演示

From the repository root, in the serving environment:

```sh
python invoker_render.py web --fixture --incremental-proxy --port 18862
```

Open `http://127.0.0.1:18862/`. The fixture echoes input through the media path; it is explicitly labeled and is not Invoker-Render inference. Procedural boxing, racing and street proxies are visualization sources, not a claim of pixel equivalence to training renderers.

## Recording / 录制

Paired recording uses browser WebCodecs to export Proxy and output WebM files with matched frame timestamps. Overload may skip an entire pair; incomplete or mismatched pairs are not exported as successful output. Source changes preserve the previous session's recording identity. Browser decoding/recording capability depends on the browser and runtime platform.

录制按帧身份配对，编码压力下可成对跳帧；配对失败不会作为完整录制导出。录制文件的播放帧率不是模型生成吞吐。

## Optional sources and access / 可选来源与访问

External game integrations are opt-in. Use `serve_web --help` for `--native-game-proxy-url`, `--threejs-racing`, and `--ballpoint-breach`; external game binaries/assets must be supplied separately. Native local-game sources require a loopback gateway and are not exposed through the public gateway mode.

Remote gateway access requires both `--public-url` and a private `--access-key-file`, with HTTPS terminated by a trusted proxy. Never publish the access key or deployment credentials in this repository.

## Validation / 验证

CPU fixtures exercise the browser protocol without running the model. Validate actual model-backed playback and completed paired recordings separately. Focused browser-unit tests live under `../../tests/`, including recording, media, charts, Gantt timing, and source-selection contracts.
