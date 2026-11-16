# Invoker-Render inference adapters

This package provides the asset, execution, and serving adapters for **Invoker-Render**, the causal four-step streaming model of Code World Model 2.0. The `projects.minimax_h3` import path is retained for compatibility with the vendored runtime and model architecture.

本目录提供 **Invoker-Render**（Code World Model 2.0 的因果四步流式模型）的资产准备、执行与服务适配层。默认使用 CWM 因果生成与 `production` 部署配置；使用方式见下方项目介绍和部署指南。

- [Project overview / 项目介绍](../../../../README.md)
- [Deployment guide / 部署指南](../../../../docs/INFERENCE.md)
- [Package structure / 开发目录](README-project.md)
- [Model contract / 推理约定](README_inference.md)

Run tools from the repository root through `invoker_render.py`. It calls `tools/launch.py`, which resolves the vendored runtime, framework, and project packages. Weights, prepared bundles, and runtime logs remain outside the source tree; paper figures and demo previews live in the repository's `assets/` directory.
