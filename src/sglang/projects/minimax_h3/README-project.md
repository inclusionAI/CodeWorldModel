# Invoker-Render package structure / 开发目录

This package owns the Invoker-Render inference adapters, deployment tools, and browser client. Framework sources are vendored; no submodule initialization is required.

本目录负责 Invoker-Render 推理适配、部署工具与浏览器客户端。框架源码直接随仓库分发，不需要初始化 Git 子模块。

| Directory / 目录 | Responsibility / 职责 |
| --- | --- |
| `inference/` | Frozen assets, native runtime adapters, streaming clients, and web gateways / 资产与服务适配 |
| `tools/` | Bundle preparation, serving, file inference, and environment checks / 命令行入口 |
| `configs/recipes/` | Model inference contracts / 模型推理协议 |
| `configs/runs/` | Parallelism and backend presets / 执行拓扑 |
| `web/playable/` | Interactive proxy capture, playback, recording, and diagnostics / 交互网页 |
| `tests/` | Adapter, CLI, media, and browser-unit regressions / 项目回归 |
| `frameworks/` | Vendored model definitions and preprocessing; session graph, scheduling, and transport / 模型基础组件与会话交付框架 |

## Entry points / 入口

The repository-root `invoker_render.py` maps each command to a tool module and runs it through `tools/launch.py`:

- `prepare` → `projects.minimax_h3.tools.prepare_omni_h3`: verify assets and prepare a deployment bundle.
- `serve` → `projects.minimax_h3.tools.serve_omni_h3`: launch the native streaming service.
- `infer` → `projects.minimax_h3.tools.infer_omni_h3`: run first-image/proxy/prompt file inference.
- `web-adapter` → `projects.minimax_h3.tools.serve_omni_h3_web`: attach the CPU media adapter to an existing service.
- `web` → `projects.minimax_h3.tools.serve_web`: run the browser gateway.

Other modules in `tools/` have no command; run them with `tools/launch.py <module>`.

The served model name is `invoker-render`. `minimax_h3` imports and H3 architecture identifiers are retained runtime contracts, not alternative public names. Execution changes must preserve the configured conditioning, sampler, history visibility, and positional alignment unless a different model contract is explicitly selected.

服务对外的模型名为 `invoker-render`。修改执行实现时，应保留所选模型契约中的条件编码、采样器、历史可见性与位置对齐语义，不把算法替换当作无损优化。

See the [deployment guide](../../../../docs/INFERENCE.md), [inference contract](README_inference.md), and [third-party notices](../../../../THIRD_PARTY_NOTICES.md). Runtime data and model weights must stay outside Git.
