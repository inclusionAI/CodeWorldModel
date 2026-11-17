# Third-party notices and source provenance

This distribution vendors the runtime directly. The source tree contains no Git
submodules or gitlinks; component licenses remain with their source packages.

| Source | Revision | Notes |
| --- | --- | --- |
| H3 project source | `6c21fd95dfeb706d361b3d02bf2100265e0375a9` | Imported project adapters and recipes; subsequent changes are recorded in this repository. |
| SGLang | `c86cbb0587999f4763d2617f396a1a0745c965f2` | Vendored runtime with causal H3 integration. |
| RAVEN | `692ec211d4cd37a99a04773186fc2bf810430f78` | Vendored package source. |
| SGLang-Omni | `63591434c6051d509c9ea3518736e3271d283453` | Vendored package source. |

Model weights, prepared bundles, datasets, and runtime logs are not included.
The report teaser in `assets/` and short preview animations in `assets/demo/`
are included for documentation; the source videos are not included. The teaser
includes frames from third-party games for research illustration; those remain
the property of their respective owners.

## Component licenses

Code written for this repository is licensed under the [Apache License 2.0](LICENSE).
Each vendored component is governed by the license file in its own directory:

- [SGLang](src/sglang/LICENSE) — Apache-2.0
- [RAVEN](src/sglang/projects/minimax_h3/frameworks/raven/LICENSE) — CC BY-NC 4.0 (non-commercial)
- [SGLang-Omni](src/sglang/projects/minimax_h3/frameworks/sglang-omni/LICENSE) — Apache-2.0

H3 architecture identifiers and `cwm_*` protocol fields are retained for model
and checkpoint compatibility. The served model name is `invoker-render`.
Model weights and other separately distributed assets retain their own terms.
See the [legal notice](LEGAL.md) and [deployment guide](docs/INFERENCE.md).

