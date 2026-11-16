# Blender proxy demo prototype

## 生成独立的网页短循环

`hero_loop.py` 创建一个新的命名场景，保留当前文件中的其他场景和物体。它把 `.blend` 副本保存到新的外部目录，不覆盖原文件。`hero_package.py` 把同帧四层打包成网页 atlas。这两个入口不使用推理 exporter 或 `17*n+5` 输入长度契约。

在 live Blender 中运行以下 Python，或使用下面的后台构建命令。输出目录必须不存在，失败后的目录也保留，重试时选择新路径。

```python
import runpy
hero = runpy.run_path(r'tools/blender_proxy/hero_loop.py', run_name='hero_loop_live')
scene = hero['build_scene'](r'path/to/hero-loop-new-run')
```

后台命令从仓库根目录运行。先查看 sample 的首帧、中帧、末帧和重复端点，再进行完整导出。

```powershell
$blender = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
$output = 'path/to/hero-loop-new-run'
& $blender --background --threads 4 --python-exit-code 1 --python tools/blender_proxy/hero_loop.py -- build $output
& $blender --background "$output/hero-loop.blend" --threads 4 --python-exit-code 1 --python tools/blender_proxy/hero_loop.py -- sample $output
& $blender --background "$output/hero-loop.blend" --threads 4 --python-exit-code 1 --python tools/blender_proxy/hero_loop.py -- render $output
python tools/blender_proxy/hero_package.py $output
python -m unittest tools.blender_proxy.test_hero_package
```

构建默认 48 帧、24 FPS、每层 `640 × 360`。四层分别导出 RGB、fine、coarse 和 depth 的 PNG 母版，额外渲染第 49 帧来检查循环端点。打包要求四层第 49 帧与第 1 帧逐像素一致，视频仅编码前 48 帧。输出保留 `.blend`、`scene-info.json`、`sample/`、`frames/` 和网页用的 `web/`。sample、frames 和 web 都拒绝覆盖已有目录。

`web/` 包含 `atlas.mp4`、四张首帧 PNG 和 `manifest.json`。四个有效矩形按 RGB 左上、fine 右上、coarse 左下、depth 右下排列，每块四边重复 16 像素边缘，atlas 为 `1344 × 784`。清单记录有效矩形、深度范围、Blender 版本、源脚本与素材 SHA256、端点检查、解码后的帧数、精确时间戳及各 tile 的误差。外部目录中的 `decoded-first.png` 和 `decoded-last.png` 用于检查实际末首衔接，端点相等不能代替这项视觉检查。

## Prototype scope

This prototype creates a fixed third-person walk and turn, with a stationary camera.
It emits a synthetic RGB reference, fine depth-semantic proxy, coarse categorical proxy,
lossless FFV1 masters, H.264 MP4 previews and separate historical prompt files.
It is an initial calibration scene, not a copy of any training data.

The standards and prompt text come from a separate proxy/prompt standard
(`PROXY_STANDARD.md`) that is not distributed with this repository; pass its
directory to `package.py --standard`.
Fine backgrounds use optical-axis depth `clip(230 - 36*log1p(z), 45, 230)`,
the recorded fine palette and 15 approximated moving capsules. Coarse backgrounds
use the categorical codebook and a rigid nine-capsule approximation. Coarse render
pixels are mapped to the nearest palette entry; the manifest reports the changed
pixel count. The mapping can change edge labels and is not a training renderer.

## Requirements and run

Blender 5.2.3 LTS was installed from the Blender Foundation using Windows Package
Manager. The packaging step needs Python with PyAV, NumPy and Pillow. The installed
Blender is used with CPU rendering and two threads. Use a new empty output path;
the scene step refuses to overwrite an existing directory.

```powershell
$blender = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
$output = 'path/to/blender-proxy-new-run'
& $blender --background --threads 2 --python-exit-code 1 --python tools/blender_proxy/scene.py -- $output
& $blender --background --threads 1 --python-exit-code 1 --python tools/blender_proxy/check_depth.py -- $output
python tools/blender_proxy/package.py $output --standard path/to/proxy-standard --frames 39 --sample-check --coarse-action 'One person walks forward, then turns right while continuing forward. The camera remains fixed.'
python -m tools.blender_proxy.verify $output
```

The `.blend` file contains the camera, geometry and 49-frame animation. The
packaged 39-frame window is frames 0–38 of that scene, at 24 FPS. It satisfies
the existing Ref2VA input contract of `17*n+5`; other recipes may have different
length requirements. `fine-lossless.mkv` and `coarse-lossless.mkv` preserve exact
rendered RGB bytes. MP4 previews use lossy H.264/YUV420 and can alter colors.

`reference.png` and the first fine frame use the same camera, geometry, and scene
time; material and background appearance differ. A bounding-box check confirms
the projected synthetic person agrees within two pixels, not semantic equivalence
or real-world visual quality. Three road pixels are checked against independent
camera-ray depth. This synthetic reference is deliberately simple and should be
replaced or stylized, without changing composition, for a quality demo.

`fine-prompt.txt` reproduces the historical fine prompt. `coarse-prompt.txt`
includes the historical coarse prompt plus a description of the sample action.
No-text conditioning is not represented by an empty prompt because supported
encoders and checkpoints have not been verified for that case. Verify the actual
artifact, fixed prompt, processor, per-profile resize, first-frame markers, and
window handling before running a model. The model-specific preview sizes in the
manifest describe historical recipes, not all mixed-training checkpoints.

## Render another Blender scene

`render_blend.py` accepts an existing `.blend`. Give every renderable mesh a
custom `proxy_category` of `road`, `building`, `vegetation`, or `human` and a
`proxy_role` of `all`, `fine`, `coarse`, or `reference`. Human proxies must use
separate `fine` and `coarse` geometry: animate the fine body and keep the coarse
body rigid relative to its root. A `reference` object can hold a separate visible
RGB mesh; it suppresses fine proxy geometry of the same category in the reference.
Assign reference materials to the mesh or name an existing material in
`proxy_reference_material`. Hidden unrelated objects are not exported. Unknown
visible mesh categories are rejected, not silently reclassified.

```powershell
$inputBlend = 'path/to/source/scene.blend'
$output = 'path/to/blender-proxy-another-run'
& $blender --background $inputBlend --threads 2 --python-exit-code 1 --python tools/blender_proxy/render_blend.py -- $output 39
python tools/blender_proxy/package.py $output --standard path/to/proxy-standard --frames 39 --coarse-action 'Describe the actual motion here.'
python -m tools.blender_proxy.verify $output
```

The exporter writes a copy of the source scene, a `scene-info.json` with the
camera, first frame and Blender version, a reference from that frame, and the
fine/coarse sequences. It changes render settings in memory to 672×384, 24 FPS,
Raw color and CPU Cycles. It does not alter the input file. Supported categories,
single-material proxy meshes and render-role rules are intentionally limited.
Other semantics, exact occlusion and training equivalence remain unvalidated.
The `--sample-check` and `check_depth.py` checks depend on this prototype's
particular colors and `Road` geometry and must not be applied to arbitrary scenes.
