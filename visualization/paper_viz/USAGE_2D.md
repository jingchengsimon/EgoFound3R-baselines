# 2D 多方法对比可视化 · 使用文档

版本：2026-09-17（样式已由用户确认定稿，参数已锁定）
适用对象：后续接手渲染的任何同学 / agent。

这份文档 + `visualization/paper_viz/` 目录，就足以在服务器上批量产出论文用的
2D 对比图与视频。**不要改样式参数**（第 5 节列出），否则新旧图会不一致。

---

## 0. 产物是什么

对每一段 300 帧（5 × 60 帧窗口）的片段，一次运行输出三样东西：

| 产物 | 内容 | 规格（定稿） |
|---|---|---|
| `fig2_2d_matrix.png` | 多帧拼接大图，每帧一行 | 5 行 × 15 列，cell 720 px |
| `panels_2d/*.png` | 每列单独一张 PNG，方便手动拼进论文 | 15 张，720×514 |
| `video2_2d_matrix.mp4` | 每 clip 一行的 15 列视频 | 300 帧 / 5746×596 / 30 fps / 10 s |

15 列合同（左→右）：

```
RGB | WiLoR | PAD-Hand | EgoForce | Dyn-HaMR | HaWoR | ReViV4D | EgoFound3R | GT |
Ego visibility | GT visibility | Ego contact | GT contact | Ego distance | GT distance
```

* 前 8 个方法列只画 hand geometry（点 + 面连线 + joint 骨架）；
* visibility / contact / distance 只对 EgoFound3R 与 GT 画；
* 几何列**完全不使用物体**（不看物体遮挡、不做物体提亮）；
  只有 GT 的 visibility / contact / distance 三列用物体（它们本身就是手—物信号）；
* Dyn-HaMR 只在注册了预测的窗口渲染，其余窗口画 `unavailable` 斜纹瓦片——这是正确行为，不要补值。

---

## 1. 环境与目录

### 服务器项目目录（推荐直接用它）

```bash
cd /mnt/workspace/sjc/paper_viz_2d/visualization/paper_viz
```

这是 baselines 仓库 `viz-paper-20260917` 分支的 git worktree（代码与本地仓库一致，
`git pull` 即可更新）。文档、脚本、代码都在这里。

### 运行时要求

| 依赖 | 要求 |
|---|---|
| Python | `/usr/local/bin/python3`（numba 0.61 / numpy 2.2 / scipy 1.18 / cv2 4.11） |
| 渲染必须用 `python3` 而不是 conda 环境 | `/mnt/workspace/sjc/envs/egofound3r` 里没有 numba |
| `ffmpeg` | 在 PATH 中（视频封装） |
| 本机 rsync + ssh | 把产物拉回本地/共享盘 |

### 关键数据根（可直接用，也可通过环境变量覆盖）

| 变量 | 默认值 | 用途 |
|---|---|---|
| `PREPARED_ROOT` | `/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_strict_arctic_20260820T000131Z/window_inputs` | RGB + 物体几何 + 标定 K |
| `HAWOR_ROOT` | `/mnt/workspace/sjc/DATA/eval_artifacts/hawor_native_camera_repair_20260915_v7_5000/datasets` | HaWoR 预测（按 `predictions.jsonl` 索引） |
| `CONTACT_ROOT` | `/mnt/cpfs/sjc/eval_artifacts/result3_completion_20260910_v1` | Ego contact head 等接触预测 |
| `MAPPING` | `/mnt/workspace/sjc/EgoFound3R_viz_8fc061a_20260917/egohandmetric_prompt/data/mano_upsampling/mano_195_to_778.npz` | 195→778 上采样契约 |
| `SRC_DIR` | `/mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_src_20260917` | relay 出来的逐窗 GT/baseline npz |

> 建议把 `SRC_DIR` 建在 `/mnt/workspace`（共享盘），不要放 `/tmp`：qingcang 节点的
> `/tmp` 重启即丢。

---

## 2. 数据前置条件（红线，务必遵守）

1. **EgoFound3R 的结果只能来自 commit `8fc061a615895bd3b5a556f7387bae306e32d9db`
   的 `infer_marker_multiclip.py`**，且默认后处理全开、`--upsample-mano` 显式打开。
   判别方法：看产物 `provenance.json` 的 `tool == "infer_marker_multiclip"`。
   ❌ 绝不能用 `formal_evaluation/scene/adapters/run_egofound3r_baseline.py` 的产物：
   它是原始前向、后处理全关，它的 metadata **也**会写 `inference_commit=8fc061a`，光看 commit 会被骗。
2. Ego 的 2D 投影用**模型预测内参**（`intrinsics_full`）经 keep_aspect 仿射逆映射回原图，
   不用标定 K；GT 与其它 baseline 用各自标定 K。
3. Ego 手在 pred 相机系，几何列只做**手自身** z-buffer 遮挡；不要把物体混进 Ego 的 z-buffer。
4. 遮挡语义：颜色永远表示数值，透明度表示遮挡；被遮挡的点线**照画**、只降 alpha（0.45）。

---

## 3. 单段渲染（先跑通一段）

以 ARCTIC 冒烟段 `s08@microwave_use_01@cam0`（帧 00063–00362）为例。

### 3.1 relay：把冻结的 GT/baseline 预测落成逐窗 npz（每次只需跑一次）

```bash
cd /mnt/workspace/sjc/paper_viz_2d/visualization/paper_viz

python3 tools/relay_sources.py --dataset arctic \
  --manifest ../overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914/selected_manifest.jsonl \
  --out /mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_src_20260917
```

* 源路径来自 `../batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/source_alignment_auxmethods_full104_5001.json`。
* 这些源**大多在 `/mnt/oss`**，该挂载经常是死状态（`transport endpoint is not connected`）。
  若报 `missing`，先用本机 `~/bin/ossutil2`（ossutil v2，配 `~/.ossutilconfig`）把对象拉到本地/工作盘，再重跑。
* HaWoR 不需要 relay（渲染时直接按 `--hawor-root` + `predictions.jsonl` 读）。

### 3.2 跑 EgoFound3R 推理（GPU，每段一次）

```bash
PYTHONPATH=/mnt/workspace/sjc/EgoFound3R_viz_8fc061a_20260917 \
/mnt/workspace/sjc/envs/egofound3r/bin/python \
  /mnt/workspace/sjc/EgoFound3R_viz_8fc061a_20260917/infer_marker_multiclip.py \
  --config /mnt/cpfs/sjc/DATA/EgoFound3R_archive/20260905/protected_checkpoints/final_dynamic_multirate_root_fusion_v2_e73dcd8_step001599/seven_dataset_dynamic_multirate_resume1400_memory_safe_1600_8gpu_zero2_20260904.toml \
  --checkpoint /mnt/cpfs/sjc/DATA/EgoFound3R_archive/20260905/protected_checkpoints/final_dynamic_multirate_root_fusion_v2_e73dcd8_step001599/checkpoints/step_001599.pt \
  --video-path <segment_300f.mp4> \
  --output-dir <EGO_INFER_ROOT>/<dataset>/<segment_id> \
  --upsample-mano --global-stride 5 --device cuda:0
```

* `<segment_300f.mp4>`：把该段连续帧按 30 fps 拼成的视频（300 帧）。
* 后处理默认全开（`--root-z-smooth` / `--hand-anchor-filter` / `--hand-fill-missing` /
  `--hand-local-smooth` / `--root-uv-smooth` / `--hand-depth-scale` / `--overlap-depth-scale`），
  不要显式关闭；`--input-resize-mode keep_aspect` 是默认值。
* 产物目录结构（batch 脚本按这个约定查找）：

```
<EGO_INFER_ROOT>/<dataset>/<segment_id>/
  inference_output.pt
  provenance.json
  viz_inputs/ego_infer_300f.npz
  viz_inputs/ego_infer_300f_plus.npz     # 含 joint21_xyz_camera
```

### 3.3 staging + 渲染（CPU，一条命令）

```bash
python3 tools/batch_2d_render.py \
  --manifest ../overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914/selected_manifest.jsonl \
  --only arctic__00063-00362 \
  --ego-infer-root /mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_infer_8fc061a_batch \
  --src-dir /mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_src_20260917 \
  --out-root /mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_2d_batch \
  --parallel 1 --jobs 8
```

或者你已经手动 staging 好了（`<staged>/selection.json` + `0_ego.npz..4_ego.npz`），
也可以直接用锁定样式的封装脚本：

```bash
./tools/run_2d.sh --inputs-dir <staged segment> --out <out dir> --jobs 8
```

只想审一帧（约 10 秒，样式迭代时用）：

```bash
./tools/run_2d.sh --inputs-dir <staged segment> --out <out dir> --stages fig2_frame --cell2d 900
```

---

## 4. 批量渲染

```bash
cd /mnt/workspace/sjc/paper_viz_2d/visualization/paper_viz

python3 tools/batch_2d_render.py \
  --manifest ../overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914/selected_manifest.jsonl \
  --ego-infer-root /mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_infer_8fc061a_batch \
  --src-dir      /mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_src_20260917 \
  --out-root     /mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_2d_batch \
  --parallel 6 --jobs 8
```

* `--parallel N`：同时渲染 N 段；`--jobs M`：每段内部用 M 个 worker 渲染视频帧。
  经验值：单机 160 核时 `N×M ≤ 96` 比较稳（例如 6×8 或 8×8）。
* **断点续跑**：已有 `report.json` 的段会跳过（状态 `skipped_existing`）；`--force` 强制重渲。
* 状态与日志：

```
<out-root>/batch_summary.json           # 每段状态、耗时、帧数
<out-root>/_logs/<segment_id>.log       # staging + 渲染的完整输出
<out-root>/_staged/<segment_id>/        # staging 出来的逐窗输入
<out-root>/<segment_id>/{fig2_2d_matrix.png,panels_2d/,video2_2d_matrix.mp4,report.json}
```

* 常见状态：`rendered` / `skipped_existing` / `missing_ego_infer`（该段还没跑推理）/
  `rendered_frame_count_mismatch` / `failed`。
* 只想看会跑哪些段、不执行：`--dry-run`；只跑某些数据集：`--datasets arctic h2o`；
  只跑前 N 段：`--limit 3`；只跑指定段或 cache：`--only <segment_id 或 cache_id>`。

### 批量前置检查（建议每次先跑）

```bash
python3 tools/relay_sources.py --dataset arctic \
  --manifest ../overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914/selected_manifest.jsonl \
  --out /mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_src_20260917 --check
```

输出里的 `missing` 就是还缺的 baseline 源（通常因为 `/mnt/oss` 掉挂载）。

---

## 5. 样式合同（锁定，改动需用户确认）

`tools/run_2d.sh` 已经把下面这套参数写死；批量脚本调用的是 `paper_viz.cli` 的默认值，
两者一致（默认值即定稿值）：

```text
--palette reference              # 8fc061a viser/训练监控配色
--signal-style face              # 面 wash + 点线在上
--face-alpha 0.30                # 信号列面 wash
--face-occluded-factor 0.35      # 被遮挡面 = 0.30×0.35
--geometry-brightness 1.60       # 左右手配色提亮（保持色相）
--signal-brightness 1.15
--geometry-line-width 1          # 几何列与信号列同款：半径 1 点 / 1 px 线
--geometry-dot-radius 1
--geometry-face-alpha 0.30
--geometry-occluded-alpha 0.45   # 被遮挡点线照画，只降透明度
--geometry-stroke-lighten 0.65   # 点线比自己的面 wash 更亮（混白 65%）
--rows 5 --cell2d 720 --cell2d-video 360 --fps 30
```

行为约定（改代码时不要破坏）：

* 几何列 = 信号列同一套渲染路径，只是颜色换成左右手常量色（左蓝 `(23,135,255)` /
  右橙 `(255,114,3)` 提亮后；点线混白后 ≈ 左 `(174,213,255)` / 右 `(255,206,167)`）。
* 一条边的两端点取值不同时，**在中点切开、两半各用自己端点的颜色**（mmpose 的
  “边随端点变色”）；不要退回“两端取平均”，那会把 visibility/contact 的类别边界糊成暗色。
* joint 骨架（左黄 / 右品红 + 深色描边）叠在网格最上层；只有物体或另一只手真的挡在前
  面时才降透明度。
* 几何列 `with_object=False`：既不参与物体遮挡，也不做物体背景提亮。

需要新样式时：改 `paper_viz/style.py` 的默认值或加 CLI 旋钮，**并同步更新本文档与
`tools/run_2d.sh`**，然后重跑至少一帧给用户确认。

---

## 6. 验收清单

```bash
jq . <out>/<segment>/report.json        # 应有 video2_frames == 300
ffprobe -v error -select_streams v:0 \
  -show_entries stream=nb_frames,width,height,r_frame_rate -of default=nw=1 \
  <out>/<segment>/video2_2d_matrix.mp4  # 300 帧 / 5746×596 / 30 fps
ls <out>/<segment>/panels_2d | wc -l    # 15
```

人工抽查：Dyn-HaMR 列在未注册窗口必须是 `unavailable` 瓦片；几何列的背面点线仍在
（只是更淡）；visibility/contact 的类别边界应为“半绿半红 / 半红半蓝”。

---

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `missing_ego_infer` | 该段还没跑 8fc061a 推理，或目录不符合 `<ego-infer-root>/<dataset>/<segment_id>/` 约定 |
| `relay_sources.py` 报 `missing` | `/mnt/oss` 死挂载；用 `~/bin/ossutil2` 拉数据后重跑 |
| 视频/产物丢失 | 别把输入或产物放 `/tmp`；qingcang 重启会清空 |
| `numba` 导入失败 | 用了 conda 环境；改用 `/usr/local/bin/python3` |
| 段跑得慢 | 提高 `--jobs`（视频帧并行）或 `--parallel`（段并行）；单段 8 workers 约 44 s |
| 图里某个方法整列灰 | 该窗口没有这个方法注册的预测（例如 Dyn-HaMR 4/5 窗口）——不要补值 |

---

## 8. 性能参考（ARCTIC 冒烟段，5 行大图 + 15 panel + 300 帧视频）

| 配置 | 耗时 | 备注 |
|---|---|---|
| 单进程 | ~7 min | 15 列 / 帧 ≈ 0.75 s |
| `--jobs 8` | **44 s** | 视频帧 fork 并行，输出逐帧一致（md5 相同） |
| 单帧审阅 `--stages fig2_frame` | ~10 s | 样式确认用 |

帧并行的前提是“帧之间独立”，所以 `--jobs` 只影响视频行；大图/panel 仍是串行。
所有优化都做过逐像素校验：单帧与优化前 `max diff = 0`。

---

## 9. 代码结构

```
visualization/paper_viz/
  paper_viz/
    cli.py        入口：--stages fig2|fig2_frame|all、--jobs、全部样式旋钮
    render2d.py   2D 列渲染：Frame2D（RGB/K/z-buffer 缓存）、signal_cell（点线面）、
                  geometry_cell（= 信号路径 + 常量色）、draw_joints、draw_contour
    raster.py     numba z-buffer 光栅化 + 派生 visibility / distance / contact
    style.py      调色板、提亮、TURBO LUT、不可用瓦片、字体
    inputs.py     逐窗 npz / RGB / 几何加载，native→GT world 对齐
    joint_order.py PAD-Hand / ReViV4D 的 21 关节置换（勿改回）
    layout.py     网格拼接、标签、页眉页脚
    video.py      ffmpeg libx264 写入
    render3d.py / scene.py   3D 多视角渲染（当前为 matplotlib 版，待按 8fc061a 的
                  pytorch3d 脚本重做，见交接文档第 9 节）
  tools/
    relay_sources.py     冻结 GT/baseline → 逐窗 npz（本文档 3.1）
    stage_ego_windows.py 权威 Ego 推理产物 → 逐窗输入（本文档 3.3 内部调用）
    batch_2d_render.py   批量编排（本文档第 4 节）
    run_2d.sh            单段 + 锁定样式封装（本文档第 3.3 节）
    profile_2d_frames.py 逐阶段性能剖析（排查慢在哪）
  USAGE_2D.md    本文档
```

完整背景、数据源核实过程、样式迭代历史见
`Visualization/HANDOFF_MULTIMETHOD_VIZ_20260917.md` 与
`Visualization/HANDOFF_2D_VIZ_STYLE_20260917.md`。
