# 3D world-space 可视化 · 使用文档

版本：2026-09-18（样式已由用户确认；性能已优化到该渲染器的实测下限）
适用对象：后续接手渲染的任何同学 / agent。

这份文档 + `visualization/paper_viz/` 目录，就足以在服务器上批量产出**3D 世界坐标系**的
对比大图、单视角 panel 与视频。**改样式前请先确认**（第 5 节列出锁定参数）。

---

## 0. 产物是什么

对每一段 300 帧（5 × 60 帧窗口）的片段，一次批量运行产出三样：

| 产物 | 内容 | 规格（默认） |
|---|---|---|
| `fig1_3d_summary.png` | 世界系多视角矩阵图：**行 = 视角、列 = Input RGB + 8 方法** | cell 320（默认），9 列 × 5 行 |
| `panels_3d/<method>/<view>.png` | 每方法每视角单独一张（方便手动拼进论文） | 2048² 或 `--panel-size` 指定 |
| `video1_3d_matrix.mp4` | 与图片同版式的视频，30 fps × 300 帧 | cell 192（默认）→ 1872×1112 |

**列合同（左→右）**：

```
Input RGB | WiLoR | PAD-Hand | EgoForce | Dyn-HaMR | HaWoR | ReViV4D | EgoFound3R | GT
```

（baseline 在前，EgoFound3R 紧邻 GT，与 2D 的列序一致。）

**行 = 视角（5 个）**：

```
front | right | back | left        ← 从地面平台四条边的斜上方看（仰角 40°）
top                                ← 正上方俯视（85°），出图后逆时针旋转 90°
```

---

## 1. 运行环境（3D 与 2D 不同，务必注意）

| 依赖 | 要求 |
|---|---|
| Python | **`/mnt/workspace/sjc/envs/egofound3r/bin/python`**（torch 2.4.1+cu124 / pytorch3d 0.7.8 / CUDA 可用） |
| `t3drender` | 私有包，PyPI 上没有；用 `pip install --no-deps git+https://github.com/WenjiaWang0312/torch3d_render.git` 装（**别加 --no-deps 之外的参数，以免动到已装好的 torch/pytorch3d**） |
| 参考渲染器 | `PYTHONPATH` 需包含 8fc061a worktree：`/mnt/workspace/sjc/EgoFound3R_viz_8fc061a_20260917`（提供 `egohandmetric_prompt.inference_multiview`：`HandMultiviewRenderer` / `mesh_parts` / `camera_overlay_parts` / `_box` / `_tube_part`） |
| `ffmpeg` | 在 PATH 中（帧序列 → mp4） |
| ⚠️ 不要用 2D 的 numba 环境 | `/usr/local/bin/python3` 没有 torch/pytorch3d |

### 数据前置条件

与 2D 完全相同（同一套 relay 出来的逐窗 npz、staged Ego 输入、prepared RGB/物体几何、HaWoR 索引、195→778 映射）。
若还没准备，先按 `USAGE_2D.md` 第 3.1/3.2 节做 relay 与 staging。

---

## 2. 快速开始（单段）

```bash
ssh qingcang-0
cd /mnt/workspace/sjc/paper_viz_2d/visualization/paper_viz     # 或本地仓库同路径

python3 tools/batch_3d_render.py \
  --staged-root /mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_2d_smoke/_staged \
  --out-root    /mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_3d_delivery \
  --src-dir     /mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_src_20260917 \
  --prepared-root /mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_strict_arctic_20260820T000131Z/window_inputs/arctic \
  --hawor-root  /mnt/workspace/sjc/DATA/eval_artifacts/hawor_native_camera_repair_20260915_v7_5000/datasets/arctic \
  --hawor-index /mnt/workspace/sjc/DATA/eval_artifacts/hawor_native_camera_repair_20260915_v7_5000/datasets/arctic/predictions.jsonl \
  --contact-root /mnt/cpfs/sjc/eval_artifacts/result3_completion_20260910_v1 \
  --mapping     /mnt/workspace/sjc/EgoFound3R_viz_8fc061a_20260917/egohandmetric_prompt/data/mano_upsampling/mano_195_to_778.npz \
  --stages summary video --devices cuda:0 cuda:1 cuda:2 cuda:3 --chunks 4 \
  --cell 320 --panel-size 2048 --keyframes 5
```

只想改样式试一帧（不动视频）：

```bash
python3 tools/render_3d_summary.py <同样的路径参数> --out /tmp/pv3d_try \
  --keyframes 5 --cell 320 --camera-overlay show
```

---

## 3. 批量（推荐路径）

把每一段（staged 目录）放进一个根目录，然后：

```bash
python3 tools/batch_3d_render.py \
  --staged-root <每段一个子目录> --out-root <输出根> \
  --src-dir ... --prepared-root ... --hawor-root ... --hawor-index ... \
  --contact-root ... --mapping ... \
  --stages summary video \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  --chunks 4 --cell 192 --panel-size 1024 --keyframes 5
```

| 参数 | 含义 / 建议 |
|---|---|
| `--stages summary video` | 图片与视频一起出；只想出图就写 `--stages summary` |
| `--devices` | 每张卡一个常驻 worker；卡越多越快（线性） |
| `--chunks` | 每段视频切成几个帧块；一般取 `--devices` 的数量 |
| `--cell` / `--panel-size` | 视频/大图的 cell、panel 边长；panel 2048² 会让每段多花约 1-2 min |
| `--keyframes` | 大图里叠加多少个时间样本（默认 5，与 RGB 列一一对应） |
| `--camera-overlay show|hide` | 是否画标定相机的轨迹与视锥（hide 时取景更近、手更大） |
| `--force` / `--dry-run` / `--limit N` | 强制重跑 / 只看任务计划 / 只跑前 N 段 |

**断点续跑**：已有 `fig1_3d_summary.png` / `video1_3d_matrix.mp4` 的任务自动跳过；
每次运行写 `<out-root>/batch_summary.json`（每段状态、帧数、panel 数）。

---

## 4. 性能与硬件（实测）

| 指标 | 数值 |
|---|---|
| 单帧矩阵（cell 192 / ss1，单卡） | **≈1.48 s**（优化前 5.8 s，3.9×） |
| 300 帧视频 | 单卡 ≈7.4 min；**4 卡 ≈2.4 min** |
| 单段完整产物（大图 + 40 panel + 视频，4 卡） | **≈2.8 min**（cell 192 / panel 1024） |
| 104 段批量 | 4 卡 ≈4.9 h；**8 卡 ≈2.5 h** |
| 画质选项 | `--bin-size 64`（默认，肉眼无差、光栅化 1.66×）；`--bin-size 0` 回到逐位一致；`--no-batch-cells` 回到逐格渲染（调试用） |

已做的无损/近无损优化：批量单元格渲染（修掉 pytorch3d `pix_to_face` 跨 batch 累计面 id 的坑）、
零拷贝上传、`bin_size=64`、RGB 预处理后台线程、帧区间分片、常驻 worker + 工作窃取、
`torch.no_grad()`、相机标注每帧只构建一次。**实测每格耗时与几何量无关**（只画 264 面的视锥
1.44 s vs 画 3.4k 面的手 1.37 s），所以在这个渲染器 + 该分辨率下已到下限；再快只能加卡或减小 cell。

---

## 5. 样式合同（锁定）

| 项 | 设定 |
|---|---|
| 列 | `Input RGB` + 8 方法（WiLoR / PAD-Hand / EgoForce / Dyn-HaMR / HaWoR / ReViV4D / EgoFound3R / GT） |
| 行 | front / right / back / left（40° 斜视）+ top（85° 俯视，**逆时针转 90°**，开关 `TOP_VIEW_ROT90_CCW`） |
| 手部配色 | 参考 8fc061a 的 Morandi 左右手色 + 时间渐变 `SIDE_LIGHT→SIDE_DARK` |
| ReViV4D | **只画 21 关节骨架**（细骨管 + 关节块），且关节顺序必须过 `paper_viz.joint_order.joints_in_gt_order("reviv4d", ...)` |
| 世界系 | 8 方法统一到 GT world（标定 c2w / native·pred 逐窗刚体对齐），再按"相机朝上"做水平化旋转 |
| 相机口径 | **每个方法优先用自己的预测外参**：Dyn-HaMR / HaWoR / ReViV4D / EgoFound3R 画**绿色 pred 相机**；GT / WiLoR / PAD-Hand / EgoForce 不预测相机，用**橙色标定相机**。内参同理优先自己预测（目前只有 EgoFound3R 有 `intrinsics_pred`，fx≈2075 vs 标定 2319），其余回落标定 K |
| 轨迹 | 视频：只画**当前帧及过去**（累积，不画未来）；图片：画整段完整轨迹。两者都与手部一起做逐窗对齐，手—相机相对关系保持不变 |
| 地面 | 平台由**整条手部轨迹** AABB 驱动（最低点下方固定 gap，footprint ×1.55） |
| 取景 | 手部轨迹 4–96 分位盒反解距离：`--camera-overlay show` margin 0.82、`hide` 0.72（`--fit-margin` 可覆盖） |
| 标签 | TTF（DejaVuSans，回退随包 PatrickHand），字号 0.115×cell，行列名**居中** |
| 图例 | `Hand colour`(Left/Right) + `Time`(Earlier→Later) + `Camera`(pred/gt)，hairline 分隔 |

---

## 6. 验收清单

```bash
# 世界系契约（手 / 相机锥 / 轨迹必须同系）——渲染前必跑，违反则退出码 1
python3 tools/check_3d_world_frame.py --staged-root <staged> \
  --src-dir ... --prepared-root ... --hawor-root ... --hawor-index ... \
  --contact-root ... --mapping ... --device cpu
ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,width,height \
  -of default=nw=1 <out>/<segment>/video1_3d_matrix.mp4      # 300 帧
ls <out>/<segment>/panels_3d/*/*.png | wc -l                 # 8 方法 × 5 视角 = 40
python3 -c "from PIL import Image;print(Image.open('<out>/<segment>/fig1_3d_summary.png').size)"
```

人工抽查：`top` 行应为俯视且已逆时针旋转；ReViV4D 列是骨架不是 mesh；Dyn-HaMR 在无注册预测的
窗口应为空（不要补值）；`--camera-overlay hide` 版本里不应出现相机锥/轨迹。

### 6.1 世界坐标系契约（硬约束）

手部几何、每方法的相机 bundle（视锥 + 坐标轴标 + 轨迹管）、标定显示相机，三者必须经过
**同一个** `level_rotation` 刚体变换后一起送入渲染器；只转其中一部分就会把手和相机画到两个世界
（2026-09-18 之前正是如此：视锥被画到离手 ~2.7 m 处、光轴偏 43°）。代码上现在有两道锁：

1. `paper_viz/sequences3d.py::level_in_place(store, rotation)` 是**唯一**的坐标系变换入口，
   三个 3D 工具都调它（不再各自写旋转循环）；
2. `paper_viz/sequences3d.py::assert_camera_bundle_frame(store, camera)` 每次建场景都会执行：
   利用"逐窗刚体重定位让每个方法的首帧相机与标定相机重合"这一恒等式，任何一处漏转/多转都会
   在窗口起点暴露成米级偏移并**直接抛错终止渲染**（不是靠肉眼看图）。

`tools/check_3d_world_frame.py` 是它的命令行外壳，可对任意 staged 段输出逐方法报告
（相机-手距离、光轴夹角、图像锥角、窗口锚点误差），适合放进批量前的 preflight。

---

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `ModelNotFound: t3drender` | 按第 1 节用 `--no-deps` 安装 |
| `Cannot re-initialize CUDA in forked subprocess` | 多卡必须走 spawn（脚本已如此）；**父进程不要在 spawn 前建 CUDA 上下文** |
| `TypeError: 'NoneType' object is not callable`（旧版本） | 曾是 `_scene_frame` 参数遮蔽 `rgb_tile()` 的 bug，已修（`b147715`） |
| 视频只有很少几帧 / `video_incomplete` | 帧数不足说明 worker 中途失败；看 stderr 与 `batch_summary.json`，`--force` 重跑该段 |
| `/mnt/oss` 掉挂载 | 用 `~/bin/ossutil2` 拉数据，或改用已 relay 到共享盘的 `paper_viz_src_20260917` |
| 显存不足 | 降 `--panel-size` 或 `--cell`；单卡只跑 `--devices cuda:0` |

---

## 8. 代码结构

```
visualization/paper_viz/
  paper_viz/
    sequences3d.py   8 方法的 world 适配（标定/native/pred 相机对齐、缺窗补 NaN、ReViV4D 关节置换）
    batch_render.py  批量单元格渲染（一次调用出 8 个方法；零拷贝上传；bin_size）
  tools/
    render_3d_summary.py  单段大图 + panel（样式调试入口）
    render_3d_video.py    单段视频（帧区间分片 / 方法分片；`--bin-size`、`--no-batch-cells`）
    batch_3d_render.py    批量：任务队列 = summary + video 帧块，常驻 worker，断点续跑  ← 常用
    batch_3d_video.py     仅视频的批量（更早的版本，保留）
    check_3d_world_frame.py  预检：手 / 相机锥 / 轨迹是否同处一个世界系（违反退出码 1）
    relay_sources.py      （2D/3D 共用）冻结 GT/baseline → 逐窗 npz
    stage_ego_windows.py  （2D/3D 共用）权威 Ego 推理产物 → 逐窗输入
  USAGE_2D.md            2D 使用文档
  USAGE_3D.md            本文档
```

参考代码：8fc061a worktree 的 `export_hand_multiview.py` + `egohandmetric_prompt/inference_multiview.py`；
EgoGrasp `eval_scripts-new`（样式与视角参考）已存档到 `paper_viz_3d_ref/` 与本地
`Visualization/3d_reference_20260917/`。
