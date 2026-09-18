# 3D world-space 可视化 · 使用文档

版本：2026-09-18 **r3**（相机锥/轨迹世界系修复 + 渲染时自动断言 + 大图 RGB 列 5 帧；
tag **`viz-3d-rgb5-20260918`**，其上为 `viz-3d-fixed-20260918` / `viz-3d-cams2-20260918` / `viz-3d-cams-20260918` / `viz-3d-first-20260918`）。
适用对象：后续接手渲染的任何同学 / agent。

这份文档 + `visualization/paper_viz/` 目录，就足以在服务器上批量产出**3D 世界坐标系**的
对比大图、单视角 panel 与视频。**改样式前请先确认**（第 7 节列出锁定参数），
**动相机显示逻辑前请先读第 5 节**（那里有会直接让渲染报错的硬约束）。

---

## 0. 产物是什么

对每一段（默认 300 帧 = 5 × 60 帧窗口）片段，一次运行可产出：

| 产物 | 内容 | 规格（默认） |
|---|---|---|
| `fig1_3d_summary.png` | 世界系多视角矩阵图：**行 = 视角、列 = Input RGB + 8 方法** | cell 320，9 列 × 5 行 |
| `panels_3d/<method>/<view>.png` | 每方法每视角单独一张（方便手动拼进论文） | `--panel-size` 指定，**`0` 表示不出 panel** |
| `video1_3d_matrix.mp4` | 与图片同版式的视频，30 fps × 帧数（默认 300） | 与 `--cell` 一致 |

**Input RGB 列的口径（大图 vs 视频不同，别搞混）**：

- **大图**：行 = 视角，**第 r 行显示第 r 个时间样本**——默认 5 个样本均匀取自整段
  `np.rint(np.linspace(0, T-1, keyframes))`，300 帧段即 **0 / 75 / 150 / 224 / 299**；
  每格左上角标注 `frame NNN · T.TTs`，方便对着视频找时刻（因此大图里 5 行 RGB **是 5 张不同的图**）。
- **视频**：RGB 格是**当前帧**，5 行相同、逐帧变化（标题里也写着当前 frame）。
- 要求 `--keyframes` 与视角数相同（默认都是 5）；两者不等时按 `row % len(views)` 落格，多出的样本会覆盖。

**列合同（左→右）**：

```
Input RGB | WiLoR | PAD-Hand | EgoForce | Dyn-HaMR | HaWoR | ReViV4D | EgoFound3R | GT
```

**行 = 视角（5 个）**：

```
front | right | back | left        ← 从地面平台四条边的斜上方看（仰角 40°）
top                                ← 正上方俯视（85°），出图后逆时针旋转 90°
```

### 0.1 三个版式变体（按论文需要选一个）

| 变体 | 命令要点 | 取景 margin | 用途 / 现象 |
|---|---|---|---|
| **显示相机**（默认） | `--camera-overlay show` | 0.82 | 每行画该方法自己的相机锥 + 轨迹（视频累积到当前帧、图片画整段）。手部尺寸正常，个别行列的视锥会贴到 cell 边被切一点 |
| **不显示相机** | `--camera-overlay hide` | 0.72 | hand-only，画面里没有任何相机几何，手最大、最干净 |
| **相机完整入画** | `--camera-overlay show --fit-with-cameras` | 1.02 | 把 8 行相机中心一并纳入拟合，保证视锥完整在格子内（手会小一些） |

2026-09-18 在冒烟段 `arctic__00063-00362`（`--cell 320 --panel-size 0`）出的交付物：

| 变体 | 图片 md5（r3：含 RGB 5 帧） | 视频 md5（r3 未改动） |
|---|---|---|
| 显示相机 | `0e320c727189fb9a7f70377792b5082b` | `0ec664a37e3cd2b9994ad98a07865b04` |
| 不显示相机 | `7665eedca448dccbc7673df20f4a5e5a` | `76a2d9758b769833d10fe4ccc1a92537` |
| 相机完整入画 | `ddbcc86187376123fbc53350edf81e06` | `ede65fa84766c187497e6b526d021cc2` |

（每个变体目录里都是同构命名：`fig1_3d_summary.png` + `video1_3d_matrix.mp4`。）

---

## 1. 运行环境（3D 与 2D 不同，务必注意）

| 依赖 | 要求 |
|---|---|
| Python | **`/mnt/workspace/sjc/envs/egofound3r/bin/python`**（torch 2.4.1+cu124 / pytorch3d 0.7.8 / CUDA 可用） |
| `t3drender` | 私有包，PyPI 上没有；`pip install --no-deps git+https://github.com/WenjiaWang0312/torch3d_render.git`（**务必 `--no-deps`**，否则会动到装好的 torch/pytorch3d） |
| 参考渲染器 | 需要 8fc061a worktree：`/mnt/workspace/sjc/EgoFound3R_viz_8fc061a_20260917`（提供 `egohandmetric_prompt.inference_multiview`：`HandMultiviewRenderer` / `mesh_parts` / `camera_overlay_parts` / `CAMERA_COLORS` / `_tube_part`）。`tools/*3d*.py` 已自带 `sys.path.insert`，只要该目录存在即可 |
| `ffmpeg` | 在 PATH 中（帧序列 → mp4） |
| ⚠️ 不要用 2D 的 numba 环境 | `/usr/local/bin/python3` 没有 torch/pytorch3d |

### 数据前置条件

与 2D 完全相同（同一套 relay 出来的逐窗 npz、staged Ego 输入、prepared RGB/物体几何、
HaWoR 索引、195→778 映射）。若还没准备，先按 `USAGE_2D.md` 第 3.1/3.2 节做 relay 与 staging。

批量前请确认这两件事（否则 3D 会静默少方法）：

1. 每个 staged 段目录里有 `selection.json` + `{i}_ego.npz`（`i` = 窗口序号）；
2. 该段的每个窗口在 `--src-dir` 下有 `gt.npz` 等 baseline npz，HaWoR 在该数据集目录下有 `predictions.npz`。

---

## 2. 快速开始（单段）

```bash
ssh qingcang-0
cd /mnt/workspace/sjc/paper_viz_2d/visualization/paper_viz     # 或本地仓库同路径

# 完整（大图 + 40 panel + 视频，4 卡）
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

# 快速版（只要大图 + 视频，2 卡，约 7.5 min/段）
python3 tools/batch_3d_render.py <同上路径参数> \
  --stages summary video --devices cuda:0 cuda:1 --chunks 4 \
  --cell 320 --panel-size 0 --keyframes 5 --camera-overlay show
```

三个变体只差一个开关：

```bash
... --camera-overlay show                      # 显示相机（默认）
... --camera-overlay hide                      # 不显示相机
... --camera-overlay show --fit-with-cameras   # 相机完整入画
```

只调样式、只想看一张图（不写视频）：

```bash
python3 tools/render_3d_summary.py <同样的路径参数> --out /tmp/pv3d_try \
  --keyframes 5 --cell 320 --camera-overlay show --no-panels
```

只出视频（支持帧区间分片 / 方法分片）：

```bash
python3 tools/render_3d_video.py <同样的路径参数> --inputs-dir <段目录> --out /tmp/pv3d_vid \
  --devices cuda:0 cuda:1 --shard-mode frames
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
| `--stages summary video` | 图片与视频一起出；只出图写 `--stages summary` |
| `--devices` | 每张卡一个常驻 worker；卡越多越快（近似线性） |
| `--chunks` | 每段视频切成几个帧块；一般取 `--devices` 的数量 |
| `--cell` / `--panel-size` | 视频与大图的 cell、panel 边长；`--panel-size 0` 关闭 panel（省 1–2 min/段） |
| `--keyframes` | 大图里叠加多少个时间样本（默认 5，与 RGB 列一一对应） |
| `--camera-overlay show\|hide` | 画不画相机锥与轨迹（hide 时取景更近、手更大） |
| `--fit-with-cameras` | 把各行相机中心纳入拟合，保证视锥完整入画（手变小） |
| `--fit-margin` | 覆盖默认 margin（show 0.82 / hide 0.72 / fit-with-cameras 1.02；越小越近） |
| `--camera-scale` | 视锥尺度（世界单位，默认 0.12） |
| `--bin-size` | 光栅化 bin 网格，默认 64（肉眼无差、快 1.66×）；`0` 回到逐位一致 |
| `--force` / `--dry-run` / `--limit N` | 强制重跑 / 只看任务计划 / 只跑前 N 段 |

**断点续跑**：已有 `fig1_3d_summary.png` / `video1_3d_matrix.mp4` 的任务自动跳过；
每次运行写 `<out-root>/batch_summary.json`（每段状态、帧数、panel 数）。

---

## 4. 相机口径合同（哪个方法用 pred、哪个用 gt）

**外参**（实现见 `paper_viz/sequences3d.py::_window_world`）：

| 列 | 外参来源 | 数据里的判据 | 图例颜色 |
|---|---|---|---|
| WiLoR / PAD-Hand / EgoForce | **标定（GT）相机** | 这些 npz 只有 `hand_vertices_camera`，**没有 `camera_c2w`** | 橙 `gt` |
| GT | **标定相机**（它本身就是标定源） | `gt.npz` 的 `camera_c2w` + `intrinsics` | 橙 `gt` |
| Dyn-HaMR / HaWoR | **自己预测**（native SLAM 世界） | `hand_vertices_world` + `camera_c2w` + `camera_valid` | 绿 `pred` |
| ReViV4D | **自己预测** | `hand_joints_camera` + `camera_c2w`（关节顺序需 `joints_in_gt_order("reviv4d", …)`） | 绿 `pred` |
| EgoFound3R | **自己预测** | `hand_vertices_camera` + `camera_c2w` + `intrinsics_pred` + `input_affine` | 绿 `pred` |

**内参**：只有 EgoFound3R 用自己的 `intrinsics_pred`（经 keep_aspect 仿射逆映射，本段实测
fx 均值 2090、逐帧 1996–2223），其余全部回落标定 K（fx 2319.9 / fy 2367.2，图 2000×2800）。

**逐窗刚体重定位（必读的三个推论）**：native / pred 世界的方法，每个 60 帧窗口都用
`T = gt_c2w[s] · inv(cam_c2w[s])` 把**手和相机一起**搬到 GT world。

1. 窗口首帧该方法的相机与标定相机**严格重合**，之后才逐渐漂移——这正是第 5 节断言的依据；
2. 因此"pred 相机"展示的是**窗口内的相机漂移**，不是绝对位姿差：ARCTIC 这类固定机位段实测只有
   0.5–4.3 cm，8 行视锥看上去几乎重合；要看出差别得用自移动机位的数据集；
3. 某方法只在部分窗口有预测时（例如 Dyn-HaMR 1/5 窗口），它的**手和相机必须落在同一段帧区间**，
   空窗口在原地补 NaN（不要把它挪到片段开头——见第 5 节坑 ②）。

---

## 5. 世界系契约（硬约束，动相机显示前必读）

> **手部几何、每方法的相机 bundle（视锥 + 坐标轴标 + 轨迹管）、标定显示相机，必须经过同一个
> `level_rotation` 刚体变换后一起送入渲染器。**

数据集世界来自 `camera_c2w`，+y 朝下；渲染器假设 +y 朝上，所以有一个水平化旋转
（ARCTIC 冒烟段实测 **151.15°**）。只转其中一部分，就会出现"手在一处、相机在另一处"的错误画面。

### 5.1 三道锁（代码已保证，不要绕过）

| 机制 | 位置 | 作用 |
|---|---|---|
| `level_in_place(store, rotation)` | `paper_viz/sequences3d.py` | **唯一**的坐标系变换入口：手、关节、相机 bundle 一起转；三个 3D 工具都调它 |
| `assert_camera_bundle_frame(store, camera)` | 同上，被 `batch_3d_video.scene_state` 调用 | 每次建场景即断言，违反**直接抛错终止渲染** |
| `tools/check_3d_world_frame.py` | 预检 CLI | 逐段输出逐方法报告（相机-手距离 / 光轴夹角 / 图像锥角 / 窗口锚点误差），违反退出码 1 |

断言依据的恒等式：逐窗重定位让每个方法在**自己窗口首帧**与标定相机重合，所以任何漏转/多转
都会在窗口起点暴露成米级偏移。

### 5.2 已经踩过的两个坑（务必不要再引入）

| 坑 | 症状 | 断言给什么 |
|---|---|---|
| 相机 bundle 没做水平化 | 视锥/轨迹被画到离手约 2.7 m 处，光轴偏 43°（正常手在视场内是 5–9°） | `window-start offset … 2.68 m` |
| 缺窗方法的相机被挪到片段开头 | Dyn-HaMR（1/5 窗口）的视锥出现在 frames 0–59，而它的手在 120–179 | `Dyn-HaMR: … 0.0167 m` |

新增"画相机 / 画轨迹"逻辑时的检查清单：

1. 相机位姿是否来自 `store[method]["camera"]["c2w"]`（已由 `level_in_place` 处理过）；
2. 是否用 `camera_frustum_vertices()` / `camera_overlay_parts()`（顶点=相机中心、开口沿 +Z 光轴）；
3. 是否按 `camera_valid` 门控（缺窗 / 无效帧不画）；
4. 跑一次 `tools/check_3d_world_frame.py`，锚点误差必须是 0（float 级）。

---

## 6. 性能与硬件（实测）

| 指标 | 数值 |
|---|---|
| 单帧矩阵（cell 192 / ss1，单卡） | **≈1.48 s**（优化前 5.8 s，3.9×） |
| 300 帧视频 | 单卡 ≈7.4 min；**4 卡 ≈2.4 min** |
| 单段完整产物（大图 + 40 panel + 视频，4 卡，cell 192 / panel 1024） | **≈2.8 min** |
| 单段（大图 + 视频，2 卡，cell 320 / 无 panel） | ≈7.4 min（2026-09-18 实测 440–455 s） |
| 104 段批量 | 4 卡 ≈4.9 h；**8 卡 ≈2.5 h** |
| 世界系预检（CPU） | 19 s / 段 |

已做的（近）无损优化：批量单元格渲染（修掉 pytorch3d `pix_to_face` 跨 batch 累计面 id 的坑）、
零拷贝上传、`bin_size=64`、RGB 预处理后台线程、帧区间分片、常驻 worker + 工作窃取、
`torch.no_grad()`、相机标注每帧只构建一次。**实测每格耗时与几何量无关**
（只画 264 面的视锥 1.44 s vs 画 3.4k 面的手 1.37 s），所以在这个渲染器 + 该分辨率下已到下限；
再快只能加卡或减小 cell。

---

## 7. 样式合同（锁定）

| 项 | 设定 |
|---|---|
| 列 | `Input RGB` + 8 方法（WiLoR / PAD-Hand / EgoForce / Dyn-HaMR / HaWoR / ReViV4D / EgoFound3R / GT） |
| Input RGB 列 | 大图：第 r 行 = 第 r 个时间样本（默认 0/75/150/224/299，格内标注帧号+时间）；视频：当前帧 |
| 行 | front / right / back / left（40° 斜视）+ top（85° 俯视，**逆时针转 90°**，开关 `TOP_VIEW_ROT90_CCW`） |
| 手部配色 | 参考 8fc061a 的 Morandi 左右手色 + 时间渐变 `SIDE_LIGHT→SIDE_DARK` |
| ReViV4D | **只画 21 关节骨架**（细骨管 + 关节块），关节顺序必须过 `paper_viz.joint_order.joints_in_gt_order("reviv4d", ...)` |
| 世界系 | 8 方法统一到 GT world（标定 c2w / native·pred 逐窗刚体对齐），再整体做水平化（第 5 节） |
| 相机口径 | 见第 4 节：Dyn-HaMR / HaWoR / ReViV4D / EgoFound3R = 绿 `pred`；GT / WiLoR / PAD-Hand / EgoForce = 橙 `gt` |
| 轨迹 | 视频只画**当前帧及过去**（累积，不画未来）；图片画整段。轨迹与视锥都来自同一个相机 bundle |
| 地面 | 平台由**整条手部轨迹** AABB 驱动（最低点下方固定 gap，footprint ×1.55） |
| 取景 | 手部轨迹 4–96 分位盒反解距离；margin 见第 0.1 节 |
| 标签 | TTF（DejaVuSans，回退随包 PatrickHand），字号 0.115×cell，行列名**居中** |
| 图例 | `Hand colour`(Left/Right) + `Time`(Earlier→Later) + `Camera`(pred/gt)；hide 版无 Camera 组 |

---

## 8. 交付验收清单

```bash
# ① 世界系契约（渲染前必跑；batch 渲染时也会自动断言，违反即中断）
python3 tools/check_3d_world_frame.py --staged-root <staged> \
  --src-dir ... --prepared-root ... --hawor-root ... --hawor-index ... \
  --contact-root ... --mapping ... --device cpu

# ② 视频规格
ffprobe -v error -select_streams v:0 \
  -show_entries stream=nb_frames,width,height,r_frame_rate,duration -of default=nw=1 \
  <out>/<segment>/video1_3d_matrix.mp4          # 期望 300 帧 / 3102×1788 / 30/1 / 10.0

# ③ panel 数量（如果出了 panel）
ls <out>/<segment>/panels_3d/*/*.png | wc -l     # 8 方法 × 5 视角 = 40

# ④ 大图尺寸
python3 -c "from PIL import Image;print(Image.open('<out>/<segment>/fig1_3d_summary.png').size)"
```

**验证方法学（重要）**：

- 比像素请比**无损 PNG 帧**（`<out>/<segment>/_frames/*.png`），**不要比 mp4**——x264 的 lookahead /
  码控与总帧数有关，同样输入在不同长度下会编出不同字节；
- 同一份代码、同样参数重复渲染，**光栅化边界上可能出现 1 个像素、1–55 灰阶的差异**，这是渲染器的
  非确定性，不是参数变化；判断"有没有改动"请用下面的像素 diff，而不是只看 md5 是否相等。

```bash
python3 - <<'PY'
import numpy as np
from PIL import Image
a = np.asarray(Image.open("old.png").convert("RGB")).astype(int)
b = np.asarray(Image.open("new.png").convert("RGB")).astype(int)
d = np.abs(a - b).max(2)
print("differing pixels:", int((d > 0).sum()), "max channel diff:", int(d.max()))
ys, xs = np.where(d > 0)
if len(ys):
    print("bbox y", int(ys.min()), int(ys.max()), "x", int(xs.min()), int(xs.max()))
PY
```

人工抽查：`top` 行已逆时针旋转；ReViV4D 是骨架不是 mesh；某方法在某窗口无预测时**手和相机一起为空**
（不要补值、也不要把相机挪到别的时段）；`hide` 版里没有任何相机几何与 Camera 图例。

---

## 9. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `ModelNotFound: t3drender` | 见第 1 节，用 `--no-deps` 安装 |
| `Cannot re-initialize CUDA in forked subprocess` | 多卡必须 spawn（脚本已如此）；**父进程不要在 spawn 前建 CUDA 上下文** |
| `RuntimeError: 3D world-frame contract violated …` | 相机锥/轨迹与手不在同一世界系（第 5 节）。报错文本会给出方法与偏移量；修好再渲染，不要"看图觉得没问题"就放过 |
| `TypeError: 'NoneType' object is not callable`（旧版本） | 曾是 `_scene_frame` 参数遮蔽 `rgb_tile()` 的 bug，已修（`b147715`） |
| 缺文件导致的 KeyError | 多半是 `selection.json` / `{i}_ego.npz` / baseline npz 不全，检查第 1 节的前置条件 |
| 视频帧数不足 / `video_incomplete` | worker 中途失败：看 stderr 与 `batch_summary.json`，`--force` 重跑该段 |
| 某方法整段空白 | 该段没有它的预测（如 Dyn-HaMR 只有 1/5 窗口），属预期；但同行相机也必须只在同一窗口出现 |
| `/mnt/oss` 掉挂载 | 用 `~/bin/ossutil2` 拉数据，或改用已 relay 到共享盘的 `paper_viz_src_20260917` |
| 显存不足 | 降 `--panel-size` 或 `--cell`；单卡只跑 `--devices cuda:0` |
| `/tmp/paper_viz_pkg` 丢失 | `/tmp` 重启即丢：按第 10.2 节两条 rsync 重部署 |

---

## 10. 代码结构与资产位置

### 10.1 仓库结构

```
visualization/paper_viz/
  paper_viz/
    sequences3d.py   8 方法的世界适配（相机 bundle、缺窗补 NaN、ReViV4D 关节置换）
                     + level_in_place() / camera_bundle_report() / assert_camera_bundle_frame()
    batch_render.py  批量单元格渲染（一次调用出 8 个方法；零拷贝上传；bin_size）
    inputs.py        npz / RGB / 几何读取，native_windows_in_gt_world() 逐窗重定位
    joint_order.py   ReViV4D 原生关节顺序 → GT 边表
  tools/
    render_3d_summary.py    单段大图 + panel（样式调试入口）
    render_3d_video.py      单段视频（帧区间分片 / 方法分片）
    batch_3d_render.py      批量：任务队列 = summary + video 帧块，常驻 worker，断点续跑  ← 常用
    batch_3d_video.py       仅视频的批量（更早版本，保留；scene_state() 被主批量复用）
    check_3d_world_frame.py 预检：手 / 相机锥 / 轨迹是否同处一个世界系（违反退出码 1）
    relay_sources.py        （2D/3D 共用）冻结 GT/baseline → 逐窗 npz
    stage_ego_windows.py    （2D/3D 共用）权威 Ego 推理产物 → 逐窗输入
  USAGE_2D.md / USAGE_3D.md
```

### 10.2 服务器路径与部署

| 路径 | 内容 |
|---|---|
| `/mnt/workspace/sjc/EgoFound3R-baselines` | 服务器上的 git 仓库（本地 `origin`） |
| `/mnt/workspace/sjc/paper_viz_2d` | **部署 worktree**（必须保持 detached，否则本地 push 会被 `receive.denyCurrentBranch` 拒） |
| `/tmp/paper_viz_pkg` | 实际被 import 的渲染包（**重启即丢**），渲染时在 `cd /tmp/paper_viz_pkg` 下运行 |
| `/mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_src_20260917` | 逐窗 GT/baseline npz（2D/3D 共用） |
| `/mnt/workspace/sjc/EgoFound3R_viz_8fc061a_20260917` | 模型 worktree（渲染器 + Ego 推理 + MANO 映射） |

同步与更新：

```bash
# 本地 → 服务器仓库
git push origin viz-paper-20260917
# 服务器部署 worktree 跟到最新
cd /mnt/workspace/sjc/paper_viz_2d
git fetch /mnt/workspace/sjc/EgoFound3R-baselines viz-paper-20260917
git checkout --detach FETCH_HEAD
# 干净重部署渲染包（两个源必须分开 rsync）
cd <repo>/visualization/paper_viz
rsync -a --delete --exclude '__pycache__' ./paper_viz/ qingcang-0:/tmp/paper_viz_pkg/paper_viz/
rsync -a --delete --exclude '__pycache__' ./tools/     qingcang-0:/tmp/paper_viz_pkg/tools/
```

### 10.3 本地备份与交付物

| 路径 | 内容 |
|---|---|
| `Visualization/3d_pipeline_20260918/` | 代码 + 文档 + 样式迭代截图（本 tag 的快照） |
| `Visualization/3d_camera_variants_20260918/` | 显示 / 不显示 / 完整入画三版交付物 + `MANIFEST_3d_camera_variants_20260918.md` |
| `Visualization/3d_reference_20260917/` | EgoGrasp 参考脚本与样式迭代截图 |

参考代码：8fc061a worktree 的 `export_hand_multiview.py` + `egohandmetric_prompt/inference_multiview.py`。

---

## 11. 变更记录

| 版本 / tag | 提交 | 内容 |
|---|---|---|
| `viz-3d-first-20260918` | `03e4cdf` … `286f691` | 3D 世界系对比第一版（summary + video） |
| `viz-3d-cams-20260918` | `b2d2a24` | 逐方法相机（pred/gt 配色）+ 视频累积轨迹 |
| `viz-3d-cams2-20260918` | `50061f9` | `--fit-with-cameras` 开关 |
| `viz-3d-fixed-20260918` | `cb806c3` → `ec9630a` → `6d4e624` → `9a8da7f` | ① 相机 bundle 补做水平化（原来视锥离手 2.7 m、光轴偏 43°）；② 缺窗方法的相机不再被挪到片段开头；③ `level_in_place` 统一入口 + 每次渲染的世界系断言 + `check_3d_world_frame.py` 预检；④ 本文档重写 |
| `viz-3d-rgb5-20260918` | `2b5c882` | 大图 Input RGB 列改为**每行一个均匀采样帧**（改前 5 行是同一帧、且是最后一帧）并在格内标注帧号/时间 |
| **`viz-3d-r3-20260918`（当前）** | 其上 doc-only 提交 | 刷新本文档的版本号与交付物 md5 表（代码与 `viz-3d-rgb5-20260918` 相同） |
