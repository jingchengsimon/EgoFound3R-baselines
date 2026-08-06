# egocentric_metrics

独立的 NumPy 优先 egocentric vision 评测 API。函数接受 NumPy 数组，也接受
已安装 PyTorch 时的 Tensor；Tensor 会在评测边界处 detach 到 CPU，不参与梯度计算。

## 坐标和形状

- 深度与三维坐标默认使用米。
- `mano_metrics` 的 `unit_scale` 默认是 `1000`，因此 MPJPE/PVE 输出毫米。
- `contact_sliding` 和 `contact_distance_metrics` 默认也输出毫米。
- 关键点/顶点序列形状为 `(T, J/V, 3)`；相机内参为 `(..., 3, 3)`，外参为 `(..., 4, 4)`。
- 所有函数都接受显式 `mask`；没有有效元素时返回 `NaN` 和/或 `valid_count=0`，不会静默返回零。

## 示例

```python
import numpy as np
from egocentric_metrics import (
    mano_metrics,
    vertex_metrics,
    keypoint_mpjpe,
    depth_metrics,
    joint_visibility_metrics,
    vertex_visibility_metrics,
    joint_contact_metrics,
    vertex_contact_metrics,
    joint_contact_distance_metrics,
    vertex_contact_distance_metrics,
)

pose = mano_metrics(pred_joints_m, gt_joints_m, root_index=0)
vertices = vertex_metrics(pred_vertices_m, gt_vertices_m)
keypoint_error = keypoint_mpjpe(pred_keypoints_m, gt_keypoints_m, alignment="procrustes")
depth = depth_metrics(pred_depth_m, gt_depth_m, mask=depth_valid)
joint_visibility = joint_visibility_metrics(
    predicted_joint_visibility_logits,
    gt_joint_visibility,
    prediction_type="logit",
)
vertex_visibility = vertex_visibility_metrics(pred_vertex_visibility, gt_vertex_visibility)
joint_contact = joint_contact_metrics(pred_joint_contact, gt_joint_contact)
vertex_contact = vertex_contact_metrics(pred_vertex_contact, gt_vertex_contact)
joint_distance = joint_contact_distance_metrics(pred_joint_contact_distance, gt_joint_contact_distance)
vertex_distance = vertex_contact_distance_metrics(pred_vertex_contact_distance, gt_vertex_contact_distance)
```

## 重要语义

`sequence_chunk_mpjpe(..., mode="first2")` 是原始 `compute_metric.py` 的
`wa2_mpjpe`：每个分块只用前两帧估计一个 Sim(3)。`mode="all"` 是原始
`waa_mpjpe`：每个分块用所有帧估计一个 Sim(3)。二者都不是逐帧
PA-MPJPE；逐帧相似变换由 `mano_metrics` 的 `pa_mpjpe` 计算。

`world_mpjpe` 是不做任何对齐的 W-MPJPE；`world_aligned_mpjpe(...,
mode="all")` 是 WA-MPJPE，与 `compute_global_metrics` 返回的
`wa_mpjpe` / `waa_mpjpe` 相同。`mode="first2"` 对应 `wa2_mpjpe`，用于
观察初始化之后的世界坐标漂移。

`jitter` 使用原脚本的三阶有限差分、`fps**3` 和除以 10。`rte` 使用固定尺度
刚体对齐，并以 GT 整段轨迹长度归一化；`percent=True` 时乘以 100。

本版本只提供核心 Python 函数，不包含数据文件读取、CLI、MANO 拟合或单位自动推断。

## 选择指标的运行器

`evaluate()` 是后续实验脚本应使用的统一入口。它只接收内存中的数组，文件读取
和数据集字段适配保持在调用方代码中。

```python
from egocentric_metrics import available_metrics, evaluate

implemented_names = available_metrics(include_placeholders=False)
results = evaluate(
    inputs={
        "pred_joints": pred_joints_m,
        "gt_joints": gt_joints_m,
        "pred_vertices": pred_vertices_m,
        "gt_vertices": gt_vertices_m,
        "pred_depth": pred_depth_m,
        "gt_depth": gt_depth_m,
    },
    metrics=["w_mpjpe", "wa_mpjpe", "pve", "vertex_auc", "abs_rel"],
    config={
        "unit_scale": 1000.0,
        "chunk_length": 100,
        "fps": 30.0,
        "auc_max_threshold": 50.0,
        "auc_unit_scale": 1000.0,
    },
)
```

每一个条目都返回 `{"status": "implemented", "value": ...}`。需要外部协议
或模型的指标返回 `{"status": "placeholder", "value": None, "reason": ...}`，
不会被伪造成零或 NaN。

可选名称按类别如下：

- 手部/时序：`mpjpe`、`pa_mpjpe`、`root_relative_mpjpe`、`pve`、`pa_pve`、
  `root_relative_pve`、`w_mpjpe`、`wa_mpjpe`、`wa2_mpjpe`、`joint_auc`、
  `vertex_auc`、`rte`、`ate`、`rpe`、`acceleration`、`acceleration_error`、
  `jitter`、`mpfje`、`mpfve`、`hand_scale_error`。
- 深度/相机/点云：`depth`、`mae_depth`、`rmse_depth`、`abs_rel`、`sq_rel`、
  `delta1`、`intrinsics`、`extrinsics`、`rra`、`rta`、`pose_auc`、`pointcloud`、
  `pointcloud_accuracy`、`pointcloud_completeness`、`chamfer_l1`、`chamfer_l2`。
- 物体/接触/分类：`add`、`adds`、`add_0_1d`、`contact_coverage`、
  `joint_visibility`、`vertex_visibility`、`joint_contact`、`vertex_contact`、
  `joint_contact_distance`、`vertex_contact_distance`、`average_precision`、
  `roc_auc`、`sim`、`detection_ap`、`map`。
- 图像/跟踪/效率：`image`、`psnr`、`ssim`、`tracking`、`average_jaccard`、
  `delta_avg_visible`、`occlusion_accuracy`、`efficiency`。
- 显式占位：`lpips`、`fid`、`intersection_volume`、`penetration_depth`、
  `geodesic_contact_distance`、`euler_lagrange_residual`、`success_rate`。

关键字段约定：joint/vertex 序列使用 `(T, J/V, 3)`；相机 pose 使用 `(T, 4, 4)`；
点云使用 `(N, 3)`；检测 box 使用 `(N, 4)` 的 `xyxy` 格式；tracking 坐标使用
`(..., 2)`，visibility 使用相同 leading shape 的布尔数组。所有阈值、坐标系方向、
根关节索引和单位都应由实验配置明确指定。
