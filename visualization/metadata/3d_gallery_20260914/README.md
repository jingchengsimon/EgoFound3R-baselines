# 3D gallery metadata

- `LATEST_3D_10S_GALLERY.json`: canonical completed v5 104-segment gallery, selection identity, and completion gates.
- `LATEST_3D_VIDEO_RESULT.json`: canonical video result path and frozen media verification.
- `PREVIOUS_3D_VIDEO_OUTPUTS_BEFORE_DELETE_20260915.json`: exact pre-deletion inventory for superseded galleries.
- `PREVIOUS_3D_VIDEO_OUTPUTS_DELETION_RESULT_20260915_REMOTE.json`: verified remote deletion result.
- `PREVIOUS_3D_VIDEO_OUTPUTS_DELETION_RESULT_20260915_LOCAL.json`: verified local deletion result.

See [`../../3D_VISUALIZATION_20260916.md`](../../3D_VISUALIZATION_20260916.md). The latest JSON files were updated without reconnecting to the currently disconnected 5001 node; they preserve the completion evidence verified before disconnection.
