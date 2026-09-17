# 3x4 face-level contact overlay pilot

The review target is `h2o_r017_000208_3x4_face_contact_example.png`.
It is the frozen 114-gallery entry H2O rank 17, sequence
`subject4_ego/h1/5`, center frame `000208`, cache
`abc07caace9823faa3ffe7ce`.

- Every overlay is clipped to the RGB image rectangle.
- Contact and visibility use MANO triangular faces. Each face score is the
  mean of its three vertex values and is drawn with alpha 0.48.
- The object and both hands share a nearest-surface z-buffer. GT visibility is
  derived from that scene z-buffer. Ego visibility is the model's 195-marker
  prediction interpolated to 778 vertices before face aggregation.
- The bottom row compares right-hand contact on the same RGB frame.
  Ego uses its reconstructed 195-to-778 mesh. S2Contact and ContactOpt use
  their own predicted 778-vertex camera-space meshes. InteractVLM supplies a
  topology-verified 6890-to-778 contact field and uses GT MANO only as its
  projection support because this artifact has no verified hand mesh.

The JSON file records the exact frame/window identity, remote artifact paths,
array shapes, contact counts, rendering contract, and per-panel overlay pixel
counts. `render_example.py` reproduces the sample from the staged inputs. This
pilot does not start the 114-clip batch.

Earlier OakInk-v2 compatibility outputs are intentionally excluded from the
tracked handoff. The H2O rank-17 PNG is the only current review target.
