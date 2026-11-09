# Paper terms and names in the code

| Paper | Code / configs |
|---|---|
| single pass (at long side s) | `B-FULL` training, `full<s>` / `E-<s>` rows, `scahi.infer.full.predict_full` |
| tiles (zoom z, network input N) | levels `t<z>n<N>` (e.g. `t2.0n1024`), `scahi.data.zoom.tile_grid` |
| single pass + tiles (stack) | stacks `native` (1280 + tiles), `native2048`, … in `scahi.infer.cache_mz.STACK_LEVELS` |
| box fusion (NMS with intersection over smaller box) | variant `F0-NMS`, metric `IOS` |
| SCAHI (pre-specified setting) | variant `F2` in `scahi.fusion.scalespace`: weights = border taper only (no validity window, no uncertainty factor) |
| SCAHI, refined grid | "X-grid" (`configs/ext/xgrid_frozen.json`, rows `XG_*`) |
| SCAHI on the single pass only (no tiles) | decomposition row B, `XG_*_B`, `D_*` |
| one-to-one / one-to-many head | branch `o2o` / `o2m` |
| calibration split | `calib` (`data/splits/calib.txt`); training split `train_main` |
| pass, level | one network forward of a crop / all passes of one zoom |
