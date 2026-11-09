# Third-party notices

## Code copied or ported into this repository

| Original | Licence as stated by the original | Where in this repository | How |
|---|---|---|---|
| DOTA_devkit, `dota_evaluation_task2.py` (CAPTAIN-WHU/DOTA_devkit, commit 9938855) | File header: "Licensed under The MIT License" (Jian Ding, based on code from Bharath Hariharan); the repository itself has no LICENSE file | `third_party/DOTA_devkit/dota_evaluation_task2.py` (unchanged copy, reference only; provenance in `third_party/DOTA_devkit/SOURCE.txt`); `scahi/dota/evaluate.py` (port, `voc_ap` verbatim) | copy / port |
| VisDrone2018-DET-toolkit (MATLAB: `evalDET.m`, `utils/compOas.m`, `utils/calcAccuracy.m`, `utils/VOCap.m`) | No licence stated (no LICENSE file in the repository, none in its README or code) | `scahi/eval/visdrone_protocol.py` (exact Python port), weighted variant in `scahi/eval/bootstrap.py` | port |
| pycocotools, `COCOeval.accumulate` | BSD-2-Clause (FreeBSD) | `scahi/eval/bootstrap.py` (weighted re-evaluation with identical results) | re-implementation |
| torchvision FCOS post-processing (`postprocess_detections`, `BoxLinearCoder`) | BSD-3-Clause | `scahi/fcos.py` | re-implementation |
| Ultralytics `LetterBox`, `Detect.postprocess` (top-k), DFL decoding | AGPL-3.0 | `scahi/infer/preprocess.py`, `scahi/infer/dfl_decode.py`, `scahi/infer/raw_head.py` (part of the AGPL-3.0-or-later code) | re-implementation of the behaviour (parity-tested), no copied text |

The SCAHI core (`scahi/fusion/scalespace.py`, `scahi/fusion/scalespace_torch.py`, Apache-2.0) contains none of the
above and imports only numpy, pandas, scipy, torch and torchvision.

## Figure

`docs/figures/assets/input.jpg` (and the photo inside `docs/figures/fig_method.{pdf,png}`) is a downscaled image of the
VisDrone dataset; it is subject to the VisDrone terms of use (academic / non-commercial) and not covered by the licences of
this repository. The TikZ source of the figure is AGPL-3.0-or-later.

## Dependencies (installed separately, not redistributed)

| Package | Licence |
|---|---|
| ultralytics 8.4.183 | AGPL-3.0 |
| torch 2.14.1 | BSD-3-Clause (with Apache-2.0 components) |
| torchvision 0.29.1, scipy, numba | BSD |
| numpy, pandas, psutil, shapely | BSD-3-Clause |
| pycocotools | BSD-2-Clause |
| opencv-python | Apache-2.0 |
| sahi, ensemble-boxes, PyYAML, pytest | MIT |
| matplotlib | Matplotlib licence (PSF-based) |
| Pillow | MIT-CMU |

## Datasets and weights

VisDrone and DOTA are not redistributed and are subject to their own terms of use (academic / non-commercial). Model
weights trained with this code start from Ultralytics' pre-trained weights (AGPL-3.0) or torchvision's COCO weights.
