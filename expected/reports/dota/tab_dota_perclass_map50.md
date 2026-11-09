# DOTA-val per-class AP50, frozen rows at S_max

Devkit port, difficult GT ignored, no cap; 458 images; values from runs/dota/val/results.json. Box fusion = class-wise NMS across passes on the o2m head (same head as SCAHI; YOLOv11s / YOLOv12s: their only head). S_max = 2048, except YOLOv12s: 1024 (its 2048 training does not fit; addendum 2). Best per class and detector in bold (ties at the shown precision all bold). FCOS (addendum 3, one head): runs/dota/val/results_fcos.json.

## YOLOv10s, 2048 px

| class | single pass | tiles + box fusion | tiles + SCAHI | tiles + SCAHI, refined grid |
|---|---:|---:|---:|---:|
| plane | 88.6 | 90.2 | **90.3** | **90.3** |
| baseball-diamond | 70.4 | 64.8 | **71.9** | 71.5 |
| bridge | 45.5 | 48.0 | 50.5 | **51.1** |
| ground-track-field | 59.2 | 68.4 | 67.1 | **70.6** |
| small-vehicle | 70.7 | 77.7 | 81.6 | **81.7** |
| large-vehicle | 79.9 | 81.7 | 75.3 | **82.5** |
| ship | 88.5 | 89.7 | 89.6 | **89.9** |
| tennis-court | 90.7 | 90.7 | **90.9** | **90.9** |
| basketball-court | 52.3 | 61.7 | **63.7** | 63.2 |
| storage-tank | 81.4 | 86.1 | **87.1** | 86.6 |
| soccer-ball-field | 65.8 | **76.4** | 75.2 | 74.2 |
| roundabout | 54.8 | 66.0 | **69.2** | 67.9 |
| harbor | 82.4 | 80.8 | **85.9** | 85.7 |
| swimming-pool | 69.5 | 75.6 | **78.8** | 78.3 |
| helicopter | 48.1 | 35.3 | **50.6** | 49.5 |
| container-crane | 0.1 | 2.3 | **2.6** | 2.2 |
| mAP50 | 65.5 | 68.5 | 70.6 | **71.0** |

## YOLOv10m, 2048 px

| class | single pass | tiles + box fusion | tiles + SCAHI | tiles + SCAHI, refined grid |
|---|---:|---:|---:|---:|
| plane | 89.9 | 90.3 | **90.4** | **90.4** |
| baseball-diamond | 72.3 | 70.1 | **75.9** | 75.2 |
| bridge | 42.7 | 43.9 | **47.2** | **47.2** |
| ground-track-field | 69.0 | 67.9 | **73.8** | 73.3 |
| small-vehicle | 75.2 | 80.5 | **82.9** | **82.9** |
| large-vehicle | 80.6 | 76.7 | 76.3 | **83.8** |
| ship | 89.4 | 89.4 | 89.6 | **89.8** |
| tennis-court | 90.7 | 90.8 | **90.9** | **90.9** |
| basketball-court | 58.3 | 67.7 | **70.0** | 69.8 |
| storage-tank | 82.0 | 85.9 | **87.2** | 87.1 |
| soccer-ball-field | 63.3 | 66.2 | 70.6 | **72.4** |
| roundabout | 58.9 | 67.6 | 68.5 | **69.2** |
| harbor | 83.7 | 77.6 | 85.5 | **86.2** |
| swimming-pool | 71.1 | 70.1 | **77.4** | 76.6 |
| helicopter | 50.8 | 44.8 | **54.7** | 52.4 |
| container-crane | 0.4 | **0.8** | 0.4 | 0.5 |
| mAP50 | 67.4 | 68.1 | 71.3 | **71.7** |

## YOLOv11s, 2048 px

| class | single pass | tiles + box fusion | tiles + SCAHI | tiles + SCAHI, refined grid |
|---|---:|---:|---:|---:|
| plane | 89.3 | 90.0 | **90.2** | 90.1 |
| baseball-diamond | 71.5 | 70.9 | 72.1 | **72.5** |
| bridge | 45.2 | 44.7 | **48.7** | 48.6 |
| ground-track-field | 59.2 | 62.4 | 65.9 | **67.7** |
| small-vehicle | 72.7 | 79.1 | **82.2** | **82.2** |
| large-vehicle | 79.9 | 81.7 | 75.9 | **84.1** |
| ship | 88.9 | 89.2 | 89.2 | **89.7** |
| tennis-court | 90.8 | 90.8 | **90.9** | **90.9** |
| basketball-court | 54.2 | 59.5 | **61.0** | 60.8 |
| storage-tank | 83.5 | 86.8 | **87.8** | 87.4 |
| soccer-ball-field | 61.4 | 63.3 | **71.2** | 70.6 |
| roundabout | 58.1 | 65.2 | **68.8** | 68.7 |
| harbor | 83.9 | 77.7 | 85.6 | **85.9** |
| swimming-pool | 68.4 | 75.4 | **79.8** | 79.5 |
| helicopter | 35.5 | 29.7 | 36.7 | **37.7** |
| container-crane | 0.5 | 0.4 | 0.4 | **0.6** |
| mAP50 | 65.2 | 66.7 | 69.1 | **69.8** |

## YOLOv12s, 1024 px (S_max = 1024)

| class | single pass | tiles + box fusion | tiles + SCAHI | tiles + SCAHI, refined grid |
|---|---:|---:|---:|---:|
| plane | 80.6 | 89.7 | **89.9** | **89.9** |
| baseball-diamond | 66.1 | 60.9 | **67.5** | 66.8 |
| bridge | 27.7 | 35.6 | 39.0 | **39.6** |
| ground-track-field | 47.3 | 58.6 | 59.6 | **60.6** |
| small-vehicle | 58.1 | 75.3 | **78.9** | **78.9** |
| large-vehicle | 77.0 | 78.0 | 74.8 | **81.5** |
| ship | 78.6 | **89.2** | 88.0 | **89.2** |
| tennis-court | 90.6 | 90.7 | **90.8** | **90.8** |
| basketball-court | 47.2 | 48.5 | 53.5 | **54.9** |
| storage-tank | 62.7 | 76.8 | **79.9** | 79.6 |
| soccer-ball-field | 63.3 | 67.3 | 70.7 | **71.6** |
| roundabout | 33.8 | 56.2 | **60.0** | 59.1 |
| harbor | 78.3 | 75.2 | 82.3 | **82.8** |
| swimming-pool | 60.8 | 72.0 | 71.9 | **72.6** |
| helicopter | 29.3 | 32.0 | 36.0 | **37.6** |
| container-crane | 0.1 | 0.0 | 0.1 | **0.2** |
| mAP50 | 56.4 | 62.9 | 65.2 | **66.0** |

## FCOS, 2048 px

| class | single pass | tiles + box fusion | tiles + SCAHI | tiles + SCAHI, refined grid |
|---|---:|---:|---:|---:|
| plane | 79.7 | 88.9 | **89.0** | **89.0** |
| baseball-diamond | 60.1 | 60.5 | **64.9** | **64.9** |
| bridge | 32.6 | 38.2 | **41.4** | **41.4** |
| ground-track-field | 43.8 | **45.6** | 41.7 | 41.7 |
| small-vehicle | 57.8 | 67.9 | **68.7** | **68.7** |
| large-vehicle | 76.3 | **78.0** | 71.6 | 71.6 |
| ship | 78.3 | **86.0** | 80.7 | 80.7 |
| tennis-court | 90.0 | 90.5 | **90.7** | **90.7** |
| basketball-court | 43.1 | 41.6 | **45.6** | **45.6** |
| storage-tank | 68.9 | **81.4** | 81.0 | 81.0 |
| soccer-ball-field | 56.9 | 66.3 | **66.6** | **66.6** |
| roundabout | 49.0 | 61.5 | **61.7** | **61.7** |
| harbor | 69.7 | 70.7 | **73.6** | **73.6** |
| swimming-pool | 55.5 | 63.9 | **65.2** | **65.2** |
| helicopter | 18.4 | 26.9 | **27.0** | **27.0** |
| container-crane | **0.0** | **0.0** | **0.0** | **0.0** |
| mAP50 | 55.0 | 60.5 | **60.6** | **60.6** |
