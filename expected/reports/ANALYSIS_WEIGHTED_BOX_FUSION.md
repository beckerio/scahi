# Randgewichtete Box-Fusion (explorativ, post hoc)

Nutzerauftrag 2026-11-07 (Spezifikation `docs/ANALYSIS_WEIGHTED_BOX_FUSION.md`, Einordnung in `docs/DECISIONS.md`, beides vor jeder Zahl); FCOS auf DOTA ergänzt 2026-11-08. **Nie Grundlage eines Urteils**; Verwendung im Paper nur nach Entscheidung der Autoren. Nur CPU, aus den vorhandenen Caches (`scripts/weighted_box_fusion.py`).

Frage: Kommt der Gewinn von SCAHI gegenüber Box-Fusion aus der abdeckungsbewussten Gewichtung oder daraus, dass die dichten Ausgaben zusammengeführt werden, bevor Boxen entschieden sind?

- Dieselben Pässe wie die eingefrorenen Box-Fusion-Zeilen bei 2048 px (Einzelpass bei der Trainingsgröße + Kacheln), derselbe Kopf, die eigenen Detektionen des Detektors pro Pass.
- (a) gewichtetes NMS: Score jeder Detektion pro Pass × SCAHI-Gewicht (Rand-Taper b_t der eigenen Kachel, Einzelpass 1, kein Validity-Fenster) am Boxmittelpunkt, dann die Box-Fusion der eingefrorenen Zeile (NMS mit Schnitt über der kleineren Box). (b) gewichtetes WBF: dieselben Scores, Weighted Boxes Fusion (Raster der Hauptstudie).
- Nur die Box-Fusion-Parameter auf calib gewählt, mit Raster und Regel der eingefrorenen Zeile.
- Vergleiche nur zwischen Zeilen auf demselben Kopf. Gepaarter Bootstrap über Bilder (1000 Resamples, Seed 0, dieselben Resamples wie die Hauptauswertung), Perzentil-95-%-KI. R50: Recall bei IoU 0,5 dicht stehender bzw. vereinzelter Objekte (Definitionen wie bisher).
- VisDrone YOLOv10s o2m: die Box-Fusion auf o2m ist neu (post hoc, abgestimmt wie X1); auf o2o gibt es keine vorab festgelegte SCAHI-Zeile.

## VisDrone val

Primärmetrik AP (zusätzlich AP50). Δ = Zeile − Referenz.

| Detektor | Kopf | Zeile | AP | AP50 | R50 dicht | R50 vereinzelt | Δ AP gg. Box-Fusion [KI] | Δ AP gg. RG [KI] | Δ AP50 gg. Box-Fusion [KI] | Δ AP50 gg. RG [KI] |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv10s | o2o | Box-Fusion (eingefroren) | 38,85 | 64,73 | 79,8 | 91,0 |  |  |  |  |
| YOLOv10s | o2o | (a) gewichtetes NMS | 39,57 | 64,75 | 86,2 | 91,5 | +0,72 [+0,52; +0,93] | -1,00 [-1,13; -0,84] | +0,02 [-0,23; +0,32] | -1,42 [-1,62; -1,20] |
| YOLOv10s | o2o | (b) gewichtetes WBF | 37,56 | 62,34 | 87,1 | 93,0 | -1,29 [-1,57; -0,98] | -3,01 [-3,25; -2,74] | -2,40 [-2,83; -1,95] | -3,84 [-4,18; -3,44] |
| YOLOv10s | o2o | SCAHI verfeinertes Raster (RG) | 40,57 | 66,17 | 85,7 | 92,5 |  |  |  |  |
| YOLOv10s | o2m | Box-Fusion (neu, post hoc, abgestimmt wie X1) | 37,85 | 62,76 | 79,7 | 87,9 |  |  |  |  |
| YOLOv10s | o2m | (a) gewichtetes NMS | 38,60 | 62,68 | 86,3 | 88,3 | +0,75 [+0,57; +0,95] | -2,12 [-2,29; -1,93] | -0,08 [-0,35; +0,20] | -3,47 [-3,73; -3,21] |
| YOLOv10s | o2m | (b) gewichtetes WBF | 35,59 | 59,11 | 85,1 | 89,3 | -2,25 [-2,55; -1,90] | -5,12 [-5,42; -4,79] | -3,66 [-4,09; -3,16] | -7,05 [-7,47; -6,57] |
| YOLOv10s | o2m | SCAHI vorab festgelegt (F2) | 39,34 | 64,19 | 63,4 | 89,0 |  |  |  |  |
| YOLOv10s | o2m | SCAHI verfeinertes Raster (RG) | 40,72 | 66,15 | 81,1 | 91,9 |  |  |  |  |
| YOLOv11s | o2m | Box-Fusion (eingefroren) | 37,22 | 62,18 | 79,1 | 87,6 |  |  |  |  |
| YOLOv11s | o2m | (a) gewichtetes NMS | 37,72 | 62,52 | 77,3 | 87,3 | +0,50 [+0,30; +0,71] | -2,31 [-2,52; -2,14] | +0,34 [+0,07; +0,62] | -2,78 [-3,07; -2,50] |
| YOLOv11s | o2m | (b) gewichtetes WBF | 35,12 | 58,64 | 84,4 | 89,1 | -2,10 [-2,42; -1,73] | -4,91 [-5,21; -4,57] | -3,55 [-3,99; -3,04] | -6,67 [-7,07; -6,19] |
| YOLOv11s | o2m | SCAHI vorab festgelegt (F2) | 38,66 | 63,58 | 62,1 | 88,9 |  |  |  |  |
| YOLOv11s | o2m | SCAHI verfeinertes Raster (RG) | 40,03 | 65,30 | 81,6 | 91,6 |  |  |  |  |

**Zusammenfassung VisDrone val** (3 Detektor-/Kopf-Paare): (a) gewichtetes NMS − Box-Fusion AP +0,50 bis +0,75 (AP50 -0,08 bis +0,34); SCAHI (RG) − gewichtetes NMS AP +1,00 bis +2,31 (AP50 +1,42 bis +3,47).

Auf calib gewählte Box-Fusion-Parameter der gewichteten Varianten: yolov10s/o2o/a/True: {"metric": "IOU", "thr": 0.7, "conf": 0.01}; yolov10s/o2o/b/True: {"thr": 0.6, "conf": 0.001}; yolov10s/o2m/a/True: {"metric": "IOU", "thr": 0.7, "conf": 0.01}; yolov10s/o2m/b/True: {"thr": 0.6, "conf": 0.001}; yolov10s/o2m/a/False: {"metric": "IOS", "thr": 0.8, "conf": 0.01}; yolo11s/o2m/a/True: {"metric": "IOS", "thr": 0.8, "conf": 0.01}; yolo11s/o2m/b/True: {"thr": 0.6, "conf": 0.001}.

## DOTA-val

Primärmetrik mAP50 (zusätzlich AP). Δ = Zeile − Referenz.

| Detektor | Kopf | Zeile | mAP50 | AP | R50 dicht | R50 vereinzelt | Δ mAP50 gg. Box-Fusion [KI] | Δ mAP50 gg. RG [KI] | Δ AP gg. Box-Fusion [KI] | Δ AP gg. RG [KI] |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv10s | o2m | Box-Fusion (eingefroren) | 68,46 | 47,41 | 94,6 | 95,6 |  |  |  |  |
| YOLOv10s | o2m | (a) gewichtetes NMS | 69,87 | 48,15 | 94,2 | 95,5 | +1,42 [+0,22; +1,72] | -1,12 [-2,27; -0,57] | +0,75 [-0,02; +1,32] | -1,97 [-2,61; -1,47] |
| YOLOv10s | o2m | (b) gewichtetes WBF | 64,30 | 43,39 | 80,6 | 95,2 | -4,16 [-5,56; -2,71] | -6,70 [-8,04; -5,15] | -4,01 [-5,06; -2,82] | -6,73 [-7,64; -5,61] |
| YOLOv10s | o2m | SCAHI vorab festgelegt (F2) | 70,64 | 49,15 | 66,8 | 96,9 |  |  |  |  |
| YOLOv10s | o2m | SCAHI verfeinertes Raster (RG) | 71,00 | 50,13 | 91,1 | 98,1 |  |  |  |  |
| YOLOv11s | o2m | Box-Fusion (eingefroren) | 66,68 | 45,93 | 92,5 | 95,5 |  |  |  |  |
| YOLOv11s | o2m | (a) gewichtetes NMS | 68,41 | 47,02 | 89,4 | 95,2 | +1,73 [+0,55; +2,41] | -1,39 [-2,23; -0,72] | +1,09 [+0,23; +1,73] | -2,34 [-3,01; -1,89] |
| YOLOv11s | o2m | (b) gewichtetes WBF | 63,32 | 42,80 | 81,4 | 96,3 | -3,36 [-4,47; -1,98] | -6,49 [-7,46; -5,05] | -3,13 [-4,00; -2,03] | -6,57 [-7,44; -5,59] |
| YOLOv11s | o2m | SCAHI vorab festgelegt (F2) | 69,14 | 48,42 | 66,8 | 96,8 |  |  |  |  |
| YOLOv11s | o2m | SCAHI verfeinertes Raster (RG) | 69,81 | 49,36 | 94,5 | 98,0 |  |  |  |  |
| FCOS | o2m | Box-Fusion (eingefroren) | 60,49 | 37,50 | 93,4 | 95,3 |  |  |  |  |
| FCOS | o2m | (a) gewichtetes NMS | 59,74 | 36,86 | 91,1 | 94,5 | -0,75 [-1,79; +0,14] | -0,85 [-2,24; +0,23] | -0,64 [-1,33; +0,01] | -0,69 [-1,44; +0,09] |
| FCOS | o2m | SCAHI vorab festgelegt (F2) | 60,59 | 37,55 | 62,6 | 93,9 |  |  |  |  |
| FCOS | o2m | SCAHI verfeinertes Raster (RG) | 60,59 | 37,55 | 62,6 | 93,9 |  |  |  |  |

**Zusammenfassung DOTA-val** (3 Detektor-/Kopf-Paare): (a) gewichtetes NMS − Box-Fusion mAP50 -0,75 bis +1,73 (AP -0,64 bis +1,09); SCAHI (RG) − gewichtetes NMS mAP50 +0,85 bis +1,39 (AP +0,69 bis +2,34).

Auf calib gewählte Box-Fusion-Parameter der gewichteten Varianten: yolov10s/a: {"metric": "IOS", "thr": 0.8, "conf": 0.01}; yolo11s/a: {"metric": "IOS", "thr": 0.7, "conf": 0.01}; yolov10s/b: {"thr": 0.4, "conf": 0.05}; yolo11s/b: {"thr": 0.4, "conf": 0.01}; fcos/a: {"metric": "IOS", "thr": 0.8, "conf": 0.01}.

## Hinweise

- DOTA FCOS (2048): Variante (a) ausgewertet; Variante (b) nicht gerechnet (Nutzerentscheidung 2026-11-08): Kosten (bis zu 1,4 M Detektionen pro Bild nach der Fusion auf calib; WBF ist bei dieser Menge nicht praktikabel), und (b) lag schon auf VisDrone und auf DOTA für YOLO unter der Box-Fusion.
- DOTA: Grenze 200 000 bzw. 20 000 pro Pass wie die eingefrorene Box-Fusion-Zeile; die zusammengeführte Ausgabe ist nicht gekürzt (eingefrorener Pfad).
- Fusionszeit pro Bild (informativ, nicht exklusiv gemessen): in `runs/analysis_wbf/*.json`.
- CSV mit allen Werten und KI-Grenzen: `reports/ANALYSIS_WEIGHTED_BOX_FUSION.csv`.
