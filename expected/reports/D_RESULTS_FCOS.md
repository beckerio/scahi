# Phase D, Addendum 3 – FCOS auf DOTA-val (v1.5, HBB)

DOTA-val (458 Bilder), **einmal ausgewertet** nach dem freigegebenen FCOS-Freeze (`configs/dota/frozen_rows_fcos.json`, Freigabe 2026-11-06T08:28, `docs/DECISIONS.md`). Alle 13 Zeilen, kein Nachtuning; Pipeline-Grenze 200000 je Bild (Einzelpass, SCAHI-top_k) und je Pass; **Abweichung vom Wortlaut** „je Bild und je Pass“ (Addendum 3, Punkt 6): bei der Box-Fusion greift die Grenze nur je Pass, die zusammengeführte Ausgabe ist nicht begrenzt – der eingefrorene, auf calib ausgewählte und exakt nachgerechnete Pfad, wie bei allen YOLO-Box-Fusion-Zeilen (Entscheidung 2026-11-07). Primär Devkit-mAP50 ohne Cap, ε = 0,30, gepaarter Bootstrap über Bilder (1000 Resamples, Seed 0, Perzentil-KI), einseitiges p gegen ε. **Ein Modell → kein Holm:** Urteil mit dem unadjustierten p ≤ 0,05 und dem Kriterium (D-H3 zusätzlich untere KI-Grenze > 0). Deskriptiv: Anzahl der fünf DOTA-Detektoren, die eine Hypothese erfüllen; die Urteile der vier YOLO-Modelle (`reports/D_RESULTS.md`, `reports/D_RESULTS_NEW.md`) bleiben unverändert. S_max = 2048. FCOS hat einen Kopf; Einzelpass, Box-Fusion und SCAHI nutzen ihn mit dem Detektor-Score sqrt(p · Centerness); keine SAHI-Zeile. Erzeugt von `scripts/dota/d_results_fcos.py` aus `runs/dota/val/results_fcos.json`.

## Hypothesen (primär: mAP50 ohne Cap)

| Hypothese | Kontrast | a | b | Δ [95 %-KI] | p (einseitig) | Kriterium | erfüllt | erfüllt (von 5, deskriptiv) | YOLO-Urteile (unverändert) |
|---|---|---|---|---|---|---|---|---|---|
| D-H1 | A_fcos_2048 − NMS-o2m_fcos_1024 | 55,01 | 52,56 | +2,45 [-0,49; +5,28] | 0,080 | ja | nein | 2 | YOLOv10s nein, YOLOv10m ja, YOLOv11s ja, YOLOv12s nein |
| D-H2a | RG_fcos_1024 − NMS-o2m_fcos_1024 | 52,96 | 52,56 | +0,40 [-0,92; +1,79] | 0,375 | ja | nein | 4 | YOLOv10s ja, YOLOv10m ja, YOLOv11s ja, YOLOv12s ja |
| D-H2b | RG_fcos_2048 − NMS-o2m_fcos_2048 | 60,59 | 60,49 | +0,09 [-0,91; +1,37] | 0,575 | nein | nein | 3 | YOLOv10s ja, YOLOv10m ja, YOLOv11s ja, YOLOv12s nein |
| D-H3 | RG_fcos_2048 − A_fcos_2048 | 60,59 | 55,01 | +5,57 [+3,77; +7,65] | 0,001 | ja | ja | 5 | YOLOv10s ja, YOLOv10m ja, YOLOv11s ja, YOLOv12s ja |

Sekundär (ohne Urteil): Δ [95 %-KI]

| Hypothese | mAP50 Cap 4000 | „alle Objekte“ | Fahrzeuge mAP50 | AP@[.5:.95] | Recall@0,5 crowded | Recall@0,5 isolated |
|---|---|---|---|---|---|---|
| D-H1 | +2,41 [-0,51; +5,31] | +0,88 [-2,16; +3,69] | -1,16 [-7,11; +5,78] | +2,45 [+0,34; +4,52] | +0,64 [-2,61; +3,39] | -6,67 [-12,15; -1,60] |
| D-H2a | +0,38 [-0,94; +1,77] | +0,24 [-1,07; +1,63] | -1,52 [-4,79; +2,34] | +0,54 [-0,25; +1,36] | -30,34 [-35,18; -24,39] | -0,87 [-1,59; -0,21] |
| D-H2b | +0,10 [-0,96; +1,33] | -0,42 [-1,50; +0,49] | -2,79 [-6,01; +0,55] | +0,05 [-0,80; +0,78] | -30,85 [-33,52; -27,06] | -1,40 [-2,15; -0,70] |
| D-H3 | +5,58 [+3,71; +7,62] | +6,21 [+4,25; +8,03] | +3,11 [-0,70; +6,39] | +3,64 [+2,38; +4,94] | -29,50 [-33,59; -23,23] | +7,12 [+2,18; +12,26] |

## Sekundäre Kontraste (vorab festgelegt, ohne Urteil)

Ohne die o2o- und SAHI-Kontraste (ein Kopf, keine SAHI-Zeile; Addendum 3). Die Inferenz-Kreuzung wählte auf calib 2560 px für die 2048er Gewichte (calib-mAP50 je Größe: 2048: 60,30, 2560: 63,08, 3072: 62,53); der Vergleich RG_2048 − X_2048 ist laut Freigabe als sekundärer Vergleich zu berichten.

| Kontrast | Δ mAP50 [95 %-KI] | Δ „alle Objekte“ | Δ AP@[.5:.95] |
|---|---|---|---|
| F2 statt RG: D-H2a | +0,40 [-0,92; +1,79] | +0,24 [-1,07; +1,63] | +0,54 [-0,25; +1,36] |
| F2 statt RG: D-H2b | +0,09 [-0,91; +1,37] | -0,42 [-1,50; +0,49] | +0,05 [-0,80; +0,78] |
| F2 statt RG: D-H3 | +5,57 [+3,77; +7,65] | +6,21 [+4,25; +8,03] | +3,64 [+2,38; +4,94] |
| D-H1, Box-Fusion o2m | +2,45 [-0,49; +5,28] | +0,88 [-2,16; +3,69] | +2,45 [+0,34; +4,52] |
| Inferenz-Kreuzung RG_2048 − X_2048 | +2,47 [+1,01; +4,27] | +3,12 [+1,26; +4,47] | +1,54 [+0,46; +2,83] |
| Zerlegung B-F2_1024 − A_1024 | -0,75 [-1,21; -0,13] | -0,79 [-1,15; -0,14] | -1,00 [-1,34; -0,75] |
| Zerlegung F2_1024 − B-F2_1024 | +11,07 [+8,31; +13,55] | +11,13 [+8,43; +13,45] | +9,03 [+7,40; +10,70] |
| Zerlegung B-RG_1024 − A_1024 | -0,19 [-0,51; +0,28] | -0,11 [-0,47; +0,31] | -0,23 [-0,42; -0,08] |
| Zerlegung RG_1024 − B-RG_1024 | +10,51 [+7,72; +13,03] | +10,45 [+7,77; +12,92] | +8,26 [+6,55; +9,94] |
| Zerlegung B-F2_2048 − A_2048 | -0,97 [-1,79; -0,54] | -1,08 [-1,68; -0,45] | -1,70 [-2,20; -1,28] |
| Zerlegung F2_2048 − B-F2_2048 | +6,54 [+5,02; +8,62] | +7,29 [+5,32; +8,86] | +5,34 [+4,20; +6,54] |
| Zerlegung B-RG_2048 − A_2048 | -0,97 [-1,79; -0,54] | -1,08 [-1,68; -0,45] | -1,70 [-2,20; -1,28] |
| Zerlegung RG_2048 − B-RG_2048 | +6,54 [+5,02; +8,62] | +7,29 [+5,32; +8,86] | +5,34 [+4,20; +6,54] |

## Alle Zeilen

| Zeile | Beschreibung | mAP50 | Cap 4000 (Bilder > Cap) | alle Objekte | AP@[.5:.95] | Fahrzeuge mAP50 | klassenagn. AP50 | Recall crowded / isolated | Bilder an der Grenze (Box-Fusion: Ausgabe ≥ Grenze, nicht gekürzt) | max. Det./Bild |
|---|---|---|---|---|---|---|---|---|---|---|
| `A_fcos_1024` | single pass | **42,64** | 42,63 (458) | 38,60 | 23,97 | 55,12 | 54,82 | 84,3 / 69,6 | 24 | 200000 |
| `NMS-o2m_fcos_1024` | tiles + box fusion | **52,56** | 52,56 (458) | 48,70 | 31,45 | 68,23 | 74,51 | 91,5 / 93,4 | 182 | 3495205 |
| `F2_fcos_1024` | tiles + SCAHI | **52,96** | 52,94 (452) | 48,94 | 31,99 | 66,70 | 72,11 | 61,1 / 92,5 | 8 | 200000 |
| `RG_fcos_1024` | tiles + SCAHI, refined grid | **52,96** | 52,94 (452) | 48,94 | 31,99 | 66,70 | 72,11 | 61,1 / 92,5 | 8 | 200000 |
| `B-F2_fcos_1024` | SCAHI, single pass (no tiles) | **41,89** | 41,89 (448) | 37,81 | 22,97 | 50,48 | 52,51 | 53,2 / 65,1 | 0 | 41826 |
| `B-RG_fcos_1024` | SCAHI, single pass (no tiles), refined grid | **42,45** | 42,44 (457) | 38,49 | 23,74 | 53,68 | 54,27 | 69,2 / 67,7 | 0 | 46626 |
| `A_fcos_2048` | single pass | **55,01** | 54,97 (458) | 49,58 | 33,91 | 67,07 | 71,66 | 92,1 / 86,7 | 76 | 200000 |
| `X_fcos_2048` | single pass of the 2048 weights at 2560 px (inference size selected on calib) | **58,12** | 58,12 (458) | 52,67 | 36,00 | 69,85 | 75,03 | 94,2 / 91,1 | 82 | 200000 |
| `NMS-o2m_fcos_2048` | tiles + box fusion | **60,49** | 60,45 (454) | 56,21 | 37,50 | 72,97 | 79,21 | 93,4 / 95,3 | 159 | 3407292 |
| `F2_fcos_2048` | tiles + SCAHI | **60,59** | 60,55 (458) | 55,79 | 37,55 | 70,18 | 76,90 | 62,6 / 93,9 | 1 | 200000 |
| `RG_fcos_2048` | tiles + SCAHI, refined grid | **60,59** | 60,55 (458) | 55,79 | 37,55 | 70,18 | 76,90 | 62,6 / 93,9 | 1 | 200000 |
| `B-F2_fcos_2048` | SCAHI, single pass (no tiles) | **54,04** | 54,04 (458) | 48,50 | 32,21 | 62,47 | 67,89 | 60,4 / 82,1 | 0 | 115809 |
| `B-RG_fcos_2048` | SCAHI, single pass (no tiles), refined grid | **54,04** | 54,04 (458) | 48,50 | 32,21 | 62,47 | 67,89 | 60,4 / 82,1 | 0 | 115809 |

## Größenbins (σ = √(w·h) px)

Primär (difficult ignoriert; der Bin < 8 hat dann keine GT: „—“) und „alle Objekte“. AP@[.5:.95] / AP50.

| Zeile | Auswertung | <8 | 8-16 | 16-32 | 32-64 | 64-128 | >128 |
|---|---|---|---|---|---|---|---|
| `A_fcos_1024` | primär | — / — | 4,9 / 9,7 | 8,2 / 20,6 | 18,8 / 35,1 | 26,0 / 44,6 | 42,7 / 67,5 |
| `A_fcos_1024` | alle Objekte | 0,8 / 4,2 | 5,4 / 11,3 | 7,4 / 19,1 | 18,4 / 34,9 | 25,8 / 44,9 | 42,4 / 67,2 |
| `NMS-o2m_fcos_1024` | primär | — / — | 9,5 / 22,2 | 21,3 / 41,0 | 26,9 / 45,2 | 31,0 / 49,7 | 43,6 / 65,7 |
| `NMS-o2m_fcos_1024` | alle Objekte | 2,0 / 8,9 | 7,5 / 19,4 | 20,7 / 40,5 | 26,3 / 44,7 | 30,4 / 49,4 | 43,1 / 65,2 |
| `F2_fcos_1024` | primär | — / — | 12,1 / 24,7 | 22,2 / 43,3 | 27,8 / 45,5 | 32,1 / 50,3 | 44,0 / 65,4 |
| `F2_fcos_1024` | alle Objekte | 2,2 / 10,5 | 11,6 / 24,9 | 21,3 / 42,7 | 27,2 / 45,3 | 31,5 / 50,9 | 43,4 / 64,7 |
| `RG_fcos_1024` | primär | — / — | 12,1 / 24,7 | 22,2 / 43,3 | 27,8 / 45,5 | 32,1 / 50,3 | 44,0 / 65,4 |
| `RG_fcos_1024` | alle Objekte | 2,2 / 10,5 | 11,6 / 24,9 | 21,3 / 42,7 | 27,2 / 45,3 | 31,5 / 50,9 | 43,4 / 64,7 |
| `B-F2_fcos_1024` | primär | — / — | 5,4 / 9,7 | 7,9 / 20,0 | 18,2 / 34,3 | 24,9 / 43,9 | 40,6 / 65,7 |
| `B-F2_fcos_1024` | alle Objekte | 0,7 / 4,0 | 5,9 / 11,0 | 7,1 / 18,8 | 17,8 / 34,1 | 24,7 / 44,2 | 40,1 / 65,2 |
| `B-RG_fcos_1024` | primär | — / — | 5,4 / 9,7 | 8,4 / 20,9 | 18,6 / 35,0 | 25,8 / 44,3 | 41,3 / 66,0 |
| `B-RG_fcos_1024` | alle Objekte | 0,7 / 3,9 | 5,8 / 11,0 | 7,6 / 19,7 | 18,3 / 34,8 | 25,6 / 44,7 | 41,0 / 65,7 |
| `A_fcos_2048` | primär | — / — | 11,3 / 22,6 | 18,3 / 35,3 | 28,6 / 47,7 | 34,9 / 53,4 | 51,2 / 72,7 |
| `A_fcos_2048` | alle Objekte | 0,9 / 4,9 | 10,2 / 20,4 | 17,0 / 33,5 | 28,3 / 48,4 | 34,4 / 53,0 | 50,8 / 72,3 |
| `X_fcos_2048` | primär | — / — | 10,3 / 27,8 | 21,1 / 39,5 | 30,4 / 50,1 | 36,3 / 56,6 | 52,4 / 73,9 |
| `X_fcos_2048` | alle Objekte | 1,2 / 6,3 | 8,6 / 24,2 | 20,0 / 38,2 | 30,3 / 51,5 | 35,7 / 56,2 | 51,8 / 73,1 |
| `NMS-o2m_fcos_2048` | primär | — / — | 15,9 / 30,6 | 26,2 / 52,5 | 32,1 / 52,4 | 38,4 / 57,5 | 52,7 / 73,3 |
| `NMS-o2m_fcos_2048` | alle Objekte | 1,5 / 9,0 | 14,9 / 29,5 | 24,8 / 48,9 | 31,8 / 53,6 | 37,8 / 57,5 | 52,1 / 72,7 |
| `F2_fcos_2048` | primär | — / — | 16,2 / 33,0 | 26,5 / 49,5 | 33,4 / 52,9 | 37,6 / 57,8 | 50,0 / 73,1 |
| `F2_fcos_2048` | alle Objekte | 2,4 / 11,5 | 14,7 / 30,6 | 25,2 / 46,9 | 32,7 / 53,2 | 37,0 / 57,1 | 49,4 / 72,5 |
| `RG_fcos_2048` | primär | — / — | 16,2 / 33,0 | 26,5 / 49,5 | 33,4 / 52,9 | 37,6 / 57,8 | 50,0 / 73,1 |
| `RG_fcos_2048` | alle Objekte | 2,4 / 11,5 | 14,7 / 30,6 | 25,2 / 46,9 | 32,7 / 53,2 | 37,0 / 57,1 | 49,4 / 72,5 |
| `B-F2_fcos_2048` | primär | — / — | 11,2 / 22,6 | 17,5 / 34,5 | 27,5 / 46,8 | 33,1 / 53,1 | 47,4 / 69,7 |
| `B-F2_fcos_2048` | alle Objekte | 0,9 / 4,8 | 10,2 / 20,5 | 16,2 / 32,7 | 27,3 / 47,4 | 32,7 / 52,6 | 46,9 / 69,1 |
| `B-RG_fcos_2048` | primär | — / — | 11,2 / 22,6 | 17,5 / 34,5 | 27,5 / 46,8 | 33,1 / 53,1 | 47,4 / 69,7 |
| `B-RG_fcos_2048` | alle Objekte | 0,9 / 4,8 | 10,2 / 20,5 | 16,2 / 32,7 | 27,3 / 47,4 | 32,7 / 52,6 | 46,9 / 69,1 |

## Latenz

Nicht Teil der Freigabe (Addendum 3, Punkt 10); keine FCOS-Zeile in den Latenztabellen.

## Trainingsbedingungen

| s | Batch | Akkumulation | effektive Batch | Lernrate | Auswahl auf calib |
|---|---|---|---|---|---|
| 1024 | 8 | 1 | 8 | 0,005 | devkit mAP50 on calib (no cap), best epoch 35 (42.90) |
| 2048 | 1 | 8 | 8 | 0,005 | devkit mAP50 on calib (no cap), best epoch 43 (60.30) |

Rezept: `docs/PREREGISTRATION_DOTA.md` Addendum 3 (torchvision `fcos_resnet50_fpn`, COCO_V1, 60 Epochen, nur horizontales Spiegeln, Auswahl auf calib).
