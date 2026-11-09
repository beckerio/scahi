# Phase D – Ergebnisse auf DOTA-val (v1.5, HBB)

DOTA-val (458 Bilder), **einmal ausgewertet** nach dem freigegebenen Freeze (`docs/FREEZE_PROPOSAL_DOTA.md`, `configs/dota/frozen_rows.json`, Freigabe 2026-10-30, `docs/DECISIONS.md`). Pipeline-Grenze 200000 je Bild/Pass für alle Zeilen; alle 32 Zeilen, kein Nachtuning. Primär: offizielle Devkit-mAP50 ohne Cap. ε = 0,30; gepaarter Bootstrap über Bilder (1000 Resamples, Seed 0, Perzentil-KI), einseitiges p gegen ε, Holm über die zwei Modelle. Erzeugt von `scripts/dota/d_results.py` aus `runs/dota/val/results.json`.

## Hypothesen (primär: mAP50 ohne Cap)

Eine Hypothese gilt, wenn beide Modelle sie erfüllen (Kriterium und Holm-p ≤ 0,05).

### D-H1 – gilt nicht

- Präregistriert: „D-H1 (Auflösung vs. Kachelung): Δ = A_2048 − NMS_1024 (o2o); Kriterium Δ > ε.“
- Berechnet: A_2048 (Einzelpass 2048, o2o) − NMS-o2o_1024 (ganzes Bild 1024 + Kacheln, Box-Fusion IoS, o2o)

| Modell | a | b | Δ [95 %-KI] | p | Holm-p | Kriterium | erfüllt |
|---|---|---|---|---|---|---|---|
| YOLOv10s | 65,50 | 63,18 | +2,31 [-0,31; +4,95] | 0,060 | 0,060 | ja | nein |
| YOLOv10m | 67,39 | 63,97 | +3,42 [+0,74; +5,59] | 0,015 | 0,030 | ja | ja |

Dieselbe Differenz, sekundär (ohne Urteil): Δ [95 %-KI]

| Modell | mAP50 Cap 4000 | „alle Objekte“ | Fahrzeuge mAP50 | AP@[.5:.95] | Recall@0,5 crowded | Recall@0,5 isolated |
|---|---|---|---|---|---|---|
| YOLOv10s | +2,12 [-0,71; +4,89] | +0,33 [-2,00; +3,13] | -0,44 [-7,45; +8,18] | +3,42 [+1,40; +5,46] | +2,78 [+0,38; +5,32] | -1,42 [-5,35; +1,70] |
| YOLOv10m | +3,17 [+0,65; +5,50] | +1,89 [-0,51; +4,34] | +1,31 [-7,03; +10,10] | +3,04 [+1,18; +4,81] | +2,50 [+0,66; +4,82] | -0,71 [-4,27; +2,07] |

### D-H2a – gilt

- Präregistriert: „D-H2 (Fusion): Δ = RG_s − NMS_s (beide o2m, identische Pässe), getrennt für s = 1024 (D-H2a) und s = 2048 (D-H2b); Kriterium Δ > ε.“
- Berechnet: RG_1024 (SCAHI, verfeinertes Raster, o2m) − NMS-o2m_1024 (Box-Fusion IoS, o2m), identische Pässe dota1024

| Modell | a | b | Δ [95 %-KI] | p | Holm-p | Kriterium | erfüllt |
|---|---|---|---|---|---|---|---|
| YOLOv10s | 66,17 | 64,86 | +1,31 [+0,36; +2,39] | 0,024 | 0,024 | ja | ja |
| YOLOv10m | 67,34 | 64,57 | +2,77 [+1,67; +3,83] | 0,001 | 0,002 | ja | ja |

Dieselbe Differenz, sekundär (ohne Urteil): Δ [95 %-KI]

| Modell | mAP50 Cap 4000 | „alle Objekte“ | Fahrzeuge mAP50 | AP@[.5:.95] | Recall@0,5 crowded | Recall@0,5 isolated |
|---|---|---|---|---|---|---|
| YOLOv10s | +1,28 [+0,30; +2,29] | +1,18 [+0,44; +2,09] | +2,64 [-1,17; +7,56] | +2,13 [+1,46; +2,85] | -2,58 [-4,03; -1,23] | +2,80 [+2,34; +3,38] |
| YOLOv10m | +2,60 [+1,57; +3,76] | +2,52 [+1,66; +3,29] | +3,86 [-0,31; +9,62] | +3,15 [+2,35; +4,01] | -0,80 [-2,16; +1,30] | +3,29 [+2,76; +3,93] |

### D-H2b – gilt

- Präregistriert: wie D-H2a, s = 2048
- Berechnet: RG_2048 (SCAHI, o2m) − NMS-o2m_2048 (Box-Fusion IoS, o2m), identische Pässe dota2048

| Modell | a | b | Δ [95 %-KI] | p | Holm-p | Kriterium | erfüllt |
|---|---|---|---|---|---|---|---|
| YOLOv10s | 71,00 | 68,46 | +2,54 [+1,43; +3,34] | 0,001 | 0,002 | ja | ja |
| YOLOv10m | 71,73 | 68,13 | +3,60 [+2,29; +4,57] | 0,001 | 0,002 | ja | ja |

Dieselbe Differenz, sekundär (ohne Urteil): Δ [95 %-KI]

| Modell | mAP50 Cap 4000 | „alle Objekte“ | Fahrzeuge mAP50 | AP@[.5:.95] | Recall@0,5 crowded | Recall@0,5 isolated |
|---|---|---|---|---|---|---|
| YOLOv10s | +2,56 [+1,44; +3,39] | +2,47 [+1,43; +3,28] | +2,41 [-0,50; +6,01] | +2,72 [+2,02; +3,39] | -3,46 [-5,04; -1,64] | +2,45 [+2,02; +2,97] |
| YOLOv10m | +3,36 [+2,17; +4,51] | +3,49 [+2,29; +4,33] | +4,74 [+0,64; +7,38] | +3,46 [+2,65; +4,26] | +10,66 [+8,28; +12,81] | +3,09 [+2,60; +3,72] |

### D-H3 – gilt

- Präregistriert: „D-H3 (entscheidend): Δ = RG_{S_max} − A_{S_max} (SCAHI o2m mit Kacheln gegen den Einzelpass o2o, beide auf den S_max-Gewichten); Kriterium Δ > ε und untere KI-Grenze > 0.“ (S_max = 2048)
- Berechnet: RG_2048 (SCAHI o2m, ganzes Bild 2048 + Kacheln) − A_2048 (Einzelpass o2o), 2048er Gewichte

| Modell | a | b | Δ [95 %-KI] | p | Holm-p | Kriterium | erfüllt |
|---|---|---|---|---|---|---|---|
| YOLOv10s | 71,00 | 65,50 | +5,50 [+3,55; +7,77] | 0,001 | 0,002 | ja | ja |
| YOLOv10m | 71,73 | 67,39 | +4,34 [+3,02; +6,19] | 0,001 | 0,002 | ja | ja |

Dieselbe Differenz, sekundär (ohne Urteil): Δ [95 %-KI]

| Modell | mAP50 Cap 4000 | „alle Objekte“ | Fahrzeuge mAP50 | AP@[.5:.95] | Recall@0,5 crowded | Recall@0,5 isolated |
|---|---|---|---|---|---|---|
| YOLOv10s | +5,79 [+3,64; +8,18] | +6,32 [+4,54; +8,64] | +6,83 [+2,50; +10,92] | +5,48 [+4,17; +7,03] | -3,82 [-6,20; -1,18] | +4,95 [+1,93; +8,81] |
| YOLOv10m | +4,45 [+3,07; +6,24] | +5,04 [+3,63; +6,87] | +5,47 [+1,68; +9,84] | +5,17 [+3,84; +6,50] | -4,17 [-6,60; -1,79] | +4,32 [+1,63; +7,76] |

## Sekundäre Kontraste (vorab festgelegt, ohne Urteil)

„Anderer Kopf“: A ist an o2o, RG an o2m gebunden; nur der Box-Fusions-Arm wechselt den Kopf. Δ in mAP50 (ohne Cap) [95 %-KI].

| Kontrast | YOLOv10s | YOLOv10m |
|---|---|---|
| F2 statt RG: D-H2a | +1,31 [+0,25; +2,38] | +2,11 [+1,01; +3,30] |
| F2 statt RG: D-H2b | +2,18 [+0,94; +3,10] | +3,19 [+1,73; +4,10] |
| F2 statt RG: D-H3 | +5,14 [+3,23; +7,29] | +3,93 [+2,43; +5,72] |
| D-H1, Box-Fusion o2m | +0,64 [-1,86; +3,41] | +2,82 [+0,36; +5,05] |
| D-H2a, Box-Fusion o2o | +2,99 [+1,81; +4,00] | +3,37 [+1,97; +4,37] |
| D-H2b, Box-Fusion o2o | +2,70 [+1,56; +4,08] | +3,03 [+1,66; +4,06] |
| Inferenz-Kreuzung RG_2048 − X_2048 | +5,50 [+3,55; +7,77] | +4,34 [+3,02; +6,19] |
| Zerlegung B-F2_1024 − A_1024 | +0,49 [-0,27; +1,03] | +0,17 [-0,55; +1,02] |
| Zerlegung F2_1024 − B-F2_1024 | +9,87 [+7,35; +12,37] | +8,08 [+5,65; +10,54] |
| Zerlegung B-RG_1024 − A_1024 | +1,43 [+0,67; +1,86] | +0,85 [+0,45; +1,74] |
| Zerlegung RG_1024 − B-RG_1024 | +8,95 [+6,49; +11,57] | +8,05 [+5,39; +10,26] |
| Zerlegung B-F2_2048 − A_2048 | +1,09 [-0,26; +2,22] | +0,28 [-0,33; +1,20] |
| Zerlegung F2_2048 − B-F2_2048 | +4,06 [+2,36; +6,22] | +3,65 [+2,08; +5,26] |
| Zerlegung B-RG_2048 − A_2048 | +1,78 [+0,82; +2,73] | +1,02 [+0,54; +1,92] |
| Zerlegung RG_2048 − B-RG_2048 | +3,72 [+2,13; +5,89] | +3,32 [+1,88; +4,87] |
| SAHI − A_1024 | +3,41 [+0,06; +6,33] | +2,53 [-0,43; +5,75] |
| SAHI − NMS-o2o_1024 | -3,98 [-5,61; -2,65] | -3,01 [-4,43; -1,70] |

Inferenz-Kreuzung: auf calib wurde bei beiden Modellen die Trainingsgröße 2048 gewählt; X_2048 = A_2048, der Kontrast ist daher gleich D-H3.

## Alle Zeilen

| Zeile | Kopf | mAP50 | Cap 4000 (Bilder > Cap) | alle Objekte | AP@[.5:.95] | Fahrzeuge mAP50 | klassenagn. AP50 | Recall crowded / isolated | Bilder an der Grenze | max. Det./Bild |
|---|---|---|---|---|---|---|---|---|---|---|
| `A_yolov10s_1024` | o2o | **55,80** | 55,76 (36) | 50,59 | 36,14 | 66,87 | 70,91 | 91,4 / 82,2 | 0 | 17160 |
| `NMS-o2o_yolov10s_1024` | o2o | **63,18** | 63,15 (31) | 60,39 | 41,23 | 75,72 | 81,09 | 92,2 / 94,5 | 0 | 42332 |
| `NMS-o2m_yolov10s_1024` | o2m | **64,86** | 64,63 (33) | 61,53 | 43,50 | 77,13 | 82,31 | 95,2 / 94,8 | 0 | 43642 |
| `F2_yolov10s_1024` | o2m | **66,17** | 66,02 (79) | 62,65 | 44,75 | 76,35 | 82,71 | 66,6 / 96,0 | 0 | 92433 |
| `RG_yolov10s_1024` | o2m | **66,17** | 65,90 (107) | 62,71 | 45,63 | 79,76 | 84,00 | 92,6 / 97,6 | 0 | 133866 |
| `B-F2_yolov10s_1024` | o2m | **56,29** | 56,30 (31) | 50,94 | 36,44 | 63,87 | 68,73 | 59,7 / 78,5 | 0 | 18027 |
| `B-RG_yolov10s_1024` | o2m | **57,22** | 57,18 (76) | 51,88 | 37,31 | 67,91 | 71,10 | 82,9 / 80,9 | 0 | 31656 |
| `A_yolov10s_2048` | o2o | **65,50** | 65,27 (30) | 60,71 | 44,65 | 75,28 | 81,26 | 95,0 / 93,1 | 0 | 46218 |
| `NMS-o2o_yolov10s_2048` | o2o | **68,29** | 68,33 (15) | 64,43 | 46,85 | 79,77 | 85,14 | 94,5 / 95,9 | 0 | 42903 |
| `NMS-o2m_yolov10s_2048` | o2m | **68,46** | 68,49 (17) | 64,56 | 47,41 | 79,70 | 84,61 | 94,6 / 95,6 | 0 | 42254 |
| `F2_yolov10s_2048` | o2m | **70,64** | 70,65 (37) | 66,69 | 49,15 | 78,44 | 85,56 | 66,8 / 96,9 | 0 | 111888 |
| `RG_yolov10s_2048` | o2m | **71,00** | 71,06 (53) | 67,03 | 50,13 | 82,11 | 86,81 | 91,1 / 98,1 | 0 | 150621 |
| `B-F2_yolov10s_2048` | o2m | **66,58** | 66,13 (23) | 61,18 | 44,87 | 72,45 | 78,54 | 64,2 / 90,6 | 0 | 55038 |
| `B-RG_yolov10s_2048` | o2m | **67,27** | 66,96 (45) | 62,04 | 45,94 | 76,49 | 82,37 | 93,3 / 92,6 | 0 | 79764 |
| `A_yolov10m_1024` | o2o | **58,44** | 58,42 (33) | 52,82 | 38,37 | 69,63 | 73,77 | 91,8 / 83,6 | 0 | 12652 |
| `NMS-o2o_yolov10m_1024` | o2o | **63,97** | 63,82 (31) | 61,17 | 43,09 | 76,56 | 82,60 | 93,3 / 94,8 | 0 | 25185 |
| `NMS-o2m_yolov10m_1024` | o2m | **64,57** | 64,48 (27) | 61,37 | 43,87 | 76,95 | 82,78 | 93,5 / 94,7 | 0 | 23569 |
| `F2_yolov10m_1024` | o2m | **66,69** | 66,54 (81) | 63,21 | 46,33 | 77,42 | 83,65 | 69,8 / 96,5 | 0 | 65907 |
| `RG_yolov10m_1024` | o2m | **67,34** | 67,08 (185) | 63,88 | 47,02 | 80,81 | 84,97 | 92,7 / 98,0 | 0 | 145074 |
| `B-F2_yolov10m_1024` | o2m | **58,61** | 58,61 (33) | 52,98 | 38,74 | 66,83 | 70,17 | 63,4 / 81,0 | 0 | 15573 |
| `B-RG_yolov10m_1024` | o2m | **59,29** | 59,26 (50) | 53,60 | 39,50 | 70,66 | 74,82 | 88,5 / 83,0 | 0 | 21750 |
| `A_yolov10m_2048` | o2o | **67,39** | 66,99 (32) | 63,06 | 46,13 | 77,87 | 83,55 | 95,8 / 94,1 | 0 | 51100 |
| `NMS-o2o_yolov10m_2048` | o2o | **68,71** | 68,54 (16) | 65,63 | 47,74 | 81,27 | 86,10 | 91,1 / 95,9 | 0 | 38111 |
| `NMS-o2m_yolov10m_2048` | o2m | **68,13** | 68,08 (15) | 64,61 | 47,85 | 78,60 | 85,18 | 81,0 / 95,3 | 0 | 29584 |
| `F2_yolov10m_2048` | o2m | **71,32** | 71,08 (34) | 67,64 | 50,57 | 79,59 | 86,62 | 68,0 / 97,3 | 0 | 112494 |
| `RG_yolov10m_2048` | o2m | **71,73** | 71,44 (100) | 68,10 | 51,30 | 83,34 | 87,46 | 91,6 / 98,4 | 1 | 200000 |
| `B-F2_yolov10m_2048` | o2m | **67,67** | 67,20 (24) | 62,80 | 46,48 | 74,29 | 81,24 | 66,5 / 92,4 | 0 | 61752 |
| `B-RG_yolov10m_2048` | o2m | **68,42** | 68,02 (42) | 63,52 | 47,30 | 77,72 | 83,97 | 89,5 / 93,8 | 0 | 81216 |
| `X_yolov10s_2048` | o2o | **65,50** | 65,27 (30) | 60,71 | 44,65 | 75,28 | 81,26 | 95,0 / 93,1 | 0 | 46218 |
| `X_yolov10m_2048` | o2o | **67,39** | 66,99 (32) | 63,06 | 46,13 | 77,87 | 83,55 | 95,8 / 94,1 | 0 | 51100 |
| `SAHI_yolov10s` | o2o | **59,20** | 59,11 (28) | 56,31 | 37,46 | 74,83 | 78,95 | 90,6 / 93,7 | 0 | 34543 |
| `SAHI_yolov10m` | o2o | **60,96** | 60,87 (22) | 58,04 | 39,78 | 76,35 | 81,56 | 91,3 / 94,2 | 0 | 21343 |

Crowded / isolated: nicht-difficult-GT mit / ohne gleichklassigen Nachbarn (IoU > 0,3); 2515 crowded, 31489 isolated GT.

## Größenbins (σ = √(w·h) px)

Primär (difficult ignoriert; der Bin < 8 hat dann keine GT: „—“) und „alle Objekte“ (difficult als normale GT). AP@[.5:.95] / AP50.

| Zeile | Auswertung | <8 | 8-16 | 16-32 | 32-64 | 64-128 | >128 |
|---|---|---|---|---|---|---|---|
| `A_yolov10s_1024` | primär | — / — | 9,4 / 19,8 | 18,0 / 35,8 | 28,3 / 47,2 | 37,3 / 57,6 | 59,6 / 76,7 |
| `A_yolov10s_1024` | alle Objekte | 1,9 / 10,6 | 9,4 / 20,2 | 16,7 / 34,0 | 27,5 / 46,2 | 37,3 / 57,6 | 59,3 / 76,2 |
| `NMS-o2o_yolov10s_1024` | primär | — / — | 19,2 / 36,6 | 29,6 / 52,7 | 35,8 / 55,0 | 41,5 / 61,9 | 56,1 / 75,4 |
| `NMS-o2o_yolov10s_1024` | alle Objekte | 7,6 / 24,6 | 18,6 / 37,5 | 29,0 / 51,9 | 35,3 / 55,0 | 41,1 / 61,5 | 55,6 / 74,9 |
| `NMS-o2m_yolov10s_1024` | primär | — / — | 15,8 / 34,6 | 29,9 / 51,7 | 37,3 / 56,5 | 43,1 / 62,8 | 59,2 / 77,6 |
| `NMS-o2m_yolov10s_1024` | alle Objekte | 7,7 / 24,7 | 15,2 / 34,2 | 29,4 / 51,1 | 36,4 / 56,1 | 42,7 / 62,5 | 58,7 / 76,9 |
| `F2_yolov10s_1024` | primär | — / — | 20,3 / 41,1 | 31,4 / 53,8 | 38,6 / 57,7 | 44,4 / 63,6 | 59,3 / 76,8 |
| `F2_yolov10s_1024` | alle Objekte | 7,9 / 26,5 | 19,4 / 41,0 | 30,5 / 52,7 | 37,9 / 57,6 | 44,1 / 63,6 | 58,6 / 76,0 |
| `RG_yolov10s_1024` | primär | — / — | 20,0 / 40,7 | 31,4 / 53,3 | 39,4 / 57,9 | 45,2 / 64,2 | 62,0 / 78,7 |
| `RG_yolov10s_1024` | alle Objekte | 7,9 / 26,2 | 19,3 / 40,7 | 30,5 / 52,3 | 38,4 / 57,6 | 45,2 / 64,2 | 61,5 / 78,1 |
| `B-F2_yolov10s_1024` | primär | — / — | 8,1 / 19,7 | 18,2 / 36,5 | 28,3 / 47,3 | 37,1 / 56,8 | 57,1 / 75,6 |
| `B-F2_yolov10s_1024` | alle Objekte | 1,5 / 9,2 | 8,1 / 20,5 | 16,9 / 34,6 | 27,6 / 46,3 | 37,4 / 57,6 | 56,6 / 75,0 |
| `B-RG_yolov10s_1024` | primär | — / — | 8,1 / 19,8 | 18,4 / 37,1 | 28,9 / 47,7 | 38,3 / 58,5 | 59,1 / 77,1 |
| `B-RG_yolov10s_1024` | alle Objekte | 1,5 / 9,5 | 8,1 / 20,7 | 17,1 / 35,0 | 28,2 / 46,8 | 38,8 / 59,2 | 58,7 / 76,8 |
| `A_yolov10s_2048` | primär | — / — | 17,2 / 37,5 | 27,9 / 50,0 | 39,3 / 59,6 | 45,5 / 63,4 | 63,5 / 79,2 |
| `A_yolov10s_2048` | alle Objekte | 3,7 / 16,1 | 14,2 / 32,2 | 26,8 / 48,9 | 38,9 / 60,3 | 45,7 / 63,8 | 63,2 / 79,0 |
| `NMS-o2o_yolov10s_2048` | primär | — / — | 22,8 / 44,3 | 36,0 / 63,7 | 40,4 / 60,0 | 46,1 / 64,9 | 63,1 / 79,8 |
| `NMS-o2o_yolov10s_2048` | alle Objekte | 6,5 / 22,4 | 20,7 / 42,4 | 35,1 / 61,7 | 40,0 / 60,1 | 45,9 / 64,9 | 62,7 / 79,4 |
| `NMS-o2m_yolov10s_2048` | primär | — / — | 21,2 / 42,1 | 36,1 / 59,9 | 41,1 / 59,4 | 47,7 / 66,6 | 63,9 / 81,3 |
| `NMS-o2m_yolov10s_2048` | alle Objekte | 7,2 / 24,7 | 19,8 / 40,9 | 35,4 / 59,0 | 40,2 / 59,5 | 47,4 / 66,5 | 63,5 / 81,0 |
| `F2_yolov10s_2048` | primär | — / — | 26,0 / 48,5 | 37,9 / 63,3 | 43,0 / 62,4 | 48,7 / 67,9 | 63,2 / 79,6 |
| `F2_yolov10s_2048` | alle Objekte | 8,6 / 28,6 | 22,8 / 44,8 | 37,0 / 62,6 | 41,9 / 61,3 | 48,7 / 68,1 | 62,5 / 78,8 |
| `RG_yolov10s_2048` | primär | — / — | 24,6 / 46,4 | 38,1 / 62,4 | 43,6 / 62,6 | 49,8 / 68,3 | 65,6 / 81,3 |
| `RG_yolov10s_2048` | alle Objekte | 8,6 / 29,2 | 21,9 / 44,0 | 37,2 / 61,8 | 42,5 / 61,7 | 49,8 / 68,5 | 65,0 / 80,8 |
| `B-F2_yolov10s_2048` | primär | — / — | 15,5 / 34,9 | 28,3 / 51,3 | 38,4 / 58,1 | 45,4 / 65,1 | 62,0 / 78,1 |
| `B-F2_yolov10s_2048` | alle Objekte | 3,9 / 16,0 | 13,5 / 32,0 | 27,0 / 49,7 | 37,7 / 58,1 | 45,4 / 65,2 | 61,4 / 77,1 |
| `B-RG_yolov10s_2048` | primär | — / — | 15,5 / 33,8 | 28,5 / 51,0 | 39,3 / 58,9 | 46,8 / 66,3 | 65,7 / 80,5 |
| `B-RG_yolov10s_2048` | alle Objekte | 3,8 / 16,2 | 13,9 / 31,8 | 27,2 / 49,6 | 38,6 / 58,9 | 46,8 / 66,3 | 65,3 / 80,1 |
| `A_yolov10m_1024` | primär | — / — | 11,2 / 20,8 | 20,0 / 38,7 | 30,8 / 49,2 | 40,7 / 60,5 | 61,5 / 78,4 |
| `A_yolov10m_1024` | alle Objekte | 3,2 / 15,8 | 10,9 / 20,2 | 18,6 / 36,9 | 30,3 / 49,0 | 40,4 / 60,7 | 61,3 / 78,2 |
| `NMS-o2o_yolov10m_1024` | primär | — / — | 14,2 / 28,7 | 31,5 / 55,2 | 37,8 / 55,6 | 43,0 / 62,8 | 59,1 / 76,7 |
| `NMS-o2o_yolov10m_1024` | alle Objekte | 9,8 / 30,5 | 13,1 / 28,9 | 31,0 / 54,5 | 37,7 / 56,1 | 42,5 / 62,6 | 58,6 / 76,4 |
| `NMS-o2m_yolov10m_1024` | primär | — / — | 18,7 / 35,5 | 32,1 / 55,0 | 38,6 / 56,7 | 43,7 / 62,2 | 60,4 / 78,8 |
| `NMS-o2m_yolov10m_1024` | alle Objekte | 9,9 / 29,1 | 17,6 / 35,0 | 31,5 / 53,9 | 38,3 / 57,1 | 43,2 / 62,0 | 59,9 / 78,3 |
| `F2_yolov10m_1024` | primär | — / — | 20,4 / 38,4 | 34,6 / 58,3 | 39,7 / 58,2 | 46,0 / 64,8 | 62,6 / 79,3 |
| `F2_yolov10m_1024` | alle Objekte | 11,0 / 31,6 | 18,9 / 37,0 | 33,7 / 57,3 | 39,5 / 58,8 | 45,6 / 64,6 | 62,1 / 78,6 |
| `RG_yolov10m_1024` | primär | — / — | 20,2 / 37,8 | 34,6 / 58,0 | 40,4 / 58,8 | 46,9 / 65,6 | 63,9 / 81,9 |
| `RG_yolov10m_1024` | alle Objekte | 10,8 / 31,9 | 18,8 / 36,6 | 33,8 / 57,1 | 40,3 / 59,4 | 46,6 / 65,5 | 63,5 / 81,2 |
| `B-F2_yolov10m_1024` | primär | — / — | 10,6 / 21,3 | 20,2 / 38,8 | 30,8 / 50,1 | 40,7 / 60,5 | 59,8 / 76,5 |
| `B-F2_yolov10m_1024` | alle Objekte | 2,4 / 10,8 | 10,2 / 20,2 | 18,7 / 36,9 | 30,1 / 49,4 | 40,5 / 60,7 | 59,4 / 76,3 |
| `B-RG_yolov10m_1024` | primär | — / — | 10,6 / 21,4 | 20,3 / 39,1 | 31,4 / 50,6 | 42,0 / 61,6 | 62,2 / 78,8 |
| `B-RG_yolov10m_1024` | alle Objekte | 2,5 / 12,0 | 10,2 / 20,2 | 18,9 / 37,3 | 30,7 / 50,0 | 41,9 / 62,1 | 61,9 / 78,6 |
| `A_yolov10m_2048` | primär | — / — | 16,1 / 36,8 | 34,0 / 63,0 | 41,1 / 61,2 | 47,6 / 65,1 | 62,8 / 78,3 |
| `A_yolov10m_2048` | alle Objekte | 5,1 / 16,6 | 14,6 / 36,7 | 32,0 / 60,1 | 40,5 / 61,9 | 47,9 / 66,0 | 62,6 / 77,9 |
| `NMS-o2o_yolov10m_2048` | primär | — / — | 21,9 / 43,2 | 37,5 / 66,0 | 43,2 / 63,9 | 48,1 / 66,5 | 60,5 / 76,2 |
| `NMS-o2o_yolov10m_2048` | alle Objekte | 7,5 / 23,2 | 20,9 / 45,3 | 36,3 / 64,5 | 42,7 / 64,7 | 47,8 / 66,9 | 60,1 / 75,6 |
| `NMS-o2m_yolov10m_2048` | primär | — / — | 21,8 / 43,7 | 36,4 / 62,0 | 42,2 / 61,7 | 48,1 / 67,0 | 60,2 / 75,0 |
| `NMS-o2m_yolov10m_2048` | alle Objekte | 7,0 / 20,2 | 19,8 / 41,7 | 35,6 / 61,3 | 41,5 / 61,6 | 47,9 / 67,4 | 59,8 / 74,3 |
| `F2_yolov10m_2048` | primär | — / — | 26,1 / 48,1 | 38,9 / 64,1 | 44,5 / 64,2 | 49,9 / 67,6 | 62,9 / 78,5 |
| `F2_yolov10m_2048` | alle Objekte | 9,3 / 28,2 | 22,6 / 44,1 | 37,9 / 63,7 | 43,7 / 63,8 | 49,7 / 67,9 | 62,5 / 77,8 |
| `RG_yolov10m_2048` | primär | — / — | 26,4 / 47,7 | 39,0 / 64,4 | 45,1 / 64,5 | 51,1 / 68,4 | 64,5 / 79,2 |
| `RG_yolov10m_2048` | alle Objekte | 9,3 / 28,4 | 22,9 / 44,5 | 38,1 / 64,0 | 44,3 / 64,2 | 50,9 / 68,7 | 64,1 / 78,5 |
| `B-F2_yolov10m_2048` | primär | — / — | 17,4 / 40,6 | 33,5 / 59,6 | 40,0 / 60,5 | 47,2 / 65,7 | 60,2 / 75,9 |
| `B-F2_yolov10m_2048` | alle Objekte | 4,8 / 17,4 | 15,0 / 35,7 | 32,1 / 58,3 | 39,2 / 60,0 | 47,8 / 66,6 | 59,8 / 74,9 |
| `B-RG_yolov10m_2048` | primär | — / — | 17,4 / 40,2 | 34,2 / 60,2 | 40,8 / 61,2 | 48,5 / 66,9 | 62,8 / 78,9 |
| `B-RG_yolov10m_2048` | alle Objekte | 4,7 / 16,3 | 15,0 / 35,6 | 32,7 / 58,8 | 40,0 / 61,3 | 49,0 / 67,9 | 62,5 / 78,2 |
| `X_yolov10s_2048` | primär | — / — | 17,2 / 37,5 | 27,9 / 50,0 | 39,3 / 59,6 | 45,5 / 63,4 | 63,5 / 79,2 |
| `X_yolov10s_2048` | alle Objekte | 3,7 / 16,1 | 14,2 / 32,2 | 26,8 / 48,9 | 38,9 / 60,3 | 45,7 / 63,8 | 63,2 / 79,0 |
| `X_yolov10m_2048` | primär | — / — | 16,1 / 36,8 | 34,0 / 63,0 | 41,1 / 61,2 | 47,6 / 65,1 | 62,8 / 78,3 |
| `X_yolov10m_2048` | alle Objekte | 5,1 / 16,6 | 14,6 / 36,7 | 32,0 / 60,1 | 40,5 / 61,9 | 47,9 / 66,0 | 62,6 / 77,9 |
| `SAHI_yolov10s` | primär | — / — | 15,3 / 33,1 | 28,6 / 51,5 | 35,3 / 54,6 | 37,9 / 55,7 | 48,4 / 71,6 |
| `SAHI_yolov10s` | alle Objekte | 6,8 / 21,8 | 14,9 / 34,4 | 28,0 / 50,7 | 34,2 / 53,7 | 37,4 / 55,6 | 48,0 / 71,0 |
| `SAHI_yolov10m` | primär | — / — | 15,4 / 31,5 | 32,0 / 56,0 | 35,6 / 53,8 | 39,6 / 57,5 | 52,8 / 72,8 |
| `SAHI_yolov10m` | alle Objekte | 9,1 / 28,5 | 14,4 / 31,2 | 31,4 / 55,6 | 35,4 / 54,0 | 38,6 / 56,8 | 52,4 / 72,3 |

## Latenz und GFLOPs der D-H3-Zeilen

Status: **final** (GPU exklusiv: ja, CPU exklusiv: ja, Leistungsgrenze 575 W, Standard); Maschine host-b. GFLOPs: Ultralytics get_flops on the fused model at the exact letterboxed input shape, summed over the passes of each image, mean over the images of the split.

| Zeile | Pfad | Latenz ms (Mittel / p95) | davon Pässe / Fusion | Spitzenspeicher MB | GFLOPs/Bild |
|---|---|---|---|---|---|
| `A_yolov10s_2048` | Ultralytics predictor | 28 / 67 | — | 693 | 178 |
| `RG_yolov10s_2048` | reference pipeline (unfused model, GPU passes, CPU SCAHI) | 284 / 625 | 122 / 161 | 2043 | 985 |
| `A_yolov10m_2048` | Ultralytics predictor | 34 / 73 | — | 908 | 476 |
| `RG_yolov10m_2048` | reference pipeline (unfused model, GPU passes, CPU SCAHI) | 385 / 990 | 173 / 212 | 3228 | 2672 |

Pfade wie in der VisDrone-Referenzspalte: A mit dem Ultralytics-Prädiktor (fusionierte Gewichte, o2o); RG mit der Referenz-Pipeline (unfusioniertes Modell mit beiden Köpfen, Pässe von dota2048 auf der GPU, SCAHI auf der CPU, ein Bild nach dem anderen), die auf 3 calib-Bildern die gecachten Detektionen reproduziert. X_2048 = A_2048. GFLOPs zählen das fusionierte Ein-Kopf-Modell je Pass.

## Explorativ, post hoc: Wirkung der Pipeline-Grenze

Zeilen mit Bildern an der Grenze 200000: dieselben Bilder ohne wirksame Grenze neu fusioniert (`scripts/dota/d_limit_check.py`); berichtete Zahlen oben unverändert.

| Zeile | Bild: Detektionen mit / ohne Grenze | mAP50 mit Grenze | ohne Grenze | Δ |
|---|---|---|---|---|
| `RG_yolov10m_2048` | P1384: 200000 / 215568 | 71,7327 | 71,7327 | 0,0000 |

## Trainingsbedingungen

| Modell | s | Batch | effektive Batch | gewählte Epoche (calib-mAP50) | Mittel ± SD Epochen 41–60 |
|---|---|---|---|---|---|
| YOLOv10s | 1024 | 8 | 64 | 49 (64,81) | 63,21 ± 0,68 |
| YOLOv10s | 2048 | 3 | 63 | 47 (73,70) | 71,62 ± 1,52 |
| YOLOv10m | 1024 | 8 | 64 | 42 (68,24) | 67,42 ± 0,41 |
| YOLOv10m | 2048 | 3 | 63 | 48 (72,02) | 70,06 ± 1,22 |

- Batch fest in allen 60 Epochen (Batch-Wache nie ausgelöst, kein OOM-Fallback, kein Resume); keine NaN/Inf, keine übersprungenen Batches.
- **OOM-Rückfall bei 2048 px:** In jeder der 60 Epochen meldet der CUDA-Allokator fehlgeschlagene Allokationen (insgesamt YOLOv10s 423, YOLOv10m 1008 Meldungen auf je 25 380 Iterationen; bei 1024: 10 bzw. 26). Sie werden vom Allokator (Cache freigeben, erneut versuchen) oder von Ultralytics' Rückfall der Label-Zuordnung (`TaskAlignedAssigner`: dieselbe Zuordnung Bild für Bild auf der GPU, Forward-Batch unverändert) abgefangen. Laut Code ist die Zuordnung identisch; die Ergebnisse sind nicht betroffen.
- **Befund (calib, vor DOTA-val festgehalten):** YOLOv10m liegt bei 2048 über die ganze Spätphase rund 1,6 mAP50-Punkte unter YOLOv10s, bei 1024 klar darüber; die 2048er Kurven schwanken stärker. Mögliche Erklärungen, **nicht geprüft**: Batch 3 (Batch-Norm-Statistik) trifft das größere Modell stärker; Kapazität/Überanpassung von YOLOv10m bei 2048; Rauschen der kleinen calib-Menge (141 Bilder).
