# Phase D, Addendum 2 – YOLOv11s und YOLOv12s auf DOTA-val (v1.5, HBB)

DOTA-val (458 Bilder), **einmal ausgewertet** nach dem freigegebenen Freeze der neuen Zeilen (`docs/FREEZE_PROPOSAL_DOTA_NEW.md`, `configs/dota/frozen_rows_new.json`, Freigabe 2026-10-31, `docs/DECISIONS.md`). Alle 19 Zeilen, kein Nachtuning; Pipeline-Grenze 200000. Berichtsregeln wie im ursprünglichen Freeze: primär Devkit-mAP50 ohne Cap, ε = 0,30, gepaarter Bootstrap über Bilder (1000 Resamples, Seed 0, Perzentil-KI), einseitiges p gegen ε; **Holm über die zwei neuen Modelle**. Ein-Kopf-Modelle: Einzelpass, Box-Fusion und SCAHI nutzen den einzigen (dichten) Kopf. S_max (Speicher-Smoke): YOLOv10s 2048, YOLOv10m 2048, YOLOv11s 2048, YOLOv12s 1024. Die Urteile der ursprünglichen zwei Modelle (`reports/D_RESULTS.md`) bleiben unverändert. Erzeugt von `scripts/dota/d_results_new.py` aus `runs/dota/val/results_new.json`.

## Hypothesen (primär: mAP50 ohne Cap)

Urteil je Modell (Kriterium und Holm-p ≤ 0,05). Nicht prüfbar (YOLOv12s ohne 2048er Zeilen, S_max = 1024): zählt als nicht erfüllt (Holm-p = 1). Die Zählung über vier Modelle ist deskriptiv.

| Hypothese | YOLOv11s | YOLOv12s | erfüllt (neu, von 2) | erfüllt (alle, von 4, deskriptiv) | ursprüngliches Urteil (YOLOv10s + YOLOv10m), unverändert |
|---|---|---|---|---|---|
| D-H1 | erfüllt | nicht prüfbar | 1 | 2 | gilt nicht (YOLOv10s nein, YOLOv10m ja) |
| D-H2a | erfüllt | erfüllt | 2 | 4 | gilt (YOLOv10s ja, YOLOv10m ja) |
| D-H2b | erfüllt | nicht prüfbar | 1 | 3 | gilt (YOLOv10s ja, YOLOv10m ja) |
| D-H3 | erfüllt | erfüllt (+9,62 [+7,10; +12,22], auf dem 1024er Pass = S_max) | 2 | 4 | gilt (YOLOv10s ja, YOLOv10m ja) |

### D-H1

| Modell | Kontrast | a | b | Δ [95 %-KI] | p | Holm-p | Kriterium | erfüllt |
|---|---|---|---|---|---|---|---|---|
| YOLOv11s | A_yolo11s_2048 − NMS-o2m_yolo11s_1024 | 65,19 | 62,14 | +3,05 [+0,55; +5,81] | 0,018 | 0,036 | ja | ja |
| YOLOv12s | A_yolo12s_2048 − NMS-o2m_yolo12s_1024 | — | — | nicht prüfbar | — | 1,000 | nein | nein |

Sekundär (ohne Urteil): Δ [95 %-KI]

| Modell | mAP50 Cap 4000 | „alle Objekte“ | Fahrzeuge mAP50 | AP@[.5:.95] | Recall@0,5 crowded | Recall@0,5 isolated |
|---|---|---|---|---|---|---|
| YOLOv11s | +3,06 [+0,59; +5,78] | +1,04 [-1,06; +3,92] | +0,45 [-6,95; +9,21] | +3,63 [+1,69; +5,58] | +3,66 [+0,17; +7,81] | -2,04 [-6,23; +1,29] |

### D-H2a

| Modell | Kontrast | a | b | Δ [95 %-KI] | p | Holm-p | Kriterium | erfüllt |
|---|---|---|---|---|---|---|---|---|
| YOLOv11s | RG_yolo11s_1024 − NMS-o2m_yolo11s_1024 | 65,26 | 62,14 | +3,12 [+2,02; +4,18] | 0,001 | 0,002 | ja | ja |
| YOLOv12s | RG_yolo12s_1024 − NMS-o2m_yolo12s_1024 | 65,97 | 62,88 | +3,09 [+2,11; +4,43] | 0,001 | 0,002 | ja | ja |

Sekundär (ohne Urteil): Δ [95 %-KI]

| Modell | mAP50 Cap 4000 | „alle Objekte“ | Fahrzeuge mAP50 | AP@[.5:.95] | Recall@0,5 crowded | Recall@0,5 isolated |
|---|---|---|---|---|---|---|
| YOLOv11s | +3,17 [+2,04; +4,17] | +2,43 [+1,84; +3,56] | +3,58 [-0,31; +8,75] | +3,55 [+2,68; +4,37] | +0,56 [-1,82; +4,37] | +3,00 [+2,54; +3,59] |
| YOLOv12s | +3,08 [+1,97; +4,33] | +2,90 [+2,03; +3,97] | +3,57 [-0,15; +8,75] | +3,14 [+2,34; +3,99] | -1,43 [-4,06; +2,40] | +2,98 [+2,52; +3,56] |

### D-H2b

| Modell | Kontrast | a | b | Δ [95 %-KI] | p | Holm-p | Kriterium | erfüllt |
|---|---|---|---|---|---|---|---|---|
| YOLOv11s | RG_yolo11s_2048 − NMS-o2m_yolo11s_2048 | 69,81 | 66,68 | +3,12 [+2,00; +3,94] | 0,001 | 0,002 | ja | ja |
| YOLOv12s | RG_yolo12s_2048 − NMS-o2m_yolo12s_2048 | — | — | nicht prüfbar | — | 1,000 | nein | nein |

Sekundär (ohne Urteil): Δ [95 %-KI]

| Modell | mAP50 Cap 4000 | „alle Objekte“ | Fahrzeuge mAP50 | AP@[.5:.95] | Recall@0,5 crowded | Recall@0,5 isolated |
|---|---|---|---|---|---|---|
| YOLOv11s | +3,16 [+1,92; +3,93] | +3,10 [+1,96; +3,72] | +2,77 [+0,16; +6,08] | +3,44 [+2,67; +4,18] | +1,99 [+0,58; +3,67] | +2,53 [+2,10; +3,05] |

### D-H3

| Modell | Kontrast | a | b | Δ [95 %-KI] | p | Holm-p | Kriterium | erfüllt |
|---|---|---|---|---|---|---|---|---|
| YOLOv11s | RG_yolo11s_2048 − A_yolo11s_2048 | 69,81 | 65,19 | +4,62 [+2,76; +6,62] | 0,001 | 0,002 | ja | ja |
| YOLOv12s (1024er Pass, S_max) | RG_yolo12s_1024 − A_yolo12s_1024 | 65,97 | 56,36 | +9,62 [+7,10; +12,22] | 0,001 | 0,002 | ja | ja |

YOLOv12s: D-H3 auf dem 1024er Pass (S_max = 1024; das 2048er Training passt nicht in den Speicher). Der Einzelpass bei 1024 ist deutlich schwächer als bei 2048 (YOLOv11s: A_1024 56,16, A_2048 65,19; für YOLOv12s gibt es keinen 2048er Einzelpass), daher fällt der Gewinn höher aus als bei den 2048er Modellen (YOLOv10s +5,50, YOLOv10m +4,34, YOLOv11s +4,62); nicht direkt vergleichbar.

Sekundär (ohne Urteil): Δ [95 %-KI]

| Modell | mAP50 Cap 4000 | „alle Objekte“ | Fahrzeuge mAP50 | AP@[.5:.95] | Recall@0,5 crowded | Recall@0,5 isolated |
|---|---|---|---|---|---|---|
| YOLOv11s | +4,68 [+2,76; +6,66] | +5,73 [+3,97; +7,92] | +6,85 [+2,06; +10,82] | +4,67 [+3,43; +6,20] | +1,11 [-1,01; +4,05] | +6,04 [+2,70; +10,04] |
| YOLOv12s | +9,52 [+7,03; +11,89] | +11,39 [+9,09; +13,96] | +12,64 [+5,68; +18,89] | +8,69 [+6,74; +10,60] | +1,99 [-2,46; +8,41] | +17,61 [+11,21; +24,64] |

## Sekundäre Kontraste (vorab festgelegt, ohne Urteil)

Ohne die o2o- und SAHI-Kontraste (Ein-Kopf-Modelle, kein SAHI; Addendum 2). Δ in mAP50 (ohne Cap) [95 %-KI].

| Kontrast | YOLOv11s | YOLOv12s |
|---|---|---|
| F2 statt RG: D-H2a | +3,06 [+1,74; +4,12] | +2,30 [+1,25; +3,29] |
| F2 statt RG: D-H2b | +2,46 [+1,32; +3,29] | — |
| F2 statt RG: D-H3 | +3,96 [+2,01; +5,95] | — |
| D-H1, Box-Fusion o2m | +3,05 [+0,55; +5,81] | — |
| Inferenz-Kreuzung RG_2048 − X_2048 | +4,62 [+2,76; +6,62] | — |
| Zerlegung B-F2_1024 − A_1024 | -0,48 [-1,02; +0,16] | -0,58 [-1,29; -0,06] |
| Zerlegung F2_1024 − B-F2_1024 | +9,53 [+7,04; +11,91] | +9,40 [+6,63; +11,90] |
| Zerlegung B-RG_1024 − A_1024 | +0,31 [+0,09; +0,86] | +0,38 [+0,04; +0,78] |
| Zerlegung RG_1024 − B-RG_1024 | +8,79 [+6,30; +11,29] | +9,23 [+6,73; +11,82] |
| Zerlegung B-F2_2048 − A_2048 | -0,27 [-1,01; +0,25] | — |
| Zerlegung F2_2048 − B-F2_2048 | +4,23 [+2,39; +6,18] | — |
| Zerlegung B-RG_2048 − A_2048 | +0,26 [-0,06; +0,77] | — |
| Zerlegung RG_2048 − B-RG_2048 | +4,36 [+2,60; +6,26] | — |

Inferenz-Kreuzung (nur YOLOv11s): auf calib wurde die Trainingsgröße 2048 gewählt; X_2048 = A_2048, der Kontrast ist daher gleich D-H3.

## Alle Zeilen

| Zeile | mAP50 | Cap 4000 (Bilder > Cap) | alle Objekte | AP@[.5:.95] | Fahrzeuge mAP50 | klassenagn. AP50 | Recall crowded / isolated | Bilder an der Grenze | max. Det./Bild |
|---|---|---|---|---|---|---|---|---|---|
| `A_yolo11s_1024` | **56,16** | 56,16 (16) | 50,91 | 36,30 | 67,33 | 72,16 | 89,2 / 80,1 | 0 | 12315 |
| `NMS-o2m_yolo11s_1024` | **62,14** | 61,92 (29) | 59,24 | 41,06 | 75,85 | 81,15 | 89,7 / 94,0 | 0 | 37140 |
| `F2_yolo11s_1024` | **65,20** | 65,07 (78) | 61,56 | 43,85 | 75,99 | 82,53 | 67,8 / 96,0 | 0 | 95556 |
| `RG_yolo11s_1024` | **65,26** | 65,09 (91) | 61,67 | 44,61 | 79,43 | 83,88 | 90,3 / 97,0 | 0 | 122946 |
| `B-F2_yolo11s_1024` | **55,68** | 55,64 (30) | 50,54 | 35,76 | 63,53 | 68,96 | 61,3 / 79,0 | 0 | 22041 |
| `B-RG_yolo11s_1024` | **56,47** | 56,46 (55) | 51,34 | 36,45 | 67,69 | 71,96 | 87,8 / 81,3 | 0 | 29538 |
| `A_yolo11s_2048` | **65,19** | 64,98 (23) | 60,28 | 44,69 | 76,30 | 82,55 | 93,4 / 92,0 | 0 | 29391 |
| `NMS-o2m_yolo11s_2048` | **66,68** | 66,49 (17) | 62,91 | 45,93 | 80,38 | 84,57 | 92,5 / 95,5 | 0 | 43493 |
| `F2_yolo11s_2048` | **69,14** | 68,97 (39) | 65,38 | 48,42 | 79,04 | 85,54 | 66,8 / 96,8 | 0 | 124452 |
| `RG_yolo11s_2048` | **69,81** | 69,66 (48) | 66,01 | 49,36 | 83,15 | 86,97 | 94,5 / 98,0 | 0 | 160533 |
| `B-F2_yolo11s_2048` | **64,92** | 64,71 (24) | 59,74 | 44,02 | 73,25 | 78,49 | 63,9 / 91,2 | 0 | 65559 |
| `B-RG_yolo11s_2048` | **65,45** | 65,24 (46) | 60,62 | 44,94 | 77,11 | 82,39 | 93,1 / 93,2 | 0 | 88365 |
| `A_yolo12s_1024` | **56,36** | 56,36 (15) | 51,27 | 36,62 | 67,56 | 72,60 | 88,2 / 79,7 | 0 | 8943 |
| `NMS-o2m_yolo12s_1024` | **62,88** | 62,79 (22) | 59,75 | 42,17 | 76,63 | 82,70 | 91,6 / 94,4 | 0 | 27900 |
| `F2_yolo12s_1024` | **65,18** | 65,07 (68) | 61,80 | 44,16 | 76,82 | 83,30 | 67,3 / 95,9 | 0 | 70962 |
| `RG_yolo12s_1024` | **65,97** | 65,88 (83) | 62,65 | 45,31 | 80,20 | 84,76 | 90,2 / 97,4 | 0 | 93723 |
| `B-F2_yolo12s_1024` | **55,78** | 55,69 (22) | 50,50 | 35,92 | 64,47 | 69,17 | 60,2 / 78,7 | 0 | 11418 |
| `B-RG_yolo12s_1024` | **56,74** | 56,69 (60) | 51,70 | 36,95 | 68,70 | 72,17 | 87,4 / 81,5 | 0 | 24423 |
| `X_yolo11s_2048` | **65,19** | 64,98 (23) | 60,28 | 44,69 | 76,30 | 82,55 | 93,4 / 92,0 | 0 | 29391 |

## Größenbins (σ = √(w·h) px)

Primär (difficult ignoriert; der Bin < 8 hat dann keine GT: „—“) und „alle Objekte“. AP@[.5:.95] / AP50.

| Zeile | Auswertung | <8 | 8-16 | 16-32 | 32-64 | 64-128 | >128 |
|---|---|---|---|---|---|---|---|
| `A_yolo11s_1024` | primär | — / — | 7,4 / 19,8 | 18,8 / 36,6 | 28,3 / 46,9 | 38,9 / 58,5 | 59,7 / 77,9 |
| `A_yolo11s_1024` | alle Objekte | 1,3 / 7,2 | 7,4 / 19,8 | 17,4 / 34,6 | 27,5 / 46,6 | 38,8 / 58,5 | 59,5 / 77,5 |
| `NMS-o2m_yolo11s_1024` | primär | — / — | 16,5 / 35,1 | 30,1 / 51,4 | 36,8 / 56,9 | 42,0 / 61,0 | 54,8 / 73,2 |
| `NMS-o2m_yolo11s_1024` | alle Objekte | 7,2 / 25,8 | 16,1 / 36,1 | 29,4 / 50,6 | 35,6 / 55,9 | 41,7 / 61,0 | 54,1 / 72,5 |
| `F2_yolo11s_1024` | primär | — / — | 17,3 / 36,0 | 34,0 / 57,7 | 39,4 / 58,9 | 44,2 / 63,2 | 57,9 / 76,9 |
| `F2_yolo11s_1024` | alle Objekte | 7,3 / 24,9 | 17,1 / 37,4 | 33,2 / 56,8 | 37,7 / 57,4 | 43,9 / 63,1 | 57,3 / 75,6 |
| `RG_yolo11s_1024` | primär | — / — | 17,4 / 36,3 | 34,2 / 57,7 | 40,0 / 59,1 | 45,2 / 63,7 | 60,0 / 78,0 |
| `RG_yolo11s_1024` | alle Objekte | 7,3 / 25,2 | 17,3 / 37,7 | 33,3 / 56,8 | 38,2 / 57,6 | 44,9 / 63,7 | 59,4 / 77,3 |
| `B-F2_yolo11s_1024` | primär | — / — | 7,4 / 19,8 | 18,6 / 36,3 | 28,1 / 46,1 | 37,9 / 57,7 | 56,6 / 75,4 |
| `B-F2_yolo11s_1024` | alle Objekte | 1,1 / 6,9 | 7,3 / 19,9 | 17,2 / 34,4 | 27,5 / 46,0 | 37,8 / 57,4 | 56,1 / 75,0 |
| `B-RG_yolo11s_1024` | primär | — / — | 7,3 / 19,6 | 18,6 / 36,6 | 28,7 / 47,5 | 38,9 / 58,9 | 58,5 / 76,7 |
| `B-RG_yolo11s_1024` | alle Objekte | 1,1 / 6,9 | 7,2 / 20,0 | 17,2 / 34,7 | 28,0 / 47,3 | 38,8 / 58,8 | 58,2 / 76,5 |
| `A_yolo11s_2048` | primär | — / — | 20,4 / 37,0 | 28,7 / 50,9 | 37,7 / 56,6 | 46,0 / 64,0 | 64,2 / 81,3 |
| `A_yolo11s_2048` | alle Objekte | 2,9 / 12,0 | 17,3 / 32,1 | 27,3 / 49,2 | 37,4 / 56,9 | 45,8 / 63,8 | 64,2 / 81,1 |
| `NMS-o2m_yolo11s_2048` | primär | — / — | 23,2 / 42,5 | 34,4 / 59,2 | 39,6 / 58,0 | 46,4 / 63,8 | 62,8 / 80,7 |
| `NMS-o2m_yolo11s_2048` | alle Objekte | 6,6 / 21,2 | 20,9 / 40,1 | 33,5 / 58,0 | 39,1 / 58,4 | 46,5 / 64,1 | 62,5 / 80,5 |
| `F2_yolo11s_2048` | primär | — / — | 26,1 / 46,6 | 36,6 / 61,0 | 41,5 / 60,3 | 48,3 / 65,6 | 63,6 / 82,1 |
| `F2_yolo11s_2048` | alle Objekte | 8,2 / 26,8 | 22,8 / 42,5 | 35,6 / 60,4 | 41,0 / 60,4 | 48,5 / 66,1 | 63,2 / 81,6 |
| `RG_yolo11s_2048` | primär | — / — | 26,7 / 46,7 | 36,7 / 61,1 | 42,2 / 60,4 | 49,6 / 66,6 | 66,9 / 84,2 |
| `RG_yolo11s_2048` | alle Objekte | 8,1 / 26,1 | 23,3 / 42,5 | 35,7 / 60,4 | 41,6 / 60,7 | 49,9 / 67,1 | 66,7 / 84,0 |
| `B-F2_yolo11s_2048` | primär | — / — | 20,0 / 37,4 | 28,5 / 50,7 | 37,2 / 56,3 | 44,8 / 62,9 | 60,6 / 79,3 |
| `B-F2_yolo11s_2048` | alle Objekte | 3,3 / 12,8 | 16,9 / 32,6 | 27,1 / 49,4 | 36,7 / 56,4 | 44,7 / 63,0 | 60,2 / 78,9 |
| `B-RG_yolo11s_2048` | primär | — / — | 19,8 / 37,3 | 28,6 / 50,8 | 38,1 / 56,8 | 46,1 / 63,8 | 64,2 / 81,6 |
| `B-RG_yolo11s_2048` | alle Objekte | 2,9 / 12,9 | 16,8 / 32,4 | 27,3 / 49,4 | 37,6 / 56,8 | 46,2 / 63,9 | 64,1 / 81,4 |
| `A_yolo12s_1024` | primär | — / — | 8,8 / 20,4 | 19,3 / 36,4 | 29,5 / 48,7 | 38,0 / 57,8 | 59,7 / 77,5 |
| `A_yolo12s_1024` | alle Objekte | 1,2 / 6,9 | 8,4 / 19,4 | 17,9 / 34,3 | 28,9 / 48,3 | 38,3 / 58,3 | 59,4 / 76,9 |
| `NMS-o2m_yolo12s_1024` | primär | — / — | 15,8 / 32,3 | 31,9 / 54,6 | 36,8 / 56,9 | 42,0 / 60,5 | 56,7 / 73,9 |
| `NMS-o2m_yolo12s_1024` | alle Objekte | 8,3 / 26,1 | 15,8 / 34,5 | 31,3 / 53,6 | 36,1 / 56,3 | 41,9 / 60,4 | 56,2 / 73,1 |
| `F2_yolo12s_1024` | primär | — / — | 18,2 / 36,2 | 34,2 / 57,7 | 38,7 / 58,0 | 43,8 / 62,5 | 58,9 / 76,4 |
| `F2_yolo12s_1024` | alle Objekte | 7,9 / 25,6 | 18,4 / 38,4 | 33,2 / 56,4 | 38,0 / 57,9 | 43,8 / 62,8 | 58,2 / 75,8 |
| `RG_yolo12s_1024` | primär | — / — | 18,0 / 35,8 | 34,4 / 57,9 | 39,4 / 58,6 | 44,6 / 62,9 | 61,0 / 78,6 |
| `RG_yolo12s_1024` | alle Objekte | 7,7 / 25,5 | 18,5 / 38,5 | 33,5 / 56,7 | 38,8 / 58,4 | 44,7 / 63,3 | 60,3 / 77,6 |
| `B-F2_yolo12s_1024` | primär | — / — | 9,1 / 20,6 | 19,4 / 37,0 | 28,9 / 48,4 | 36,9 / 56,9 | 57,1 / 74,2 |
| `B-F2_yolo12s_1024` | alle Objekte | 1,1 / 6,4 | 8,6 / 19,6 | 18,0 / 35,1 | 28,4 / 48,1 | 37,2 / 57,6 | 56,6 / 73,7 |
| `B-RG_yolo12s_1024` | primär | — / — | 9,1 / 20,5 | 19,7 / 37,4 | 29,7 / 49,2 | 38,2 / 57,8 | 59,8 / 77,6 |
| `B-RG_yolo12s_1024` | alle Objekte | 1,1 / 6,3 | 8,7 / 19,6 | 18,3 / 35,4 | 29,1 / 48,9 | 38,6 / 58,5 | 59,5 / 77,0 |
| `X_yolo11s_2048` | primär | — / — | 20,4 / 37,0 | 28,7 / 50,9 | 37,7 / 56,6 | 46,0 / 64,0 | 64,2 / 81,3 |
| `X_yolo11s_2048` | alle Objekte | 2,9 / 12,0 | 17,3 / 32,1 | 27,3 / 49,2 | 37,4 / 56,9 | 45,8 / 63,8 | 64,2 / 81,1 |

## Latenz der D-H3-Zeilen (host-b)

Siehe `reports/tab_latency_breakdown.md` (Nachtrag mit Ankerzeilen, gleiches Verfahren, 575 W, exklusiv).

## Trainingsbedingungen

| Modell | s | Batch | gewählte Epoche (calib-mAP50) | Mittel ± SD Epochen 41–60 |
|---|---|---|---|---|
| YOLOv11s | 1024 | 8 | 43 (65,24) | 64,20 ± 0,70 |
| YOLOv11s | 2048 | 3 | 47 (72,83) | 71,65 ± 1,18 |
| YOLOv12s | 1024 | 8 | 44 (65,02) | 63,61 ± 0,69 |

Rezept unverändert (60 Epochen, COCO-Init, Auswahl auf calib). YOLOv12s bei 2048: Speicher-Smoke mit Batch 3 und 2 fehlgeschlagen (`runs/dota/smoke.json`) → S_max = 1024 nach der präregistrierten Regel.
