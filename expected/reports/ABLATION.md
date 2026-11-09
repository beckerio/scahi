# SCAHI-Bausteine: Ablation (explorativ, post hoc)

Nutzerauftrag 2026-10-30, nach den Hauptergebnissen; **nie Grundlage eines Urteils**. Nur CPU, aus den vorhandenen Caches (`scripts/ablation_components.py`). In allen Armen dieselben Pässe: Einzelpass bei der Trainingsgröße 2048 + Kacheln, Zweig o2m.

| Arm | Variante | Bedeutung |
|---|---|---|
| (a) | F2n | Rand-Taper, ohne Abdeckungsnormalisierung |
| (b) | F1 | Abdeckungsnormalisierung, ohne Rand-Taper |
| (c) | F2 | Rand-Taper + Abdeckungsnormalisierung (F2, vorab festgelegt) |
| (d) | F3 | (c) + Validity-Fenster |
| (e) | F2 | (c) mit verfeinertem Peak-Raster (RG) |

- (a) behält die Rand-Taper-Gewichte und lässt nur die Division durch die Abdeckung C weg (Summe der gewichteten Scores über die Pässe), sodass (a) − (c) allein die Normalisierung misst.
- Parameter von (a), (b), (d): auf calib aus dem vorab festgelegten F2-Raster (24 Konfigurationen) gewählt, Kriterium die Primärmetrik des Datensatzes, genau wie die bestehenden F2-Zeilen. (c) und (e): eingefrorene calib-Auswahl. Keine neue Tuning-Dimension.
- cell_factor-Sweep: (c) mit cell_factor 5 / 7 / 9 / 11 / 13, übrige Parameter wie (c), ohne Auswahl.
- Gepaarter Bootstrap über Bilder (1000 Resamples, Seed 0, dieselben Resamples wie die Hauptauswertung), Perzentil-KI, Differenz gegen (c). Dup/Frag: Detektionen mit Score ≥ 0,25, Definitionen von `scahi.eval.diagnostics` (DOTA: alle GT einschließlich difficult).

## Einordnung: Abdeckungsnormalisierung

- In den eingefrorenen Stacks bilden die Rand-Taper über überlappende Kacheln eine Zerlegung der Eins, und beide Ebenen (Einzelpass auf das ganze Bild, Kacheln) decken das ganze Bild ab. Damit ist C konstant (C = 2), und die explizite Normalisierung skaliert die Scores nur um einen festen Faktor; sie ändert weder die Rangfolge noch die Boxen. Messbar bleibt allein, dass ohne Division mehr Peaks über `tau_peak` liegen (alle mit niedrigem Score), daher (a) − (c) ≈ 0.
- „Ein Pass ohne Vorhersage zählt als Evidenz dagegen“ folgt aus der Summe über die Pässe (ein Pass, der die Stelle abdeckt, aber nichts vorhersagt, trägt 0 bei), nicht aus der Division.
- (b) zeigt den Effekt der abdeckungsbewussten Gewichtung selbst: ohne Taper variiert C (Anzahl der abdeckenden Pässe), und die Gewichte bilden keine Zerlegung der Eins mehr.
- Eine explizite Normalisierung wäre bei Stacks mit nur teilweiser Abdeckung wirksam (Ebenen, die nur einen Bildausschnitt abdecken); solche Stacks enthalten die eingefrorenen Zeilen nicht. Dafür gibt es kein neues Experiment.

C an allen fusionierten Positionen (Zellen des Volumens; `scripts/ablation_coverage.py`, `runs/ablation/coverage.json`; val):

| Datensatz / Gewichte / Arm | Variante | Positionen im Bild | C min / max / Mittel | Anteil C = 2 | häufigste C-Werte (Anteil) | Positionen außerhalb des Bildes |
|---|---|---|---|---|---|---|
| dota / YOLOv10s / (b) | F1 | 5224913 | 2,00 / 10,00 / 3,409 | 24.24 % | 3,00 (45,6 %), 5,00 (24,6 %), 2,00 (24,2 %), 7,00 (3,1 %) | 1859 |
| dota / YOLOv10s / (c) | F2 | 5520990 | 2,00 / 2,00 / 2,000 | 100.00 % | 2,00 (100,0 %) | 1907 |
| dota / YOLOv10s / (e) | F2 | 13414784 | 2,00 / 2,00 / 2,000 | 100.00 % | 2,00 (100,0 %) | 7671 |
| dota / YOLOv10m / (b) | F1 | 5646899 | 2,00 / 10,00 / 3,421 | 23.72 % | 3,00 (45,7 %), 5,00 (25,0 %), 2,00 (23,7 %), 7,00 (3,1 %) | 4835 |
| dota / YOLOv10m / (c) | F2 | 5930876 | 2,00 / 2,00 / 2,000 | 100.00 % | 2,00 (100,0 %) | 4875 |
| dota / YOLOv10m / (e) | F2 | 12239960 | 2,00 / 2,00 / 2,000 | 100.00 % | 2,00 (100,0 %) | 22243 |
| visdrone / YOLOv10s / (c) | F2 | 3326808 | 2,00 / 2,00 / 2,000 | 100.00 % | 2,00 (100,0 %) | 1121 |
| visdrone / YOLOv10s / (e) | F2 | 10523998 | 2,00 / 2,00 / 2,000 | 100.00 % | 2,00 (100,0 %) | 8 |
| visdrone / YOLOv10s / (b) | F1 | 3326808 | 2,00 / 7,00 / 3,546 | 21.56 % | 3,00 (44,4 %), 5,00 (26,4 %), 2,00 (21,6 %), 7,00 (5,2 %) | 1121 |

## VisDrone (val)

Primärmetrik: AP (Toolkit-Port, = AP@[.5:.95]). Abbildung `reports/figures/ablation/fig_ablation_visdrone.png`.

Kontrolle: der Sweep-Punkt cell_factor = 5 berechnet (c) neu; |ΔAP| gegen die bestehende Zeile: YOLOv10s 0.0e+00, YOLOv10m 0.0e+00, YOLOv11s 0.0e+00, YOLOv12s 0.0e+00, YOLOv11m 0.0e+00.

| Gewichte | Arm | Variante | AP | Δ AP gegen (c) [95 %-KI] | AP50 | Recall@0,5 crowded / isolated | Δ Recall crowded gegen (c) | Dup / Frag (%) |
|---|---|---|---|---|---|---|---|---|
| YOLOv10s | a | F2n | 39,34 | +0,00 [-0,00; +0,00] | 64,20 | 63,5 / 89,1 | +0,08 [+0,00; +0,22] | 0,8 / 0,6 |
| YOLOv10s | b | F1 | 38,56 | -0,78 [-0,90; -0,64] | 63,64 | 64,2 / 89,0 | +0,79 [-0,08; +1,65] | 1,5 / 2,1 |
| YOLOv10s | c | F2 | 39,34 | (Referenz) | 64,19 | 63,4 / 89,0 | — | 0,4 / 0,4 |
| YOLOv10s | d | F3 | 37,68 | -1,66 [-1,98; -1,31] | 62,11 | 62,1 / 87,9 | -1,30 [-2,56; -0,12] | 0,4 / 0,4 |
| YOLOv10s | e | F2 | 40,72 | +1,37 [+1,27; +1,50] | 66,15 | 81,1 / 91,9 | +17,64 [+16,08; +19,20] | 1,0 / 0,4 |
| YOLOv10s | cf5 | F2 cf 5 | 39,34 | +0,00 [+0,00; +0,00] | 64,19 | 63,4 / 89,0 | +0,00 [+0,00; +0,00] | 0,4 / 0,4 |
| YOLOv10s | cf7 | F2 cf 7 | 40,23 | +0,89 [+0,79; +0,98] | 65,67 | 71,9 / 90,8 | +8,53 [+7,30; +9,76] | 0,4 / 0,4 |
| YOLOv10s | cf9 | F2 cf 9 | 40,50 | +1,16 [+1,06; +1,27] | 66,07 | 76,8 / 91,5 | +13,34 [+12,02; +14,72] | 0,5 / 0,4 |
| YOLOv10s | cf11 | F2 cf 11 | 40,64 | +1,30 [+1,19; +1,42] | 66,12 | 79,2 / 91,7 | +15,76 [+14,46; +17,14] | 0,8 / 0,4 |
| YOLOv10s | cf13 | F2 cf 13 | 40,64 | +1,30 [+1,19; +1,42] | 66,05 | 81,5 / 91,8 | +18,06 [+16,60; +19,50] | 1,0 / 0,4 |
| YOLOv10m | a | F2n | 41,29 | +0,00 [+0,00; +0,00] | 66,94 | 66,3 / 90,4 | +0,04 [+0,00; +0,14] | 0,6 / 0,6 |
| YOLOv10m | b | F1 | 40,44 | -0,85 [-0,99; -0,72] | 66,28 | 66,9 / 90,3 | +0,59 [-0,20; +1,34] | 1,6 / 2,2 |
| YOLOv10m | c | F2 | 41,29 | (Referenz) | 66,93 | 66,3 / 90,4 | — | 0,4 / 0,4 |
| YOLOv10m | d | F3 | 39,50 | -1,79 [-2,12; -1,44] | 64,64 | 64,1 / 89,3 | -2,22 [-3,56; -0,88] | 0,3 / 0,4 |
| YOLOv10m | e | F2 | 42,67 | +1,38 [+1,23; +1,51] | 68,69 | 84,9 / 92,9 | +18,65 [+17,21; +20,14] | 1,1 / 0,4 |
| YOLOv10m | cf5 | F2 cf 5 | 41,29 | +0,00 [+0,00; +0,00] | 66,93 | 66,3 / 90,4 | +0,00 [+0,00; +0,00] | 0,4 / 0,4 |
| YOLOv10m | cf7 | F2 cf 7 | 42,25 | +0,96 [+0,85; +1,07] | 68,41 | 76,2 / 91,9 | +9,87 [+8,63; +11,16] | 0,4 / 0,4 |
| YOLOv10m | cf9 | F2 cf 9 | 42,44 | +1,15 [+1,03; +1,27] | 68,62 | 79,6 / 92,3 | +13,29 [+11,94; +14,74] | 0,4 / 0,4 |
| YOLOv10m | cf11 | F2 cf 11 | 42,58 | +1,28 [+1,15; +1,40] | 68,75 | 81,7 / 92,6 | +15,38 [+14,03; +16,81] | 0,5 / 0,4 |
| YOLOv10m | cf13 | F2 cf 13 | 42,64 | +1,35 [+1,21; +1,47] | 68,74 | 83,5 / 92,7 | +17,18 [+15,75; +18,71] | 0,9 / 0,4 |
| YOLOv11s | a | F2n | 38,66 | +0,00 [-0,00; +0,00] | 63,58 | 62,1 / 88,9 | +0,00 [+0,00; +0,00] | 0,8 / 0,7 |
| YOLOv11s | b | F1 | 37,92 | -0,74 [-0,87; -0,60] | 62,97 | 63,1 / 88,7 | +1,00 [+0,30; +1,71] | 1,6 / 2,3 |
| YOLOv11s | c | F2 | 38,66 | (Referenz) | 63,58 | 62,1 / 88,9 | — | 0,5 / 0,4 |
| YOLOv11s | d | F3 | 36,91 | -1,75 [-2,07; -1,42] | 61,23 | 60,5 / 87,5 | -1,63 [-3,02; -0,35] | 0,4 / 0,4 |
| YOLOv11s | e | F2 | 40,03 | +1,37 [+1,25; +1,52] | 65,30 | 81,6 / 91,6 | +19,48 [+17,77; +21,28] | 1,3 / 0,5 |
| YOLOv11s | cf5 | F2 cf 5 | 38,66 | +0,00 [+0,00; +0,00] | 63,58 | 62,1 / 88,9 | +0,00 [+0,00; +0,00] | 0,5 / 0,4 |
| YOLOv11s | cf7 | F2 cf 7 | 39,58 | +0,92 [+0,83; +1,03] | 65,01 | 71,7 / 90,5 | +9,62 [+8,28; +10,91] | 0,4 / 0,4 |
| YOLOv11s | cf9 | F2 cf 9 | 39,85 | +1,19 [+1,10; +1,31] | 65,34 | 77,0 / 91,0 | +14,92 [+13,52; +16,39] | 0,6 / 0,4 |
| YOLOv11s | cf11 | F2 cf 11 | 39,97 | +1,32 [+1,21; +1,45] | 65,42 | 79,4 / 91,3 | +17,31 [+15,66; +18,98] | 0,8 / 0,4 |
| YOLOv11s | cf13 | F2 cf 13 | 40,03 | +1,37 [+1,25; +1,50] | 65,36 | 81,4 / 91,4 | +19,23 [+17,56; +20,93] | 1,2 / 0,5 |
| YOLOv12s | a | F2n | 39,80 | +0,00 [+0,00; +0,00] | 64,77 | 63,2 / 89,5 | +0,00 [+0,00; +0,00] | 0,8 / 0,5 |
| YOLOv12s | b | F1 | 38,98 | -0,82 [-0,94; -0,70] | 64,24 | 64,0 / 89,5 | +0,79 [+0,04; +1,53] | 1,6 / 2,1 |
| YOLOv12s | c | F2 | 39,80 | (Referenz) | 64,77 | 63,2 / 89,5 | — | 0,5 / 0,3 |
| YOLOv12s | d | F3 | 38,14 | -1,66 [-1,97; -1,33] | 62,60 | 61,8 / 88,2 | -1,42 [-2,59; -0,31] | 0,4 / 0,3 |
| YOLOv12s | e | F2 | 41,18 | +1,38 [+1,25; +1,51] | 66,46 | 85,2 / 92,2 | +21,95 [+20,45; +23,52] | 1,4 / 0,4 |
| YOLOv12s | cf5 | F2 cf 5 | 39,80 | +0,00 [+0,00; +0,00] | 64,77 | 63,2 / 89,5 | +0,00 [+0,00; +0,00] | 0,5 / 0,3 |
| YOLOv12s | cf7 | F2 cf 7 | 40,71 | +0,91 [+0,80; +1,01] | 66,22 | 72,9 / 91,1 | +9,70 [+8,40; +10,95] | 0,4 / 0,4 |
| YOLOv12s | cf9 | F2 cf 9 | 40,95 | +1,15 [+1,04; +1,27] | 66,57 | 77,5 / 91,7 | +14,30 [+12,83; +15,72] | 0,5 / 0,4 |
| YOLOv12s | cf11 | F2 cf 11 | 41,11 | +1,31 [+1,20; +1,43] | 66,64 | 80,4 / 92,0 | +17,14 [+15,61; +18,75] | 0,7 / 0,4 |
| YOLOv12s | cf13 | F2 cf 13 | 41,19 | +1,39 [+1,26; +1,52] | 66,66 | 82,6 / 92,1 | +19,36 [+17,77; +20,99] | 1,0 / 0,4 |
| YOLOv11m | a | F2n | 41,54 | +0,00 [-0,00; +0,00] | 66,98 | 66,0 / 90,5 | +0,00 [+0,00; +0,00] | 0,6 / 0,5 |
| YOLOv11m | b | F1 | 40,52 | -1,01 [-1,14; -0,87] | 66,18 | 65,8 / 90,3 | -0,21 [-0,90; +0,48] | 1,5 / 2,2 |
| YOLOv11m | c | F2 | 41,54 | (Referenz) | 66,98 | 66,0 / 90,5 | — | 0,4 / 0,4 |
| YOLOv11m | d | F3 | 39,93 | -1,61 [-1,96; -1,26] | 64,89 | 64,8 / 89,1 | -1,25 [-2,45; +0,00] | 0,4 / 0,3 |
| YOLOv11m | e | F2 | 42,86 | +1,32 [+1,20; +1,45] | 68,73 | 85,0 / 93,0 | +19,02 [+17,53; +20,62] | 0,9 / 0,4 |
| YOLOv11m | cf5 | F2 cf 5 | 41,54 | +0,00 [+0,00; +0,00] | 66,98 | 66,0 / 90,5 | +0,00 [+0,00; +0,00] | 0,4 / 0,4 |
| YOLOv11m | cf7 | F2 cf 7 | 42,41 | +0,88 [+0,78; +0,99] | 68,34 | 76,0 / 92,0 | +9,99 [+8,79; +11,32] | 0,4 / 0,4 |
| YOLOv11m | cf9 | F2 cf 9 | 42,66 | +1,12 [+1,02; +1,22] | 68,74 | 80,3 / 92,6 | +14,26 [+12,80; +15,76] | 0,5 / 0,4 |
| YOLOv11m | cf11 | F2 cf 11 | 42,76 | +1,22 [+1,11; +1,34] | 68,80 | 82,7 / 92,8 | +16,64 [+15,18; +18,15] | 0,6 / 0,4 |
| YOLOv11m | cf13 | F2 cf 13 | 42,80 | +1,27 [+1,15; +1,38] | 68,73 | 84,0 / 92,8 | +17,98 [+16,43; +19,63] | 0,8 / 0,4 |

## DOTA v1.5 HBB (DOTA-val, post hoc)

Primärmetrik: mAP50 (Devkit, ohne Cap). Abbildung `reports/figures/ablation/fig_ablation_dota.png`.

**Post-hoc-Analyse auf dem bereits einmal ausgewerteten zurückgehaltenen Satz DOTA-val.** Arm (e) ist hier SCAHI mit dem verfeinerten Raster, **auf DOTA vorab festgelegt** (auf VisDrone war das Raster post hoc). Die Validity-Fenster von (d) sind die vorab festgelegten VisDrone-Parameter (Eingabe-σ 6–256 px); große DOTA-Objekte liegen teils außerhalb, die Scores von (d) überschreiten dort 1 (Ränge und AP davon nicht verfälscht). Die Fenster werden nicht nachträglich an DOTA angepasst. Da die Validity-Fenster nicht Teil von SCAHI sind (schon auf VisDrone ohne Nutzen) und auf DOTA zusätzlich falsch skaliert, steht (d) hier **nur der Vollständigkeit halber, nicht als Argument**.

| Gewichte | Arm | Variante | mAP50 | Δ mAP50 gegen (c) [95 %-KI] | AP | Recall@0,5 crowded / isolated | Δ Recall crowded gegen (c) | Dup / Frag (%) |
|---|---|---|---|---|---|---|---|---|
| YOLOv10s | a | F2n | 70,64 | -0,00 [-0,00; +0,04] | 49,15 | 66,8 / 96,9 | +0,00 [+0,00; +0,00] | 0,6 / 2,4 |
| YOLOv10s | b | F1 | 69,52 | -1,12 [-1,87; -0,52] | 47,35 | 64,5 / 96,6 | -2,39 [-3,91; -0,84] | 0,3 / 2,0 |
| YOLOv10s | c | F2 | 70,64 | (Referenz) | 49,15 | 66,8 / 96,9 | — | 0,3 / 1,1 |
| YOLOv10s | d | F3 | 51,96 | -18,68 [-21,64; -15,72] | 35,08 | 65,6 / 95,9 | -1,27 [-2,13; -0,68] | 0,2 / 2,2 |
| YOLOv10s | e | F2 | 71,00 | +0,36 [-0,22; +0,92] | 50,13 | 91,1 / 98,1 | +24,29 [+19,76; +27,47] | 1,4 / 1,2 |
| YOLOv10s | cf5 | F2 cf 5 | 70,64 | +0,00 [+0,00; +0,00] | 49,15 | 66,8 / 96,9 | +0,00 [+0,00; +0,00] | 0,3 / 1,1 |
| YOLOv10s | cf7 | F2 cf 7 | 71,18 | +0,54 [+0,22; +1,06] | 49,90 | 88,9 / 97,6 | +22,11 [+17,64; +25,69] | 0,5 / 1,1 |
| YOLOv10s | cf9 | F2 cf 9 | 71,36 | +0,72 [+0,19; +1,19] | 50,11 | 94,9 / 97,7 | +28,07 [+23,11; +31,58] | 0,9 / 1,1 |
| YOLOv10s | cf11 | F2 cf 11 | 71,16 | +0,52 [-0,04; +1,11] | 50,18 | 96,3 / 97,9 | +29,42 [+24,72; +32,77] | 1,5 / 1,2 |
| YOLOv10s | cf13 | F2 cf 13 | 71,24 | +0,60 [+0,06; +1,13] | 50,26 | 96,6 / 97,9 | +29,74 [+25,17; +32,99] | 2,0 / 1,2 |
| YOLOv10m | a | F2n | 71,32 | -0,00 [-0,00; +0,04] | 50,57 | 68,0 / 97,4 | +0,00 [+0,00; +0,00] | 1,0 / 3,5 |
| YOLOv10m | b | F1 | 70,33 | -0,99 [-1,54; -0,31] | 48,99 | 68,2 / 97,2 | +0,28 [-1,38; +2,11] | 0,4 / 2,9 |
| YOLOv10m | c | F2 | 71,32 | (Referenz) | 50,57 | 68,0 / 97,3 | — | 0,4 / 1,6 |
| YOLOv10m | d | F3 | 50,89 | -20,43 [-23,77; -17,04] | 34,58 | 67,2 / 96,1 | -0,72 [-2,33; +0,70] | 0,2 / 3,3 |
| YOLOv10m | e | F2 | 71,73 | +0,41 [+0,14; +0,93] | 51,30 | 91,6 / 98,4 | +23,66 [+19,98; +26,20] | 0,8 / 1,6 |
| YOLOv10m | cf5 | F2 cf 5 | 71,32 | +0,00 [+0,00; +0,00] | 50,57 | 68,0 / 97,3 | +0,00 [+0,00; +0,00] | 0,4 / 1,6 |
| YOLOv10m | cf7 | F2 cf 7 | 71,83 | +0,50 [+0,22; +0,93] | 51,18 | 90,9 / 98,0 | +22,94 [+18,93; +25,69] | 0,6 / 1,6 |
| YOLOv10m | cf9 | F2 cf 9 | 72,09 | +0,76 [+0,32; +1,15] | 51,43 | 95,4 / 98,1 | +27,44 [+23,16; +30,46] | 1,0 / 1,6 |
| YOLOv10m | cf11 | F2 cf 11 | 71,73 | +0,40 [-0,03; +0,83] | 51,38 | 96,3 / 98,1 | +28,39 [+24,24; +31,34] | 1,4 / 1,6 |
| YOLOv10m | cf13 | F2 cf 13 | 71,65 | +0,33 [-0,09; +0,86] | 51,49 | 96,7 / 98,2 | +28,79 [+24,71; +31,70] | 1,9 / 1,6 |
