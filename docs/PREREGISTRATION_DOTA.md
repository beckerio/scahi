# Präregistrierung Phase D – DOTA v1.5 (horizontale Boxen)

**Status: ENTWURF zur Freigabe (Stopp b, `docs/DOTA_BRIEF.md` §4).** Committet vor jedem Trainingsergebnis. Nach der Freigabe
ändern sich Hypothesen, Zeilen, Metriken und Regeln nur noch über datierte Addenda, und zwar vor dem jeweiligen Ergebnis.
Grundlage: `docs/DOTA_BRIEF.md` (bei Konflikt maßgeblich), Audit `reports/D_audit.md`, Auswertungs-Port
`scahi/dota/evaluate.py`, Split `data/splits/dota/`. VisDrone bleibt unberührt.

## 1. Daten, Split, Auswertung

- Labels: offizielle DOTA-v1.5-HBB-Dateien; Bilder: Ultralytics `DOTAv1.5.zip` (JPEG q70, neu kodiert; Einschränkung).
- Split: DOTA-train → `train_main` (1269 Bilder) + `calib` (141; stratifiziert nach Objekten je Bild × medianer Objektgröße,
  Seed 20261030); P1872 ausgeschlossen (Annotation passt nicht zum Bild). **DOTA-val (458 Bilder) = zurückgehaltener
  Testsatz**, genau einmal nach dem Freeze ausgewertet. DOTA-test und der Server werden nicht benutzt.
- Training auf allen GT einschließlich „difficult“ (Boxen auf das Bild begrenzt). Die Auswertung ignoriert difficult-GT wie
  das Devkit.
- **Primärmetrik:** offizielle All-Class-mAP bei IoU 0,5 (DOTA_devkit Task 2, VOC07-11-Punkt, 16 Klassen; Port exakt gegen
  das Original validiert, `runs/dota/eval_validation.json`). Eine Klasse ohne nicht-difficult-GT in der ausgewerteten
  Bildmenge (auch in einem Bootstrap-Resample) fällt aus dem Mittel.
- **Detektions-Cap 4000 je Bild** (2 × p99 der GT je Trainingsbild, auf 500 aufgerundet) für alle Methoden, auch beim
  Tuning und bei der Modellauswahl; zusätzlich das offizielle Ergebnis ohne Cap (sekundär).
- Sekundär (gleiches Matching): AP@[.5:.95] (COCO-101-Punkt); Fahrzeug-Teilmenge (small + large vehicle) für beide
  Metriken; klassenagnostische AP50; Größenbins (AP, AP50; GT außerhalb des Bins ignoriert, ungematchte Detektionen außerhalb
  ignoriert). Hinweis aus dem Audit: alle GT < 8 px und rund 75 % der GT 8–16 px sind difficult; der Bin < 8 ist daher leer,
  8–16 px dünn besetzt.

## 2. Modelle und Rezept

- YOLOv10s und YOLOv10m (Familie vorab fest), COCO-Init, **60 Epochen**, Ultralytics 8.4.183, Augmentierung wie VisDrone
  (Ultralytics-Standard), AMP, Seed 0.
- Einzelpass-Leiter: Training und Inferenz bei langer Seite s ∈ {1024, 2048, 3072} (ganzes Bild letterboxed auf s).
- Batch: 8 bei 1024, 3 bei 2048 und 3072 (wie VisDrone 1280 / 2048). Eine Reduktion nach OOM bei der festen Batch
  (8 → 6, 3 → 2), Batch ≥ 2, keine stille Batch-Änderung (Lauf bricht ab, wenn Ultralytics die Batch selbst halbiert).
- **S_max** je Modell = größtes s, dessen Training nach dieser Regel passt. 3072 wurde vorab mit dem Speicher-Smoke auf
  den 48 dichtesten Trainingsbildern geprüft (`runs/dota/smoke.json`): passt nicht → **S_max = 2048** (Abschnitt 6).
- Modellauswahl auf calib: offizielle mAP50 (Cap 4000) des **o2o-Kopfs**, ausgewertet in jeder 5. Epoche und in jeder der
  letzten 20; die o2m-Kurve (+ NMS) wird mitgeschrieben.

## 3. Inferenzzeilen (je Modell und s, auf den Gewichten von s)

Alle Parameter werden **nur auf calib** gewählt (Kriterium: calib-mAP50, Cap 4000). Pässe werden einmal je Gewichtssatz
gecacht (calib und DOTA-val; o2o und o2m), Fusionen laufen aus den Caches.

| Zeile | Pässe | Fusion | Kopf | Tuning auf calib |
|---|---|---|---|---|
| A_s | ganzes Bild bei s | keine (NMS-frei) | o2o | conf ∈ {0,001; 0,01; 0,05; 0,1} |
| NMS_s | ganzes Bild bei s + Kacheln z = 1 | NMS mit IoS | o2o und o2m | IoS-Schwelle {0,3 … 0,8} × conf {0,01; 0,1} (12) |
| F2_s | dieselben Pässe | SCAHI, vorab festgelegt | o2m | VisDrone-F2-Raster (24 Konfigurationen) |
| RG_s | dieselben Pässe | SCAHI, verfeinertes Raster | o2m | 108 Varianten + Referenz, Zeitgrenze 2× |
| B-F2_s | nur ganzes Bild bei s | SCAHI, Parameter von F2_s | o2m | keines (Zerlegung) |
| B-RG_s | nur ganzes Bild bei s | SCAHI, verfeinertes Raster | o2m | eigene Auswahl wie RG_s |
| SAHI | 1024er Gewichte, Slices + ganzes Bild | SAHI-Merge | o2o | VisDrone-SAHI-Raster v2 |

- **Kacheln z = 1:** native Auflösung, Kacheln von 1024 × 1024 px mit Überlappung 0,3, Eingabe 1024 px (Bilder mit langer
  Seite ≤ 1024 px: eine Kachel = ganzes Bild nativ).
- **F2 (vorab festgelegt):** tau_splat ∈ {0,005; 0,02} × tau_peak ∈ {0,001; 0,01} × cell_factor ∈ {3; 5} × bins_per_oct
  ∈ {3; 4; 5} (= VisDrone), iou_cluster 0,5.
- **Verfeinertes Raster (auf DOTA vorab festgelegt, nicht post hoc):** das eingefrorene X-grid-Verfahren von VisDrone
  (`configs/ext/xgrid_frozen.json`), mit dem **vollständigen** Raster für die Zeilen B und C: tau_splat ∈ {0,005;
  0,00125; 0,0003125} × tau_peak ∈ {1e-3; 1e-4; 1e-5} × cell_factor ∈ {7; 9; 11; 13} × bins_per_oct ∈ {6; 7; 8}
  (108 Varianten) plus die Referenz = die calib-gewählte F2-Konfiguration desselben Gewichtssatzes. Auswahl: maximale
  calib-mAP50 unter den Varianten, deren Fusionszeit je Bild ≤ 2× die der Referenz ist (gemessen im selben Prozess wie auf
  VisDrone). Gleichstand: kleinere Fusionszeit.
- **Optional** (nur bei passendem Kostenrahmen, in Abschnitt 6 entschieden): dieselben Zeilen mit zusätzlichen Kacheln z = 2.

## 4. Hypothesen (auf DOTA-val, einmal ausgewertet)

ε = 0,30 mAP50-Punkte (aus VisDrone übernommen; ein Seed je Lauf; als Annahme ausgewiesen). Je Modell m: Δ mit gepaartem
Bootstrap über DOTA-val-Bilder (1000 Resamples, Seed 0, Perzentil-KI 2,5/97,5 %), einseitiger Bootstrap-p-Wert
p = (1 + #{b: Δ*_b ≤ ε}) / (B + 1), **Holm über die zwei Modelle** je Hypothese, α = 0,05.
**Ein Modell erfüllt eine Hypothese**, wenn sein Kriterium gilt **und** sein Holm-adjustiertes p ≤ 0,05.
**Eine Hypothese gilt**, wenn **beide** Modelle sie erfüllen.

- **D-H1 (Auflösung vs. Kachelung):** Δ = A_2048 − NMS_1024 (o2o); Kriterium Δ > ε.
- **D-H2 (Fusion):** Δ = RG_s − NMS_s (beide o2m, identische Pässe), **getrennt für s = 1024 (D-H2a) und s = 2048
  (D-H2b)**; Kriterium Δ > ε.
- **D-H3 (entscheidend):** Δ = RG_{S_max} − A_{S_max} (SCAHI o2m mit Kacheln gegen den Einzelpass o2o, beide auf den
  S_max-Gewichten); Kriterium Δ > ε **und** untere KI-Grenze > 0.
- **Nicht prüfbar** (eine nötige Zeile existiert nicht, z. B. Training passt nicht): zählt als **nicht erfüllt**
  (Holm-p = 1). S_max ist dabei immer prüfbar, weil es das größte passende s ist.

Sekundär, mit KI, ohne Urteil: dieselben Kontraste für F2 statt RG; D-H1 und D-H2 zusätzlich mit dem jeweils anderen Kopf;
Fahrzeug-Teilmenge; AP@[.5:.95]; offizielle mAP50 ohne Cap; Zerlegung A → B → C (B-F2, B-RG); crowded vs. isolated
Recall; GFLOPs (exakt je Eingabeform) und Latenz aller D-H3-Zeilen; SAHI gegen A_1024 und NMS_1024.

## 5. Ablauf, Gates, Freeze

- Training → Auswahl auf calib → Caches (calib, DOTA-val) → Tuning und Fusion nur auf calib. Paritäts-Gate wie VisDrone
  (Dekodierung ≤ 4 ulp, |ΔmAP50| ≤ 0,01 zwischen eigener und Ultralytics-Dekodierung auf calib).
- Während der Entwicklung nur calib-Zahlen; keine DOTA-val-Zahl vor dem Freeze.
- `FREEZE_PROPOSAL_DOTA`: alle Zeilen, Provenienz, SHA-256 der Gewichte, Replay jeder eingefrorenen Zeile über den
  Auswertungspfad auf calib (exakt). Berichtsregeln wie VisDrone test-dev (alle Zeilen, kein Nachtuning, Größenbins,
  exploratorische Zeilen getrennt). Stopp c, dann DOTA-val einmal.

## 6. Kosten und Zeitplan (Smoke-Tests 2026-10-30, `runs/dota/smoke.json`)

**Speicher-Smoke (48 dichteste train_main-Bilder, bis 10 206 Objekte je Bild):** 1024 / Batch 8 und 2048 / Batch 3 passen
für beide Modelle (Spitze reserviert 25,8–29,3 von 31,3 GB); **3072 passt weder mit Batch 3 noch mit Batch 2** (Ultralytics
halbiert auf 1). Nach Abschnitt 2 ist damit **S_max = 2048 für beide Modelle**; 3072 wird nicht trainiert. D-H3 vergleicht
also RG_2048 mit A_2048. Pipeline-Smoke (2 Epochen, 16 Bilder, calib-Auswahl über den Devkit-Port): für alle vier
Trainings ohne Fehler.

| Training | Batch | s/Iteration (zufällige Bilder) | Training | calib-Auswertungen (28 × 2 Köpfe) | Summe | konservativ (× 1,5) |
|---|---|---|---|---|---|---|
| YOLOv10s 1024 | 8 | 0,054 | 0,14 h | 0,09 h | 0,24 h | 0,4 h |
| YOLOv10s 2048 | 3 | 0,086 | 0,61 h | 0,10 h | 0,71 h | 1,1 h |
| YOLOv10m 1024 | 8 | 0,082 | 0,22 h | 0,10 h | 0,31 h | 0,5 h |
| YOLOv10m 2048 | 3 | 0,121 | 0,85 h | 0,33 h | 1,18 h | 1,8 h |

Weitere GPU-Arbeit (Schätzung): Paritäts-Gate und Einzelpass-Zeilen ≈ 0,5 h; Caches (calib + DOTA-val, ganzes Bild +
Kacheln z = 1, beide Köpfe, 4 Gewichtssätze) ≈ 1–2 h; SAHI-Tuning und -Zeile auf den 1024er Gewichten ≈ 1–2 h; Latenz der
D-H3-Zeilen ≈ 0,5 h. **Gesamt ≈ 7–9 GPU-h** (Budget 30). Die CPU-Fusion (F2-Raster, 2 × 108 Varianten des verfeinerten
Rasters je Gewichtssatz und Kopf, auf calib) ist der zeitkritische Teil; sie wird nach dem ersten Training auf echten
Caches gemessen und im Zeitplan nachgetragen.

**Zeitplan gegen die Migration (Sonntag 09:00):** nach Freigabe sofort die vier Trainings (YOLOv10s 1024, 2048, dann
YOLOv10m 1024, 2048; ≈ 4 h konservativ), danach Gates, Einzelpass-Zeilen und Caches (Freitagnacht bis Samstag),
Tuning auf calib (CPU, Samstag). Was bis Sonntag 08:30 nicht fertig ist, läuft auf host-a weiter (Caches und Gewichte im
Manifest, Replay-Gate zuerst). DOTA-val erst nach dem Freeze (Stopp c).

**Optionale z = 2-Kacheln: nicht vorgesehen** (Empfehlung). Begründung: Sie zielen auf Objekte < 16 px, die in DOTA v1.5
fast alle difficult sind und von der offiziellen Auswertung ignoriert werden; sie vervierfachen die Kachelzahl und damit
Cache und CPU-Fusion.

## 7. Annahmen und Einschränkungen

- ε aus VisDrone, ein Seed je Lauf.
- Bilder JPEG-neu-kodiert; HBB auf gedrehten, länglichen, dicht geparkten Objekten (OBB = future work).
- Die offizielle Auswertung ignoriert difficult-GT, darunter alle Objekte < 8 px: ein Vorteil bei sehr kleinen Objekten
  ist in der Primärmetrik nicht sichtbar.
- calib-AP von helicopter (4 Instanzen) und container-crane (11, ein Bild) ist verrauscht; beide gehen mit gleichem Gewicht
  in die calib-mAP50 ein (wie in der offiziellen Metrik).

## Addendum 1 – 2026-10-30 ~08:50 (Freigabe Stopp b mit Änderungen; vor dem ersten Training)

**Status: FREIGEGEBEN** mit den folgenden Änderungen; wo sie vom Text oben abweichen, gilt dieses Addendum.

1. **Metriken.**
   - **Primär:** offizielle Devkit-mAP50 **ohne Detektions-Cap** (das Devkit hat keinen). Auch die Modellauswahl auf calib
     und jedes calib-Tuning nutzen diese Metrik.
   - **Sekundär:** dieselbe Metrik mit Cap 4000 je Bild (Regel 2 × p99), zur Vergleichbarkeit mit dem VisDrone-Verfahren.
   - Bilder mit mehr als 4000 GT-Objekten (alle GT inkl. difficult; aus den Annotationen, keine Ergebnisse):
     **calib 0 von 141; DOTA-val 1 von 458 mit 12,4 % aller DOTA-val-GT** (dichtestes Bild 8609 GT); train_main 2 von 1269
     (8,0 %). Begründung: 4000 liegt unter der Objektzahl der dichtesten Bilder; ein Cap schneidet dort korrekte
     Detektionen ab und benachteiligt Methoden, die mehr Objekte finden.
   - **Pipeline-Grenze:** Damit die Primärmetrik nicht durch interne Grenzen gekappt wird, liegen alle internen Grenzen
     (Prädiktor `max_det`, Detektionen je Pass für die Box-Fusion, `top_k` von SCAHI) bei **20 000 je Bild bzw. Pass**
     (> 2 × das dichteste Bild). Jede Zeile berichtet, wie viele Bilder diese Grenze erreichen.
2. **„Difficult“.**
   - Training: difficult-GT sind normale Positive (bestätigt: alle GT von train_main stehen in den Trainingslabels, außer
     zwei Boxen von P0334, die vollständig außerhalb des Bildes liegen).
   - **Vorab festgelegte Sekundäranalyse „alle Objekte“:** derselbe Port mit difficult-GT als normale GT
     (`scahi.dota.evaluate.gt_arrays(..., all_objects=True)`). Berichtet wird D-H3 darauf mit gepaarten Bootstrap-KIs
     (sekundär, ohne Urteil) sowie die Größenbins < 8 px und 8–16 px (AP und AP50) aller D-H3-Zeilen.
3. **Größen:** S_max = 2048 für beide Modelle nach der Regel (3072 passt wegen der Objektdichte nicht mit Batch ≥ 2). Kein
   Training bei 3072, kein Cropping, keine Objekt-Unterabtastung, keine anderen Rezeptänderungen. Auf DOTA ist das eine
   echte Speichergrenze des Trainings, anders als bei VisDrone (dort lag die Grenze an der festen Rezept-Batch).
   Hypothesen, die eine nicht passende Größe bräuchten, gälten als nicht erfüllt (nicht prüfbar); D-H1–D-H3 sind mit
   1024 und 2048 alle prüfbar.
4. **Inferenz-Kreuzung (jetzt vorab festgelegt, sekundär, ohne Urteil):** Einzelpass (o2o) mit den 2048er Gewichten bei
   Inferenzgröße 2048, 2560 und 3072; die beste Größe wird auf calib gewählt (primäre Metrik) und auf DOTA-val neben D-H3
   berichtet: Δ = RG_2048 − (bester Einzelpass der Kreuzung), mit gepaartem Bootstrap-KI. conf wie bei A_s auf calib.
5. **Optionale z = 2-Kacheln: gestrichen.**
6. **P1872:** Ausschluss aus train_main und calib bestätigt.
7. **Holm-Regel wie im Entwurf:** „Modell erfüllt“ = Kriterium und Holm-adjustiertes p ≤ 0,05; die Hypothese gilt, wenn
   beide Modelle sie erfüllen.
8. **Zeitplan wie vorgeschlagen;** die CPU-Zeit der Fusion mit dem verfeinerten Raster wird auf den ersten echten Caches
   gemessen und in die Prognose aufgenommen. Was bis Sonntag 08:30 nicht fertig ist, läuft auf host-a weiter.

## Addendum 2 – 2026-10-31 ~16:00 (Nutzerfreigabe: YOLO11s und YOLO12s; vor jedem Training dieser Modelle)

**Status: vom Nutzer freigegeben (2026-10-31), committet vor dem ersten Training und vor jedem Ergebnis der neuen Modelle.**
Alles oben (inkl. Addendum 1) gilt unverändert, soweit hier nichts anderes steht. Die Ergebnisse und Urteile der zwei
ursprünglichen Modelle (`runs/dota/val/results.json`) bleiben, wie sie sind.

1. **Modelle:** zusätzlich **YOLO11s** und **YOLO12s** (Ultralytics 8.4.183, COCO-Init), Seed 0.
2. **Rezept identisch:** 60 Epochen; Batch 8 bei 1024 und 3 bei 2048; eine Reduktion nach OOM (8 → 6, 3 → 2), Batch ≥ 2,
   keine stille Batch-Änderung. Speicher-Smoke auf den 48 dichtesten train_main-Bildern bei 2048 (Rezept-Batch) und bei
   3072 (Batch 3, dann 2). **S_max-Regel unverändert:** 3072 wird nur trainiert, wenn es nach dieser Regel ohne
   Rezeptänderung passt; sonst S_max = 2048.
3. **Köpfe:** YOLO11s/12s haben nur den dichten Kopf (strukturell o2m, im Code „o2m“). Modellauswahl auf calib: Devkit-
   mAP50 ohne Cap dieses Kopfs (+ NMS des Prädiktors, wie in der VisDrone-Erweiterung). Einzelpass A_s und Kreuzung X:
   dieser Kopf + NMS, conf-Raster unverändert. Box-Fusion NMS_s: dieser Kopf (eine Zeile, Raster unverändert). F2, RG,
   B-F2, B-RG: derselbe Kopf. Damit: **D-H1** Δ = A_2048 − NMS_1024, **D-H2a/b** Δ = RG_s − NMS_s, **D-H3** Δ =
   RG_{S_max} − A_{S_max}, Kriterien unverändert. Sekundäre Kontraste „mit dem anderen Kopf“ entfallen für diese Modelle.
   SAHI wird für die neuen Modelle nicht gerechnet (sekundär, nicht Teil der Freigabe).
4. **Unverändert:** Primärmetrik (Devkit-mAP50 ohne Cap), Cap 4000 sekundär, „alle Objekte“ (difficult als normale GT),
   difficult als Positive im Training, Pipeline-Grenzen, ε = 0,30, Bootstrap (1000, Seed 0), Kreuzung (2048er Gewichte bei
   2048 / 2560 / 3072, Auswahl auf calib), Größenbins, crowded / isolated, GFLOPs und Latenz der D-H3-Zeilen.
5. **Holm und Urteil:** je Hypothese Holm über die **zwei neuen Modelle** (gleiche Familiengröße wie im ursprünglichen
   Plan); „Modell erfüllt“ = Kriterium und Holm-adjustiertes p ≤ 0,05. Berichtet wird **je Modell**; zusätzlich
   **deskriptiv die Anzahl der Modelle (von vier), die die Hypothese erfüllen** („gilt über Detektoren auf DOTA“ nur in
   dieser Form). Die ursprünglichen Zwei-Modell-Urteile bleiben unverändert.
6. **Ablauf:** Training → calib-Auswahl → Paritäts-Gate (Dekodierung des Kopfs, |ΔmAP50| ≤ 0,01 auf calib) → Caches
   (calib, DOTA-val) → calib-Tuning (Einzelpass-conf, NMS, F2, RG, B-RG) → Freeze-Vorschlag mit Replay jeder Zeile auf
   calib → **Stopp für die Freigabe** → DOTA-val einmal, nur für die neuen Zeilen. Latenz der neuen D-H3-Zeilen auf host-b im
   selben Verfahren (exklusiv, 575 W), in die Latenz-Aufteilung aufgenommen.
7. **Kosten (Schätzung, hochgerechnet aus DOTA YOLOv10 und VisDrone YOLO11s/12s):** Trainings YOLO11s 1024 ≈ 0,2 h,
   2048 ≈ 0,5 h; YOLO12s 1024 ≈ 0,45 h, 2048 ≈ 2,5 h (auf VisDrone 2048 nur Batch 2, ≈ 5× YOLO11s); calib-Auswertungen
   ≈ 0,3 h; Smokes, Gates, Einzelpässe, Kreuzung, Caches ≈ 1,2 h; Latenz ≈ 0,3 h. **Gesamt ≈ 5,5 GPU-h, konservativ
   (× 1,3) ≈ 7 GPU-h** (< 10 GPU-h: Start ohne weitere Rückfrage). CPU-Tuning je Gewichtssatz ≈ 20 min (8 Kerne).
8. **Zeitplan:** Migration STOP Sonntag 08:30 bleibt. Der Freeze-Vorschlag ist für die Nacht geplant; DOTA-val und Latenz
   der neuen Zeilen folgen nach der Freigabe, sofern sie vor 08:30 eintrifft (Bedarf ≈ 1 h). Sonst entscheidet der Nutzer,
   ob die Migration wartet (Latenzen nur auf host-b).

*Korrektur der Zeitangabe von Addendum 2 (2026-10-31, 15:50):* geschrieben ~14:55, committet 14:59 (Commit 992be47), nicht
„~16:00“; Inhalt unverändert.

## Addendum 3 – 2026-11-01 ~22:30 (Nutzerfreigabe: FCOS auf DOTA bei 1024 und 2048; vor jedem Training)

**Status: vom Nutzer freigegeben (2026-11-01, „GO for DOTA at 1024 and 2048 px“), committet vor dem ersten Training und
vor jedem FCOS-Ergebnis.** Alles oben (inkl. Addendum 1 und 2) gilt unverändert, soweit hier nichts anderes steht. Die
Ergebnisse und Urteile der vier YOLO-Modelle (`runs/dota/val/results.json`, `results_new.json`) bleiben, wie sie sind.
VisDrone: kein FCOS.

1. **Modell:** FCOS aus torchvision 0.29.1 (`fcos_resnet50_fpn`, ResNet-50-FPN, Gewichte `COCO_V1`), im Code Modellname
   `fcos`. Neue 16-Klassen-Ausgabeschicht (Prior 0,01); Klassifikations-Conv-Turm, Regressions- und Centerness-Zweig
   behalten die COCO-Gewichte; 3 trainierbare Backbone-Stufen (torchvision-Standard), eingefrorene BatchNorm im Backbone,
   GroupNorm im Kopf. Seed 0. Keine Änderung an torchvision.
2. **Rezept genau wie in `reports/FCOS_SPIKE.md` §3:** 60 Epochen; ganzes Bild auf lange Seite s ∈ {1024, 2048} skaliert
   (torchvision-Transform, ImageNet-Normierung, Seitenverhältnis erhalten), keine Kacheln im Training; Augmentierung
   **nur horizontales Spiegeln** (p = 0,5); SGD, Momentum 0,9, Weight Decay 1e-4, Lernrate 0,01 bei Batch 16 linear
   skaliert (effektive Batch 8 → 0,005), linearer Warm-up über Epoche 1, danach Cosinus-Abfall auf 1 % bis Epoche 60,
   AMP (fp16). Trainingslabels identisch mit den YOLO-Trainings (difficult = Positive, P1872 ausgeschlossen, P0334
   beschnitten).
3. **Batch:** 1024 px mit **Batch 8**; 2048 px mit **Batch 1 und Gradientenakkumulation über 8 Bilder** (dieselbe
   effektive Batch 8 wie bei 1024). **Begründung (vorab festgelegte Ausnahme nur für FCOS):** Die Regel „Batch ≥ 2“
   existiert wegen der BatchNorm-Statistik. torchvision-FCOS nutzt GroupNorm im Kopf und eingefrorene BatchNorm im
   Backbone; der Grund trifft daher nicht zu. Speicher-Smoke (`runs/fcos/memory.json`): Batch 2 bei 2048 passt auf dem
   dichtesten Trainingsbild (P1868, 10 206 GT) nicht (Zielzuordnung ≈ 11 GiB). Vor dem 2048er Training wird die
   S_max-Regel mit dem Speicher-Smoke auf den **48 dichtesten train_main-Bildern** bei 2048 (Batch 1) und zur
   Dokumentation bei 3072 (Batch 1) geprüft; trainiert werden nach der Freigabe nur 1024 und 2048 (S_max = 2048, sofern
   2048 passt).
4. **Modellauswahl auf calib:** Devkit-Port, mAP50 ohne Cap, Zeitplan wie bei den YOLO-DOTA-Trainings (jede 5. Epoche und
   jede der letzten 20; 28 Auswertungen), beste Epoche.
5. **Score (Korrektur der früheren Vorgabe „p × ctr“, Nutzer 2026-11-01):** Jede Konfiguration – Einzelpass, Kacheln +
   Box-Fusion, SCAHI – nutzt den eigenen Score des Detektors **sqrt(Klassenwahrscheinlichkeit × Centerness)** wie in
   torchvisions Nachverarbeitung. SCAHI-Cache: je Position und Klasse sqrt(p_c · ctr) als „Wahrscheinlichkeit“ (Zweigname
   `o2m` wie bei YOLO11/12, `std` = 0); Positionen mit max_c sqrt(p_c · ctr) ≥ 0,001 werden gecacht.
6. **Inferenz und Grenzen:** torchvision-Nachverarbeitung mit score_thresh 0,001, 200 000 Kandidaten je Ebene,
   NMS 0,6, **Pipeline-Grenze 200 000 je Bild und Pass**, auch für das calib-Tuning (bei FCOS würde die frühere
   calib-Grenze 20 000 bereits im Einzelpass greifen – bis 150 000 Detektionen je Bild im Spike –, und die Regel „die
   Grenze darf nicht greifen“ gilt unverändert). Je Zeile wird die Zahl der Bilder an der Grenze berichtet.
7. **Zeilen:** A_1024, A_2048 (Einzelpass: torchvision-Prädiktor, conf-Raster {0,001; 0,01; 0,05; 0,1} auf calib),
   Inferenz-Kreuzung X_2048 (2048er Gewichte bei 2048 / 2560 / 3072, Auswahl auf calib, sekundär), NMS_s (Kacheln + Box-
   Fusion: die Pass-Detektionen des Detektors – torchvision-Nachverarbeitung je Pass, Boxen auf den Pass beschnitten –,
   dann NMS über die Pässe mit dem unveränderten IoS-Raster, 12 Konfigurationen), F2_s, RG_s, B-F2_s, B-RG_s (Raster und
   Auswahlregeln unverändert, 2×-Zeitgrenze). **Keine SAHI-Zeile** für FCOS.
8. **Paritäts-Gate (FCOS-Fassung, gleiche Rolle wie bei YOLO):** (a) eigene Nachverarbeitung der dichten Ausgaben gegen
   torchvisions eigene Nachverarbeitung derselben Vorwärtsrechnung, bitgleich, für jede verwendete Eingabegeometrie
   (ganzes Bild bei s, 1024er Kachel; je 3 calib-Bilder); (b) calib-mAP50 (primär, ohne Cap) des Einzelpasses aus den
   eigenen Pass-Detektionen gegen den torchvision-Prädiktor, |Δ| ≤ 0,01. Ohne bestandenes Gate keine Caches und keine Fusion.
9. **Hypothesen und Regeln unverändert:** D-H1 Δ = A_2048 − NMS_1024, D-H2a/b Δ = RG_s − NMS_s, D-H3 Δ =
   RG_{S_max} − A_{S_max}, Kriterien wie oben (ε = 0,30; D-H3 zusätzlich untere KI-Grenze > 0); primär Devkit-mAP50 ohne
   Cap; sekundär Cap 4000, „alle Objekte“, AP@[.5:.95], Größenbins, crowded / isolated; S_max-Regel; Inferenz-Kreuzung;
   Bootstrap 1000, Seed 0. **Ein Modell → kein Holm**: Urteil je Modell mit dem unadjustierten einseitigen p ≤ 0,05 und
   dem Kriterium. Zusätzlich deskriptiv die Anzahl der Modelle **von fünf** DOTA-Detektoren, die eine Hypothese erfüllen;
   die früheren Urteile bleiben unverändert.
10. **Ablauf:** Speicher-Smoke (S_max-Regel) → Training 1024 und 2048 → calib-Auswahl → Paritäts-Gate → Caches (calib,
    DOTA-val) → calib-Tuning (Einzelpass-conf und Kreuzung, Box-Fusion, F2, RG, B-F2, B-RG) → Freeze-Vorschlag mit Replay
    jeder Zeile auf calib → **Stopp für die Freigabe** → DOTA-val einmal, nur für die FCOS-Zeilen. Latenz der FCOS-Zeilen
    nicht Teil dieser Freigabe.
11. **Kosten (aus dem Spike):** Training 1024 ≈ 1,4 GPU-h, 2048 ≈ 6,2 GPU-h; Gates, Einzelpässe, Kreuzung, Caches
    ≈ 0,6 GPU-h; CPU-Tuning ≈ 0,5–1 h je Gewichtssatz. Gesamt ≈ 8–9 GPU-h.

*Korrektur der Zeitangabe von Addendum 3 (2026-11-01, 22:28):* geschrieben und committet 20:34 (Commit 6aa3ebf, Commit-Zeit
2026-11-01T20:34:36+01:00), nicht „~22:30“; Inhalt unverändert. Erstes Artefakt des FCOS-Trainingslaufs 20:40:20
(`runs/dota/fcos_smoke.json`).

## Addendum 4 – 2026-11-02 (Nutzerauftrag: SCAHI als Ersatz für den SAHI-Merge; vor jedem Ergebnis dieser Zeilen)

**Status: vom Nutzer beauftragt (2026-11-02), committet vor jedem Cache, Gate- oder Tuning-Ergebnis dieser Zeilen.**
Alles oben (inkl. Addendum 1–3) gilt unverändert, soweit hier nichts anderes steht. Alle bisherigen Zeilen, Ergebnisse und
Urteile (`runs/dota/val/results.json`, `results_new.json`, die SAHI-Zeilen eingeschlossen) bleiben, wie sie sind.
**Vorrang:** Die laufende FCOS-Arbeit (Addendum 3) hat Vorrang bei CPU und Speicher; nichts hier darf sie verlangsamen oder
ihr Speicherbudget gefährden.

1. **Frage (D-H5):** Schlägt SCAHI auf **genau den Pässen** der eingefrorenen SAHI-Konfiguration (dieselben 1024er Gewichte,
   dieselbe Slice-Größe und Überlappung, derselbe herunterskalierte Pass über das ganze Bild) SAHIs eigenen Merge?
2. **Modelle:** YOLOv10s und YOLOv10m (die Detektoren mit eingefrorenen SAHI-Zeilen auf DOTA), Gewichte und SHA-256 wie in
   `configs/dota/frozen_rows.json` (`SAHI_yolov10s`, `SAHI_yolov10m`).
3. **Pässe (SAHI-Layout, je Modell aus der eingefrorenen Zeile):** Slices 640 × 640 px nach `sahi.slicing.get_slice_bboxes`
   (sahi 0.12.8, wie `get_sliced_prediction`), Überlappung 0,1 (YOLOv10s) bzw. 0,2 (YOLOv10m), jeder Slice mit dem
   Ultralytics-Letterbox auf 640 px; dazu SAHIs Standard-Pass: ganzes Bild mit langer Seite 640 px. Eine Vorwärtsrechnung je
   Eingabe; der Cache speichert wie alle Caches die dichten Ausgaben beider Köpfe (o2o, o2m; Positionen mit max.
   Klassenwahrscheinlichkeit ≥ 0,001), calib und DOTA-val.
4. **Paritäts-Gate (Regel wie bisher):** (a) Je Bild, auf calib und DOTA-val: die o2o-Pass-Detektionen aus dem Cache
   (`topk_like_ultralytics`, conf ≥ 0,001, Grenze 200 000 je Pass wie die eingefrorenen SAHI-Rohvorhersagen) gegen die
   gespeicherten Rohvorhersagen hinter den eingefrorenen SAHI-Zeilen (`runs/dota/fusion/<m>_1024/sahi_raw_<split>_o<ov>.pkl`):
   gleiche Anzahl je Bild, gleiche Klassen, Koordinaten und Scores ≤ 4 ulp (float32). Auf DOTA-val wird dabei keine
   Metrik berechnet. (b) calib-mAP50 (primär, ohne Cap) der eingefrorenen SAHI-Konfiguration aus den Cache-Pässen gegen
   die eingefrorene `calib_mAP50`, |Δ| ≤ 0,01. Ohne bestandenes Gate kein Tuning; Stopp und Bericht.
5. **Zeilen (je Modell):**
   - **SAHI** – wie eingefroren (Referenz, o2o, SAHI-Merge); bereits ausgewertet, **keine Neuauswertung**; für den
     gepaarten Bootstrap werden die gespeicherten DOTA-val-Detektionen (`runs/dota/val/dets/SAHI_<m>.parquet`) genutzt.
   - **SAHI-F2** – SAHI-Pässe + SCAHI (o2m, wie alle YOLOv10-SCAHI-Zeilen), vorab festgelegt: VisDrone-F2-Raster
     (24 Konfigurationen, iou_cluster 0,5), Auswahl auf calib.
   - **SAHI-RG** – SAHI-Pässe + SCAHI (o2m), verfeinertes Raster: 108 Varianten + Referenz (die calib-gewählte
     SAHI-F2-Konfiguration), Auswahl max. calib-mAP50 unter Fusionszeit je Bild ≤ 2× Referenz (im selben Prozess gemessen),
     Gleichstand → kleinere Fusionszeit.
   Tuning nur auf calib, Verfahren, Raster, Auswahlfunktion (`scahi.dota.pipeline.select`) und Pipeline-Grenze 200 000
   unverändert wie bei den übrigen SCAHI-Zeilen. SAHI behält seinen eingefrorenen Kopf (o2o).
6. **Hypothese D-H5:** Δ = SAHI-RG − SAHI auf DOTA-val, primär Devkit-mAP50 ohne Cap; Kriterium Δ > ε (ε = 0,30);
   gepaarter Bootstrap über DOTA-val-Bilder (1000 Resamples, Seed 0, Perzentil-KI), einseitiges p wie oben, **Holm über die
   zwei Modelle**, α = 0,05. Ein Modell erfüllt D-H5, wenn Kriterium und Holm-p ≤ 0,05 gelten; **D-H5 gilt, wenn beide
   Modelle sie erfüllen.**
7. **Sekundär, ohne Urteil:** SAHI-F2 − SAHI; Cap 4000; „alle Objekte“; AP@[.5:.95]; Größenbins wie bei den übrigen
   DOTA-Zeilen. **Latenz** auf host-a in **einem exklusiven Lauf** (keine anderen GPU-/CPU-Jobs; also nach dem FCOS-Tuning),
   für SAHI, SAHI-F2 und SAHI-RG beider Modelle, berichtet als Pässe + Merge-Schritt (SAHI-Merge bzw. SCAHI-Fusion),
   Protokoll wie `runs/dota/val/latency.json`.
8. **Ressourcen:** GPU-Caches sofort, wenn die GPU frei ist (in einem eigenen Scope mit Speicherdeckel, oom_score_adj 1000,
   auf den CPU-Kernen, die das FCOS-Tuning nicht nutzt). CPU-Tuning erst nach dem Ende des FCOS-Tunings, mit denselben
   Speicherregeln (Scope-Deckel, ≥ 20 GB frei, oom_score_adj 1000).
9. **Ablauf:** Commit dieses Addendums → Caches (calib, DOTA-val) → Paritäts-Gate → (nach FCOS) calib-Tuning SAHI-F2,
   SAHI-RG → Freeze-Vorschlag mit Replay jeder neuen Zeile auf calib → **Stopp für die Freigabe** → DOTA-val einmal, nur
   für die neuen Zeilen.

*Korrektur zu Addendum 4, Punkt 4 (2026-11-02, vor jedem Gate-Ergebnis über ganze Splits und vor jeder mAP dieser Zeilen):*
Die Rohvorhersagen hinter den eingefrorenen SAHI-Zeilen stammen aus Ultralytics' **fusioniertem** Modell mit TF32-Faltungen
(PyTorch-Standard), die Caches aus dem **unfusionierten** Modell (wie alle Caches), ebenfalls mit TF32. Beide Rechnungen
runden verschieden (Smoke auf 4 calib-Bildern: Scores bis ≈ 0,01, Boxen p99 ≈ 3 px; Netzeingaben bitgleich; mit TF32 aus
stimmen die Vorwärtsrechnungen überein). „≤ 4 ulp gegen die gespeicherten Rohvorhersagen“ ist damit nicht erreichbar und
war nicht die frühere Regel. **Es gilt die Regel wie bisher** (`scahi.dota.pipeline.gate`, Abschnitt 5): (1) an beiden neuen
Eingabegeometrien (Slice 640 px, ganzes Bild mit langer Seite 640; je 3 calib-Bilder, TF32 aus) eigene Dekodierung gegen
Ultralytics' Dekodierung derselben Vorwärtsrechnung, je Koordinate ≤ max(1e-4 px, 4 float32-ulp), Wahrscheinlichkeiten
< 1e-4, und die o2o-Pass-Detektionen, wie der Cache-Pfad sie bildet, gegen Ultralytics' End-to-End-Ausgabe; (2) calib-mAP50
(primär, ohne Cap) der eingefrorenen SAHI-Konfiguration aus den Cache-Pässen gegen die eingefrorene `calib_mAP50`,
|Δ| ≤ 0,01. Deskriptiv, ohne Urteil: je Bild auf calib und DOTA-val Anzahl und gepaarte Abweichungen der Cache-Pass-
Detektionen gegenüber den gespeicherten Rohvorhersagen (auf DOTA-val ohne Metrik). Im Smoke gefunden und behoben (nur im
neuen Modul `scahi.dota.sahi_scahi`): Ultralytics' o2o-Nachverarbeitung nutzt in beiden Top-k-Stufen k = min(max_det,
Ankerzahl der **ganzen** Eingabe); aus dem Cache (nur Anker mit max. Klassenwahrscheinlichkeit ≥ 0,001) muss k daher aus
der Eingabeform des Passes kommen. Außerdem führt SAHI den Standard-Pass nur bei mehr als einem Slice aus; das gilt auch
für die SCAHI-Zeilen (dieselben Pässe). Offengelegt: Teil (1) lief bereits als Funktionstest auf genau diesen je 3 calib-Bildern
(beide Modelle, bestanden); keine mAP und keine Zahl auf DOTA-val vor diesem Commit.

## Addendum 5 – 2026-11-03 (Nutzerauftrag: Q4 für Ein-Kopf-Detektoren; vor jedem Ergebnis dieser Zeilen)

**Status: vom Nutzer beauftragt (2026-11-03), committet vor jeder SAHI-Rohvorhersage, jedem Cache-, Gate- oder
Tuning-Ergebnis dieser Zeilen.** Alles oben (inkl. Addendum 1–4) gilt unverändert, soweit hier nichts anderes steht. Alle
bisherigen Zeilen, Ergebnisse und Urteile bleiben, wie sie sind; insbesondere bleibt das Urteil zu D-H5 (YOLOv10)
unverändert. **Vorrang:** Die laufende FCOS-Arbeit (Addendum 3) behält Vorrang bei CPU und Speicher.

1. **Frage (D-H6):** Bei YOLOv10 nutzt SAHI den o2o-Kopf und SCAHI den o2m-Kopf. YOLO11s und YOLO12s haben einen einzigen
   dichten Kopf; auf **denselben Pässen und demselben Kopf** isoliert der Vergleich den Merge-Schritt: Schlägt SCAHI auf
   genau den Pässen der calib-gewählten SAHI-Konfiguration SAHIs eigenen Merge?
2. **Modelle und Gewichte:** YOLO11s und YOLO12s, jeweils die 1024er Gewichte aus Addendum 2 (wie
   `configs/dota/frozen_rows_new.json`, `A_<m>_1024`: YOLO11s SHA-256 `031e495c…df76e4`, YOLO12s `e76d111e…19ccf5`).
3. **Zeile SAHI (neu getunt, je Modell):** exakt Verfahren und Raster der eingefrorenen YOLOv10-SAHI-Zeilen
   (`scahi.dota.pipeline.sahi_raw` / `sahi_tune`, Raster v2 aus `scahi.infer.sahi_baseline`): Slices 640 × 640 px nach
   `get_sliced_prediction` (sahi 0.12.8) mit Standard-Pass (ganzes Bild, lange Seite 640, nur bei mehr als einem Slice),
   Überlappungen 0,1 / 0,2 / 0,3; Merges GREEDYNMM / NMM / NMS × IOU / IOS × Schwellen {0,3 … 0,8} × conf {0,01; 0,1} und
   WBF × 6 Schwellen × 4 skip-Schwellen (288 Konfigurationen je Modell); Prädiktor-Konfidenz 0,001, max_det je Slice =
   Pipeline-Grenze 200 000, kein 500er-Cap nach dem Merge; Auswahl max. calib-mAP50 (primär, ohne Cap), wie `sahi_tune`.
   Einziger, durch das Modell bedingter Unterschied: die Pass-Nachverarbeitung ist Ultralytics' Standard für einen
   Ein-Kopf-Detektor (NMS je Pass, IoU 0,7, Standard des Prädiktors) statt der o2o-Ausgabe von YOLOv10.
4. **Pässe der SCAHI-Zeilen (SAHI-Layout der gewählten SAHI-Konfiguration):** dieselben Slices und derselbe Standard-Pass
   (gewählte Überlappung), fusioniertes Modell des SAHI-Prädiktorpfads, Ultralytics-Eingabevorbereitung (Letterbox,
   `scale_boxes`), ein Slice je Vorwärtsrechnung – wie D-H5 (`scahi.dota.sahi_scahi`). Der Cache speichert die dichten
   Ausgaben des einzigen Kopfes (Positionen mit max. Klassenwahrscheinlichkeit ≥ 0,001), calib und DOTA-val.
5. **Paritäts-Gate (Regel wie D-H5 / `pipeline.gate`):** (1) an beiden Eingabegeometrien (Slice 640 px, ganzes Bild mit
   langer Seite 640; je 3 calib-Bilder, TF32 aus) eigene Dekodierung gegen Ultralytics' Dekodierung derselben
   Vorwärtsrechnung (je Koordinate ≤ max(1e-4 px, 4 float32-ulp), Wahrscheinlichkeiten < 1e-4), und die Pass-Detektionen,
   wie der Cache-Pfad sie bildet (NMS je Pass wie Ultralytics), gegen Ultralytics' Ausgabe; (2) calib-mAP50 (primär,
   ohne Cap) der gewählten SAHI-Konfiguration aus den Cache-Pässen gegen ihren calib-mAP50 aus dem SAHI-Tuning,
   |Δ| ≤ 0,01. Deskriptiv: je Bild Anzahl und Abweichungen gegenüber SAHIs eigenen Rohvorhersagen (DOTA-val ohne
   Metrik). Ohne bestandenes Gate kein SCAHI-Tuning; Stopp und Bericht.
6. **Zeilen (je Modell):** **SAHI** (Punkt 3); **SAHI-F2** – SAHI-Pässe + SCAHI, vorab festgelegt (VisDrone-F2-Raster,
   24 Konfigurationen, iou_cluster 0,5, Auswahl auf calib); **SAHI-RG** – SAHI-Pässe + SCAHI, verfeinertes Raster
   (108 Varianten + Referenz = calib-gewählte SAHI-F2-Konfiguration; Auswahl max. calib-mAP50 unter Fusionszeit je Bild
   ≤ 2× Referenz, Option-3-Zeitmessung: Referenz und Variante je Task im selben Prozess auf derselben Maschine;
   Gleichstand → kleinere Fusionszeit). Alle SCAHI-Zeilen auf genau den Pässen der gewählten SAHI-Konfiguration und
   demselben Kopf wie SAHI. Tuning nur auf calib; Verfahren, Raster, Auswahlfunktion (`pipeline.select`) und
   Pipeline-Grenze 200 000 wie bei D-H5.
7. **Hypothese D-H6:** Δ = SAHI-RG − SAHI auf DOTA-val, primär Devkit-mAP50 ohne Cap; Kriterium Δ > ε (ε = 0,30);
   gepaarter Bootstrap über DOTA-val-Bilder (1000 Resamples, Seed 0, Perzentil-KI), einseitiges p, **Holm über die zwei
   Modelle**, α = 0,05. Ein Modell erfüllt D-H6, wenn Kriterium und Holm-p ≤ 0,05 gelten; **D-H6 gilt, wenn beide
   Modelle sie erfüllen.**
8. **Sekundär, ohne Urteil (wie D-H5):** SAHI-F2 − SAHI; Cap 4000; „alle Objekte“; AP@[.5:.95]; Größenbins. **Latenz** nur,
   wenn vor Freitag 2026-11-06 ein exklusives Fenster passt (Protokoll wie D-H5); sonst für diese Zeilen keine Latenz.
9. **Ressourcen:** GPU-Schritte sofort (GPU frei; FCOS rechnet nur auf der CPU). CPU-Schritte im 10G-Slot (Kerne 0–1,
   MemoryMax 10G, MemorySwapMax 0, oom_score_adj 1000). Der FCOS-Deckel bleibt bei 95G, bis dieses Addendum
   abgeschlossen ist; danach 105G wie geplant.
10. **Ablauf:** Commit dieses Addendums → SAHI-Rohvorhersagen calib für alle drei Überlappungen (GPU) → SAHI-Tuning (CPU)
    → Caches des gewählten SAHI-Layouts, calib und DOTA-val (GPU) → Paritäts-Gate → SAHI-F2, SAHI-RG calib-Tuning (CPU)
    → **ein gemeinsamer Freeze-Vorschlag** (SAHI- und SCAHI-Zeilen beider Modelle) mit Replay jeder Zeile auf calib →
    **Stopp für die Freigabe** → DOTA-val einmal (SAHI-Rohvorhersagen DOTA-val der gewählten Überlappung, alle sechs
    Zeilen). Frist für den Freeze-Vorschlag: Donnerstag 2026-11-05 abends.
