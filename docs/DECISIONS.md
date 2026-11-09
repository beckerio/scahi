# Entscheidungen und Planabweichungen

Public copy; venue-specific details are redacted. Dates, decisions and reasons are unchanged.

| Datum | Entscheidung | Begründung |
|---|---|---|
| 2026-10-21 | Datenpfade: aufbereitet = `<restricted dataset copy, not used> Original = `/path/to/datasets/VisDrone/VisDrone2019-DET`, Toolkit = `/path/to/datasets/VisDrone/VisDrone2018-DET-toolkit` (`configs/data/visdrone.yaml`). | Platzhalter im Auftrag waren nicht gesetzt; per Dateisuche gefunden. Original wird im Audit byte-genau gegen die offiziellen ZIPs (`Visdrone_2021/*.zip`) geprüft, daher kein erneuter Download. Spiegelkopien existieren unter `<external drive>/Datasets/Datasets/`. |
| 2026-10-21 | PyTorch 2.14.1+cu130 (pip, Index `download.pytorch.org/whl/cu130`). | Treiber 595.99 meldet CUDA 13.2; RTX 5090 (sm_120). |
| 2026-10-21 | Octave 10.3.0 aus conda-forge im Environment. | Ermöglicht die Validierung von `scahi eval` gegen `evalDET.m` (Phase 1). |
| 2026-10-21 | Audit-Matching klassenagnostisch; YOLO-Koordinaten mit der Original-Bildgröße denormalisiert. | Ziel ist das Aufdecken des Mappings; normierte Koordinaten sind resize-invariant. |
| 2026-10-21 | „Stark geändert“ = max. Kanaldifferenz > 30; Ignore-Masken für die „außerhalb“-Messung um 8 px dilatiert. | Trennt JPEG-Rekodierungsrauschen (≈ 0,5 Graustufen) von inhaltlicher Änderung. |
| 2026-10-21 | **Nutzerentscheid Klassen-Settings:** primär `visdrone10` (toolkitkonform); sekundär `person_vehicle` (umbenannt aus `vehicle.yaml`; person = 1+2, vehicle = 4+5+6+9). Umfang sekundär: nur B-FULL, B-SAHI, T2, T4, beste Fusionsvariante. | Freigabe Phase 0. |
| 2026-10-21 | **Nutzerentscheid Ignore-Semantik (beide Settings):** Ignore ist klassenspezifisch (Kanalmaske pro Ignore-Box) und maskiert nur negative Beiträge; vom Assigner positiv zugeordnete Anchors behalten den vollen Loss. person_vehicle: verworfene Kategorien 3/7/8/10 → `ignore_vehicle` (nur Kanal vehicle). COCO: Feld `ignore` mit `ignore_classes`; YOLO: `ignore_all` = nc, `ignore_vehicle` = nc+1. | Freigabe Phase 0, siehe docs/PLAN_AMENDMENTS.md. |
| 2026-10-21 | Ultralytics-Settings repo-lokal: `conda env config vars set YOLO_CONFIG_DIR=<repo>/.ultralytics` (gitignored), dort `sync=False`, `datasets_dir=<repo>/data`, `runs_dir=<repo>/runs/ultralytics`, `weights_dir=<repo>/weights`. Zusätzlich setzt `scahi/__init__.py` die Variable per `setdefault` (direkter Aufruf des Env-Pythons lädt conda-Env-Variablen nicht). Die globale `~/.config/Ultralytics/settings.json` wurde beim ersten Import (vor dieser Entscheidung) einmal von Ultralytics auf ein neues Schema migriert und wird seitdem nicht mehr angefasst. | Nutzervorgabe Punkt 7. |
| 2026-10-21 | **Auswertung = exakter Port des Toolkits** (Upstream-Commit 0054457) als Headline von `visdrone_ignore`; pycocotools (maxDets 1/10/100/500) zusätzlich auf *denselben* gedroppten Daten für AP pro Klasse, Größenbins (AP, AP50, AR) und als zweite Gesamtzahl (`metrics.coco`). | Planabweichung: Das Toolkit rechnet AP nicht mit pycocotools, sondern mit eigenem Code (VOCap all-point, maxDets pro Bild über alle Klassen in Dateireihenfolge, Klassenmittel gewichtet mit der Zahl der Bilder pro Klasse). Mit pycocotools allein wäre die Vorgabe „≤ 0,1 AP Abweichung zu evalDET.m“ nicht erreichbar. |
| 2026-10-21 | Toolkit-Drop-Regel: nur Kat. 0 (ignored regions) entfernt GT und Detektionen; 'others' (11) ist im Toolkit lediglich keine ausgewertete Klasse und wirkt nicht als Ignore-Region. Der Port folgt dem; im *Training* bleibt 'others' Ignore (`ignore_all`). | Treue zum Toolkit; unterschiedliche Rollen von Training und Evaluation dokumentiert. |
| 2026-10-21 | person_vehicle-Protokolle: `visdrone_ignore` (Kat. 0 für alle Klassen + Kat. 3/7/8/10 nur für vehicle, gleiche Flächenregel ≥ 0,5 wie das Toolkit), `visdrone_strict` (nur Kat. 0; verworfene Kategorien = Hintergrund), `plain`. Toolkit-Validierung nur für visdrone10. | Nutzervorgabe Punkt 3. |
| 2026-10-21 | Validierung gegen `evalDET.m` mit synthetischen Detektionen (gestörte GT, Klassenverwechslung, Duplikate, Zufalls-FP, FP in Ignore-Regionen, > 100 Detektionen/Bild) auf ganz val; Octave führt die unveränderten Upstream-Funktionen `saveAnnoRes` + `calcAccuracy` aus. Einziges Shim: `mean2.m` (= `sum(x(:))/numel(x)`, fehlt in Octave ohne image-Paket). Erneute Validierung mit echten Detektionen in Phase 2. | Noch kein Modell vorhanden; synthetische Fälle decken alle Codepfade ab. |
| 2026-10-21 | Diagnostik-Definitionen (`scahi/eval/diagnostics.py`): Duplikat = ungematchte Detektion mit IoU ≥ 0,5 zu bereits gematchter GT derselben Klasse; Fragment = ungematchte Nicht-Duplikat-Detektion mit IoF ≥ 0,8 zu einer GT bei IoU < 0,5 zu allen; Raten relativ zur Zahl der Detektionen mit score ≥ 0,25; Kachelrandabstand = Abstand des Boxzentrums zur nächsten *inneren* Kachelkante. | Auftrag lässt Normierung und conf offen. |
| 2026-10-21 | (überholt am 2026-10-22) Kalibrier-Split zunächst bildweise wie spezifiziert. | train besteht aus 208 Videosequenzen; 645/647 bildweise gezogene calib-Bilder haben Nachbarframes in train_main. |
| 2026-10-21 | YOLO-Labels: Boxen auf das Bild geclippt, Boxen < 1 px Breite/Höhe nach Clipping entfallen (train: 3 mit `yolo_ignore`, 1 mit `yolo_plain`); COCO-JSON behält alle Originalzeilen unverändert (Roundtrip-getestet). | Ultralytics verlangt normierte Koordinaten in [0, 1]. |
| 2026-10-21 | `pyarrow` nachinstalliert (Parquet für Detektionen/Caches); `requirements.lock` aktualisiert. | |
| 2026-10-22 | **Nutzerentscheid Kalibrier-Split:** sequenzgruppiert. Da die unstratifizierte Sequenzziehung deutlich abwich (16–32 px: 41,6 % vs. 34,4 % in train_main; 64–128 px: 7,8 % vs. 11,6 %; car 47,6 % vs. 42,0 %), wird **größenstratifiziert** gezogen: beste von 5000 Seed-reproduzierbaren Ziehungen ganzer Sequenzen mit Bildanteil 9–11 %, Kriterium L1(Größenbins) + 0,5·L1(Klassen) gegenüber ganz train. Ergebnis: 594 Bilder, 30 Sequenzen, max. Abweichung ~1,4 Prozentpunkte. Aktiv: `calib.txt`/`train_main.txt`; dokumentiert: `calib_imagewise.txt`, `calib_by_sequence_random.txt`. Statistik: `runs/p1/*_convert/splits_summary.json`. | Nutzervorgabe 1a. |
| 2026-10-22 | Sequenzüberlappung der offiziellen Splits nur dokumentiert (train ↔ val: 24 gemeinsame Sequenz-IDs, 297/28 Bilder; test-dev: keine). Splits unverändert. | Nutzervorgabe 1b. Gleiche ID heißt nicht zwingend gleiches Video. |
| 2026-10-22 | **Nutzerentscheid Auswertung:** Headline = Toolkit-Port; pycocotools nur für Aufschlüsselungen. Jede Kennzahl trägt ihre Quelle (`metrics.sources`), Tabellen kennzeichnen die Quelle. **Ausgabevertrag aller Methoden** (inkl. SAHI-Bibliothek und Fusion): pro Bild nach Score sortiert, auf 500 Detektionen über alle Klassen gekappt (`scahi.eval.detections.finalize_detections`); `scahi eval` erzwingt dies zusätzlich und protokolliert `n_det_capped`. | Nutzervorgabe 2. Die Toolkit-Validierung vom 21.10. lief vor Einführung des Kappens auf ungekappten synthetischen Dateien; der Port selbst ist unverändert. |
| 2026-10-22 | **Fehlerhafte GT:** In der Auswertung unverändert (Toolkit-Treue). Im Training (YOLO-Labels): Duplikate gleicher Klasse (IoU ≥ 0,99, die erste Box bleibt) und Boxen < 1 px nach Clipping entfernt (train visdrone10: 4 Duplikate, 3 bzw. 1 degeneriert). **Korrektur Phase-0-Befund:** Die 3 im Audit genannten „Duplikate“ sind identische Boxen mit zwei verschiedenen Kategorien (Klassenkonflikte; train: 4, val: 1, test-dev: 1); sie bleiben vorerst in beiden Labels – Nutzerentscheid offen. | Nutzervorgabe 4. |
| 2026-10-22 | B-SAHI ruft `get_sliced_prediction` mit `force_postprocess_type=True` auf. | SAHI 0.12.8 (`sahi/predict.py`, `LOW_MODEL_CONFIDENCE = 0.1`) schaltet sonst bei niedrigem conf stillschweigend von GREEDYNMM auf NMS/IOU um; die GREEDYNMM-Baseline wäre dann keine. |
| 2026-10-22 | Gewichte: `weights/yolov10s.pt` (SHA256 65fa8332…), `weights/yolov10m.pt` (SHA256 6dc78f7a…), Ultralytics-Assets v8.4.0. | Reproduzierbarkeit. |
| 2026-10-22 | **YOLOv10-Zweige in Ultralytics 8.4.183:** Checkpoints laden mit `end2end=False`; der Predictor setzt `end2end = (args.nms is False)` und fused den nicht gewählten Zweig weg. Default (`nms=None`) = **o2m + NMS** (iou 0,7), *nicht* das NMS-freie o2o. Wir wählen o2o explizit (`predict(nms=False)`; in SAHI über `YOLO.overrides["nms"]=False`). B-FULL berichtet beide Zweige (`fusion=none`/branch o2o = Hauptzeile; `fusion=nms`/branch o2m+nms = Ultralytics-Default); Modellauswahl auf calib über o2o, o2m-Kurve wird mitprotokolliert. SAHI-Baselines nutzen o2o pro Slice. | Auftrag nahm an, Fusing entferne o2m; in dieser Version entfernt der Default o2o. Erster B-FULL-640-Lauf (Start 08:56) wurde deshalb nach ~10 Epochen abgebrochen und neu gestartet. |
| 2026-10-22 | **Trainings-Loss YOLOv10 in 8.4.183:** `DetectionModel.init_criterion` → `E2ELoss` (beide Zweige, je `v8DetectionLoss`): o2m `tal_topk=10`, o2o `tal_topk=7, tal_topk2=1`, Gewichte o2m 0,8 → 0,1 (Decay), o2o = 1 − o2m. Das ist nicht das Original-YOLOv10-Rezept (`E2EDetectLoss`: Summe, topk 10/1). Für Phase 3 (Loss-Subklasse) relevant. | API gegen installierte Version geprüft. |
| 2026-10-22 | **Nutzerentscheide Phase 2:** B-SAHI-FT ab COCO-Gewichten, 20 Epochen; Regel: alle Slicing-Trainings (B-SAHI-FT, T0–T4) mit denselben COCO-Gewichten und demselben Budget an gesehenen Bildern (Referenz B-SAHI-FT), Budget in `metrics.json` (`training`). B-FULL 100 Epochen. Klassenkonflikt-Annotationen bleiben. 1 Seed pro Baseline + 2 Seeds B-FULL 640 am Ende. | Freigabe 16,3 GPU-h. |
| 2026-10-22 | **Reproduzierbarkeitshinweis SAHI:** B-SAHI mit `force_postprocess_type=True`; zusätzlich Infozeile `sahidefault` (Bibliotheksstandard: bei conf < 0,1 Umschaltung GREEDYNMM → NMS/IOU). Tuning: ein Bibliothekslauf pro Overlap mit zur Laufzeit registrierter Identitäts-Nachbearbeitung, Merging offline mit SAHIs eigenen Klassen (auf 5 Bildern als identisch zum direkten Aufruf verifiziert); finale val-Zeilen über `get_sliced_prediction`. | Nutzervorgabe 5. |
| 2026-10-22 | (korrigiert, siehe Aufklärung unten) Reines Training ~3× schneller als geschätzt: B-FULL 640 ≈ 16 s/Epoche (2,7 ms/Bild) statt 50 s; der 400-Bilder-Benchmark war von Start-Overhead dominiert. Für den ganzen Job gilt das nicht. | Gemessen im Queue-Lauf. |
| 2026-10-22 | **Phase-4-Rohzugriff (Vorarbeit):** Ungefustes `DetectionModel` liefert im Eval-Modus `(y, preds)`; `preds[branch]` = `{"boxes": [B, 4·16, A], "scores": [B, nc, A], "feats": [P3, P4, P5]}` mit über die Level konkatenierten Anchors (nicht pro Level `[B, 4·reg_max+nc, H, W]` wie im Auftrag angenommen). Eigenes DFL-Decoding (Softmax, Erwartungswert, Varianz, ltrb·stride) + Nachbau des zweistufigen Top-k (`Detect.postprocess`): Parität o2o max. 3·10⁻⁵ px / 0 Score (Toleranz 1e-4), o2m 1,2·10⁻⁴ px (xywh-Umrechnung in float32). | `tests/test_raw_head.py`. |
| 2026-10-22 | Eigenes Preprocessing folgt exakt Ultralytics `LetterBox` (INTER_LINEAR, zentriertes Padding 114, Rundung `round(d − 0.1)`); Vollbild-Pass: lange Seite = Zoom × lange Bildseite, beide Seiten auf Vielfache von 32. | Ein erster Entwurf mit INTER_AREA + Padding oben links wich bei ⅓-Skalierung um bis zu 0,07 im Score ab; jetzt Cache = Predictor (Score ±1e-3, Box ±0,1 px). |
| 2026-10-22 | Zoom-Planer (`scahi tile --plan`, `scahi/data/zoom.py`): Kacheln der Größe T = N/z in Originalpixeln, Overlap 0,25, Randkacheln nach innen verschoben; σ-Verteilung nur aus train (= train_main + calib). Startkandidat {⅓, 1, 2}: 97,9 % der GT im Band, Bandüberlappung 2,4/3,0 Oktaven, Kosten 1/8/40 Pässe (relativ 1 : 14 : 71 Eingabe-Megapixel). | `runs/p3/20261022_tileplan/plan.json`. |
| 2026-10-22 | Brückenexperiment trainiert über einen Symlink-Spiegel `data/restricted_mirror/train/{images,labels}`. | Ultralytics schreibt den Label-Cache neben das `labels/`-Verzeichnis und hätte sonst die `labels.cache` der fremden restricted-Kopie überschrieben (geprüft: bleibt unverändert). Hinweis: Ultralytics entfernt exakt doppelte Label-Zeilen beim Laden selbst. |
| 2026-10-22 | `sahidefault`-Zeile (B-FULL 640) ist identisch mit B-SAHI GREEDYNMM, weil das Tuning conf = 0,1 wählte; die Bibliothek schaltet erst bei conf < 0,1 um. | Ergebnis, kein Fehler; `runs/p2/20261022_v10s_BFULL640_sahi640o10_*`. |
| 2026-10-22 | Octave-Abgleich mit echten Detektionen (B-FULL 640, val, 251 441 Detektionen): Abweichung 4·10⁻¹¹ → Toolkit-Port validiert. | `runs/p2/20261022_toolkit_validation_real/toolkit_validation.json`. |
| 2026-10-22 | **Nutzerentscheid Zweige:** Hauptzeile o2o für B-FULL und für alle Box-Fusions-Baselines pro Kachel; o2m+NMS jeweils Zusatzzeile. Phase-5-Fusionsvarianten testen beide Zweige (H3). SAHI/Box-Fusion mit o2m pro Kachel als zusätzliche Baseline (B-FULL 640, später B-SAHI-FT), an das Queue-Ende gehängt (`configs/queue/phase2b.txt`, Start per `scripts/chain_queue.sh` nach Ende der laufenden Queue). | Freigabe 22.10. |
| 2026-10-22 | **Trainings-Loss bleibt Ultralytics 8.4 für alle Trainings; Bezeichnung in allen Reports: „YOLOv10 (Ultralytics 8.4)“.** Details (`ultralytics/utils/loss.py`, `E2ELoss`): beide Zweige mit `v8DetectionLoss`; o2m `tal_topk=10`, o2o `tal_topk=7, tal_topk2=1`; Gewicht o2m(e) = max(1 − e/(E−1), 0)·(0,8 − 0,1) + 0,1, einmal pro Epoche aktualisiert (`trainer.py`), o2o = 1 − o2m; Gesamtloss = o2m·L_o2m + o2o·L_o2o, Log-Items vom o2o-Zweig. Original-YOLOv10 (`E2EDetectLoss`, in 8.4 noch vorhanden, aber nicht genutzt): ungewichtete Summe, o2m topk 10, o2o topk 1. Die Ignore-Loss-Subklasse in Phase 3 setzt auf genau diese `E2ELoss`/`v8DetectionLoss`-Implementierung auf. Achtung: die Gewichtskurve hängt an der Epochenzahl E – bei budgetgleichen Trainings mit unterschiedlichem E dokumentieren. | Nutzervorgabe 3. |
| 2026-10-22 | **Box-Fusions-Tuning v1 (bereits gelaufen) enthielt die Match-Metrik nicht:** GREEDYNMM und NMS mit fester Metrik IOS (SAHI-Default), Schwelle {0,3; 0,5; 0,7} × conf {0,01; 0,1} × Overlap {0,1; 0,2; 0,3} = 18 Konfigurationen; WBF iou_thr {0,3; 0,5; 0,7} × skip_box_thr = conf {0,01; 0,1} × Overlap = 18; NMM fehlte. **Tuning v2** (Jobs `tune2_*`, `sahi_o2m_*`): GREEDYNMM/NMM/NMS × Metrik {IOU, IOS} × Schwelle {0,3 … 0,8} × conf {0,01; 0,1} × Overlap = 72; WBF iou_thr {0,3 … 0,8} × skip_box_thr {0,001; 0,01; 0,05; 0,1} × Overlap = 72 (gleiches Budget je Verfahren). v1-Zeilen bleiben stehen (Suffix ohne `-t2`). | Nutzervorgabe 4. |
| 2026-10-22 | **Aufklärung Kostentabelle:** Die Aussage „~3× zu pessimistisch“ galt nur für das reine Training (B-FULL 640: 0.43 h für 100 Epochen, 15.6 s/Epoche, statt 1,39 h geschätzt). Die Auswertung war dagegen stark unterschätzt: 1.11 h für 36 calib-Auswertungen (2 Zweige) + conf-Tuning + val-Inferenz (≤ 111 s je calib-Auswertung statt ~17 s geschätzt), weil `evaluate()` neben dem Port auch pycocotools (7 Größenbereiche) und Diagnostik rechnete. Job gesamt 1.54 h statt 1,92 h. | `runs/p2/20261022_costs/job_time_breakdown.json`. |
| 2026-10-22 | **Schneller Port-Pfad** (`evaluate(..., port_only=True)`) für Modellauswahl und Tuning: gleiche Port-AP (bitgleich geprüft, 7,3 s statt 36,5 s auf val), keine pycocotools/Diagnostik. Frequenz der calib-Auswertung in Phase 2 unverändert (Vergleichbarkeit); aktiv ab Job `bfull1280_v10` (B-SAHI-FT lief noch mit dem vollen Pfad – nur Laufzeit betroffen). | Nutzervorgabe 6. |
| 2026-10-22 | **calib-Auswertung Phase 3 (geplant, einheitlich für T0–T4 und B-SAHI-FT-Vergleiche):** 10 Auswertungen pro Training, gleichmäßig über das Bildbudget verteilt (alle 10 % der gesehenen Bilder), nur o2o-Zweig für die Auswahl, Port-Pfad. | Nutzervorgabe 6; vor Phase-3-Start bestätigen. |
| 2026-10-22 | Effektiver Zoom pro Bild wird im Phase-2-Report für jede Inferenzvariante relativ zur Trainingsskala ausgewiesen (`effective_zoom_table`); keine Schlussfolgerung SAHI vs. Vollbild vor B-SAHI-FT und B-FULL 1280/1536. | Nutzervorgabe 7. |
| 2026-10-22 | Zoom-Stapel-Kandidaten S1–S4 (`scripts/zoom_candidates.py`, `reports/ZOOM_STACKS_20261022.md`): niedrigste Stufe stets Vollbild-Pass (Eingabe = z × Bild); N gilt nur für gekachelte Stufen. | Annahme zur Auslegung von „{0.64, …} mit N=…“; bei anderer Lesart (gekachelte Stufe 0,64) neu rechnen. |
| 2026-10-22 | **Unterbrechung 20:00 (Nutzerentscheid):** Kette `scahi_chain` beendet (14:02). Deadline-Gate im Runner: ein Job startet nur, wenn jetzt + Schätzung (2. Spalte in `configs/queue/phase2_resume.txt`) vor der Deadline (`runs/queue/DEADLINE`, 19:45) endet; sonst endet die Queue dort (Reihenfolge bleibt, Status `deferred`). Trainings halten am letzten Epochenende, dessen Nachfolger (längste bisherige Epoche inkl. calib-Auswertung + 10 %; vor der letzten Epoche zusätzlich 20 min für die Auswertungen danach) nicht mehr vor der Deadline endet (`EpochEndHalt`). Übergang: alter Runner per STOP nach `bfull1280_v10` beendet, einmalige Warte-Session `scahi_resume_wait` startet danach `scahi queue resume --deadline 19:45`. | Nutzervorgabe 1/4. |
| 2026-10-22 | Fortsetzen nutzt den Ultralytics-Standard-Trainer (keine Trainer-Subklasse in Phase 2) mit unseren Callbacks `CalibSelector` und `EpochEndHalt`; Halt per Exception in `on_model_save` (nach `last.pt`, vor `final_eval`/Optimizer-Strip). Smoke-Test bestanden (`runs/p2/20261022_resume_smoke/summary.json`). **Nicht bitgleich:** Datenreihenfolge und Augmentierungs-Zufall beginnen nach dem Fortsetzen wie in Epoche 1 (RNG-Zustand nicht im Checkpoint; im Smoke-Test sind die Trainingsverluste von Epoche 2 nach Resume identisch mit Epoche 1 bei kaum veränderten Gewichten in der Warmup-Phase). LR-Plan, o2m-Gewicht und Epochenzählung sind exakt. Für die Phase-3-Trainer-Subklasse wird derselbe Test wiederholt. | Nutzervorgabe 4. |
| 2026-10-22 | **Kollaps-Plot (Auslegung):** Recall/Präzision/Kalibrierung (ECE) über σ im Original vs. σ_in = σ·imgsz/max(W,H) im Netzeingang, B-FULL-640-Modell bei 640/1280/1536 (vorhandene val-Detektionen, eigene Zuordnung IoU ≥ 0,5 nach Toolkit-Ignore-Regel, Arbeitspunkt score ≥ 0,25). „Punkt 6 der letzten Nachricht“ lag in diesem Arbeitskontext nicht vor; ausgelegt als Test, ob das Modellverhalten nur von der scheinbaren Größe im Eingang abhängt. Diagnose auf val, keine Parameterableitung. | `scripts/reliability.py`, `reports/RELIABILITY_20261022.md`. |
| 2026-10-23 | **Maschinenwechsel host-a → host-b.** conda-Env `scahi` neu aus `environment.yml` (conda-forge, Python 3.11.17, Octave 10.3.0), torch 2.14.1+cu130/torchvision 0.29.1+cu130 (Index `whl/cu130`), Rest exakt nach `requirements.lock` (Zeilen `@ file://` = conda-Builds der Octave-Abhängigkeiten, kamen über conda), `pip install -e .`. `pip freeze` = `requirements.lock` ohne Abweichung. Export: `env_export/host-b/`. | Treiber host-b 580.178.04 (CUDA 13.0) reicht für cu130; host-a hatte 595.99 (CUDA 13.2). |
| 2026-10-23 | `git config core.fileMode false` (nur lokale Repo-Konfiguration). | Die Kopie von host-a hatte alle Dateien auf 0777 gesetzt (90 reine Modusänderungen) → `git_dirty` wäre dauerhaft true. |
| 2026-10-23 | **Originaldaten auf host-b:** offizielle ZIPs vom Spiegel `<external drive>/Datasets/Datasets/VisDrone-Dataset/Visdrone_2021/` nach `/path/to/datasets/VisDrone/VisDrone2019-DET/` entpackt (gleicher Pfad wie auf host-a → Symlinks in `data/canonical` gültig; lokale SSD statt NTFS/fuse). SHA256 der ZIPs erstmals dokumentiert (`runs/m/20261023_data_check/`). restricted-Kopie (`/path/to/datasets/STORE/<restricted dataset copy> wird auf host-b nicht verwendet. | Prüfung: MD5 aller 8629 Bilder = Phase-0-Audit; Boxzahlen = Audit; Konverter neu: YOLO-Labels/Listen/YAMLs byte-identisch, COCO-JSONs byte-identisch bis auf `info.created`. |
| 2026-10-23 | **Maschinenspezifisches** in gitignorter `configs/local/<maschine>.yaml` (genau eine Datei; Dateistamm = Maschinenname; `data:` überschreibt Schlüssel von `configs/data/visdrone.yaml`). `scahi/machine.py` schreibt `machine` {name, hostname, gpu, driver, cuda_driver, cuda_runtime} in jede `env.json` und `metrics.json`; Leaderboard-Spalte `machine` (Läufe ohne Block = `host-a`). | Nutzervorgabe M3. Latenzen nur innerhalb einer Maschine vergleichen. |
| 2026-10-23 | **Reproduktionsnachweis (Fall A) bestanden:** B-FULL 1280 @1280 o2o und B-SAHI-FT NMS/IOS 0,8 (o0,2, conf 0,01) auf val mit host-a-Gewichten: AP 33,66 / 32,75, ΔAP = 0,000; Detektionen bitgleich. Fortsetzung der Queue ohne weitere Freigabe (Fortsetzungsregel M5). | `runs/m/20261023_repro/summary.json`. |
| 2026-10-23 | **Queue host-b** (`configs/queue/phase2_host_b.txt`): Reihenfolge der Nutzer-Queue, aber `bfull1536_v10` vorgezogen, weil B-MZ, B-SAHI-1280, Degradationstest und Kollaps auf calib erst Code brauchten (GPU sonst ~2 h leer). Alle GPU-Jobs laufen streng nacheinander → Latenzen unbeeinflusst. Neue exp_ids mit Präfix `20261023` (fest, für idempotentes Resume); B-FULL-Jobs behalten `20261022` (Sammelmuster von `reliability.py`). | Planabweichung (Reihenfolge), Begründung Auslastung. |
| 2026-10-23 | **B-MZ** (`scahi/infer/multizoom.py`, Jobs `bmz_s4`, `bmz_s1`): B-FULL-1280-Gewichte; pro Pass Ultralytics-Predictor wie B-FULL (o2o `nms=False`, o2m Ultralytics-NMS 0,7; conf 0,01, max_det 500); niedrigste Stufe = Vollbild mit imgsz = ⌈z·lange Seite/32⌉·32 (S4: 0,64 → 1248 px = effektiv z 0,65 bei 1920 px), höhere Stufen Kacheln T = N/z (`tile_grid`), vom Predictor auf N skaliert. Fusion aller Pässe mit SAHIs `batched_nms` (= `NMSPostprocess`, Test `tests/test_multizoom.py`). Tuning auf calib: Metrik {IOU, IOS} × Schwelle {0,3…0,8} × conf {0,01; 0,1} × Overlap {0,1; 0,2; 0,3} = 72 (= NMS-Budget Tuning v2); Hauptzeile NMS/IOS (Nutzervorgabe), bestes IOU auf calib in `inference.best_iou_metric_on_calib`. Rohdetektionen pro Pass (`raw_calib_*.parquet`, `raw_val.parquet`) für spätere F0-Vergleiche. | Ganzbild-Pass = B-FULL bis auf Batch-Numerik (cuDNN; Batch 6 vs. 1 max. Δscore 0,006; Arrays = Pfade bitgleich). |
| 2026-10-23 | **Degradationstest** (`scripts/degradation.py`): drei Bedingungen bei gleichem σ_in: A nativ (σ ∈ [32, 128] px), B dieselben Objekte nach INTER_AREA ↓d + bilinear ↑d, C „ursprünglich kleine“ Objekte (σ ∈ [32/d, 128/d]) bei Inferenz mit imgsz·d (gleiches σ_in, gleicher Informationsverlust, andere Population). Methodik wie Kollaps-Plot (eigene Zuordnung IoU ≥ 0,5, score ≥ 0,25, Halboktaven, n ≥ 50), val, o2o. | Auslegung von „Vergleich mit ursprünglich kleinen Objekten bei gleichem σ_in“; Kontrolle: A reproduziert die B-FULL-1280-val-Detektionen (max. Δ 3·10⁻⁵). |
| 2026-10-23 | **Kollaps auf calib** (`reliability.py <DATE> calib`, Job `reliability_calib`): calib-Detektionen aller B-FULL-Modelle bei ihren Inferenzgrößen (Predictor, Batch 16). Bandrand a: Recall über σ_in in Viertel-Oktaven relativ zum Plateau (σ_in ∈ [32, 128)), σ_in bei 25/50/75 % des Plateaus. Vorschlag zur Prüfung: weicher Rand = 25–75 %, a = 50-%-Punkt; Festlegung durch den Nutzer. | Nutzervorgabe Queue Punkt 4. |
| 2026-10-23 | Leaderboard-Spalten `run_id` (= exp_id:Protokoll, eindeutig), `eval_data` (original/maskiert), `seed`, `machine`; Brückenexperiment im Phase-2-Report als 2×2 (Training restricted/original × Eval maskiert/original) für visdrone_ignore und plain, je Zweig. Zoom-Planer: Stapel S5 {1,0 (nativ), 2} N=640 (`reports/ZOOM_STACKS_20261023.md`). | Nutzervorgabe Queue Punkte 5 und 8. |
| 2026-10-23 | **Nutzerentscheid Band:** a = 8 px, weicher Rand 6–14 px (≈ 25–75-%-Recall-Punkte auf calib); b = 192 vorerst (nach `reliability_1536` erneut prüfen). Abgeleitet aus B-FULL 640 auf calib (σ_in @ 25/50/75 % = 5,6/8,7/14,3 px, `runs/p2/20261022_reliability_calib/reliability.json`, Zwischenstand vor Job `reliability_calib`). T4 kann abweichen → F3e schätzt eigene Validitätskurven auf calib. | Nutzervorgabe 1. |
| 2026-10-23 | **Degradationstest, Bedingung C** („ursprünglich klein bei gleichem σ_in“) = dasselbe Modell (B-FULL-1280-Gewichte), Originalbild als Vollbild-Pass bei imgsz 1280·d (Zoom d relativ zur Trainings-/Auswerteskala; für d = 2 imgsz 2560), Objekte mit nativem σ ∈ [32/d, 128/d] px, σ_in = σ·imgsz·d/max(W, H) → gleiche σ_in-Bins wie A/B. Die genannte 0,710 stammte aus C_d2 dieser Definition (Lauf während des B-FULL-1536-Trainings, verworfen). Recall und Präzision mit 95-%-Bootstrap-KI über Bilder (1000). Alle Bedingungen werden im Job `degradation_1280` bei freier GPU neu gerechnet. | Nutzervorgabe 2a–c. |
| 2026-10-23 | **Latenzregel:** Latenzen, die gemessen wurden, während ein anderer Rechenprozess die GPU nutzte, sind ungültig. Jede Latenzmessung (B-FULL, SAHI, B-MZ) prüft per `nvidia-smi --query-compute-apps` am Anfang, alle 100 Bilder (außerhalb der Zeitnahme) und am Ende; `metrics.json` → `efficiency.latency_gpu_exclusive` (+ fremde PIDs), Leaderboard-Spalte. Bisher auf host-b gemessene Latenzen: M4-Reproduktion bei freier GPU (vor Queue-Start); Trainingsbenchmark ebenso. `bfull1536_v10` lief mit Code vor dieser Prüfung; seine val-Latenz wird gesondert bewertet (eigene GPU-Nutzung während des Trainings, aber nicht während der Auswertung). Die Laufzeit des 1536-Trainings ist durch meine parallelen Entwicklungstests (bis ~11:00) leicht verlängert → nicht für Kostenkalibrierung verwenden. | Nutzervorgabe 2d. |
| 2026-10-23 | **B-MZ-native = primäre Multi-Zoom-Baseline** (Job `bmz_native`, Stapel `native`): exakt der B-FULL-1280-Pass (imgsz 1280, wie trainiert) + z = 2-Kacheln (T = 512 px, N = 1024), NMS/IOS, Tuning auf calib wie B-MZ, o2o (Hauptzeile) und o2m. Bericht neben B-FULL 1280 als Inkrement „+ z = 2“. S4 {0,64, 2} und S1 sind Nebenzeilen; für die B-FULL-1280-Gewichte (Trainingsskala z ≈ 0,67 bei 1920 px) liegt z = 0,64/0,65 knapp, z = ⅓ deutlich unter der Trainingsskala. Queue nach `bfull1536_v10` per STOP angehalten und mit `bmz_native` vor `bmz_s4` neu gestartet. | Nutzervorgabe 3. |
| 2026-10-23 | **Autorun-Auftrag** (`docs/AUTORUN_BRIEF.md`) übernommen; Vorrang bei Widersprüchen. Ersetzt: (1) „nach jeder Phase anhalten“ (AGENT_BRIEF §1.1) und „Phase 3 erst nach Phase-2-Report und Freigabe“ (Nutzer 23.10. vormittags) → Halt nur bei Gate-Verstoß, Budgetüberschreitung (35 GPU-h ab 23.10. 11:30) oder vor test-dev; (2) „> 2 GPU-h nur mit Freigabe“ → Gesamtbudget; (3) Band-Startwerte a = 12, b = 192 (AGENT_BRIEF §6) → Training T4: Ignore bei σ_in < 6 oder > 256 px, Fusionsfenster a = 8 (6–14), b = 192 (192–256); (4) Stapel S5 im Zoom-Planer (N = 640 auf z = 2) → Phase 4 nutzt S5 mit N = 1024 auf z = 2. | Nutzervorgabe. |
| 2026-10-23 | **GPU-Nutzung während der Queue:** Entwicklungstests (Smoke-Tests Phase 3) laufen kurz parallel zur Queue; davon betroffene Latenzen sind über `latency_gpu_exclusive = false` markiert und werden nicht berichtet. Alle berichteten Latenzen stammen aus dem Latenz-Job am Ende von Phase 5 (exklusive GPU). | Konservativ: Genauigkeitszahlen sind von paralleler Last unabhängig (deterministische Inferenz), Latenzen nicht. |
| 2026-10-23 | **Phase 3 (umgesetzt, `scahi/train/ignore.py`, `scahi/train/p3.py`, `scahi/data/chips_p3.py`):** Ignore-Loss als Subklasse von `v8DetectionLoss`, über `E2ELoss(loss_fn=…)` für beide Zweige; Modellklasse `IgnoreDetectionModel` (eigenes `init_criterion`, weil Ultralytics' Resume das Kriterium neu erzeugt). Datensatz-IDs: 0–9 Objekte, 10 `ignore_all`, 11+c Trunkierungs-Ignore Klasse c. Alle T-Varianten: 20 Epochen × 46 506 Bilder, `scale` = 0,2 (Vorgabe; B-SAHI-FT lief mit dem Default 0,5 → G3c prüft T0 gegen B-SAHI-FT einschließlich dieses Unterschieds), Auswahl wie B-SAHI-FT (10 calib-Auswertungen, SAHI 640/0,2/GREEDYNMM 0,5/conf 0,01, o2o), val-Kontrollzeile mit B-SAHI-FTs getunter NMS/IOS-Konfiguration. T0 = exakt die B-SAHI-FT-Daten. T1: Ignore-Regionen grau (114), Pixel positiver Boxen wiederhergestellt. T2: SAHI-Regel (≥ 10 % sichtbar) + `ignore_all`. T3: positiv ab 50 % sichtbar, darunter Ignore nur der eigenen Klasse. T4: Zoomraster {⅓, 0,47, 0,67, 0,94, 1,33, 1,89, 2,0}, Kachel T = 640/z, Overlap 0,2, lange Seite exakt 640 px (Ganzbild bei z·Seite < 640 mit 114 aufgefüllt, damit Ultralytics nicht umskaliert), **eine feste Stichprobe pro Epoche** (46 506 Chips, 90 % Objekt-Chips gleichverteilt über die Zoomstufen, 10 % Hintergrund; fehlende Kandidaten einer Stufe auf die übrigen verteilt; `p3_T4/sampling.json`) statt Neuziehung pro Epoche – gleiche Epochenstruktur wie T0–T3 (konservativ). | Gate G3a: `tests/test_ignore_loss.py` grün; Augmentierungskontrolle `reports/figures/aug_check/` plausibel (Sichtprüfung). |
| 2026-10-23 | **Phase 4/5 (Entwurf):** Cache pro Zoomstufe (`scahi/infer/cache_mz.py`), Eingabegeometrie wie der B-MZ-Predictor; F0 aus dem Cache: o2o = `topk_like_ultralytics`, o2m = Ultralytics' `non_max_suppression` je Pass (Abgleich mit dem Predictor: Anzahl ±5, Δscore ≤ 0,008 – gefuste vs. ungefuste Gewichte). Skalenraum-Fusion `scahi/fusion/scalespace.py` (Operationalisierung im Modul-Docstring): Score eines Peaks = Σ über Pässe des passweisen Maximums von w·p_c in der Nachbarschaft / C; Mitglieder nur mit IoU ≥ 0,5 zur stärksten Vorhersage der Nachbarschaft; bis zu 3 Klassen pro Peak. **G5b wird mit F2 geprüft** (F3 hat bei nur einem Pass per Konstruktion Lücken außerhalb des Bands). | **Verfahrensfehler, korrigiert:** Ein erster Entwicklungsvergleich (Cluster ja/nein) lief versehentlich auf 60 val-Bildern; die Entscheidung wurde auf 60 calib-Bildern wiederholt und dort getroffen (F2 Einzelpass o2m: −1,32 AP ohne, −0,47 AP mit Cluster gegenüber o2m+NMS). Ab jetzt Entwicklung nur auf calib. |
| 2026-10-23 | **Zwei Queues:** GPU-Queue (`configs/queue/phase2to5_host_b.txt`) und parallele CPU-Queue für die Phase-5-Fusion (`configs/queue/phase5_cpu.txt`, tmux `scahi_queue_cpu`, Status `reports/QUEUE_STATUS_CPU.md`); STOP/HALT gelten für beide. Phase-4-Cache der B-FULL-1280-Gewichte vorgezogen (direkt nach `bmz_native`, braucht dessen Overlap). **Latenzen:** Während die CPU-Queue rechnet, sind Latenzen der GPU-Jobs CPU-seitig gestört (von `latency_gpu_exclusive` nicht erfasst) → alle berichteten Latenzen stammen ausschließlich aus dem Latenz-Job am Ende von Phase 5, der ohne parallele Queue läuft. | Wanduhrzeit; konservativ bei Latenzen. |
| 2026-10-23 | **Latenz-Exklusivität auch CPU-seitig (Nutzervorgabe):** jede Latenzmessung prüft zusätzlich, ob die CPU-Queue einen Job ausführt und ob fremde Prozesse > 1 Kern Last erzeugen (`scahi.machine.check_exclusive`); Felder `latency_cpu_exclusive`, `latency_status` (final nur bei GPU- und CPU-Exklusivität, sonst informational), Leaderboard-Spalten. Vor dieser Prüfung gemessene host-b-Latenzen per `scripts/tag_latency.py` nachträglich markiert (während CPU-Queue-Jobs: `false`, sonst `null`; alle „informational“). Der finale Latenz-Job (`p5_latency`) startet erst, wenn der CPU-Queue-Runner beendet ist und eine Exklusivitätsprüfung sauber ist. | Nutzervorgabe 2 (23.10. nachmittags). |
| 2026-10-23 | **Präregistrierung und val-Fusionszahlen (Offenlegung):** `docs/PREREGISTRATION.md` ist seit Commit d067729c (11:53:57) unverändert. Zu diesem Zeitpunkt existierten **keine** val-Ergebnisse von Phase 3–5-Methoden; **es existierten aber Entwicklungszahlen der dichten Fusion auf 60 val-Bildern** (Einzelpass w1280, B-FULL-1280-Gewichte, F1/F2/F3, ca. 11:45; F2 31,6 / nach Cluster-Änderung 32,8 gegen o2m+NMS 33,7), die ich gesehen habe – siehe Eintrag „Verfahrensfehler, korrigiert“. Sie betreffen weder T4 noch einen Stapel und flossen in keine präregistrierte Festlegung ein; die Entscheidung zur Cluster-Regel wurde auf calib getroffen. Seit dem Präregistrierungs-Commit: G5a (Geometrie), G5b (calib), noch keine val-Zeile der dichten Fusion. | Nutzervorgabe 1. |
| 2026-10-23 | **p5_fuse_BFULL1280 hing 75 min** (alle 8 Worker bei 0 % CPU, Fork-Deadlock: der Elternprozess hatte für G5b torch benutzt). Job abgebrochen (Status `failed`, Versuch 1), Worker-Pools der Phase-5-Jobs nutzen jetzt `spawn`; CPU-Queue neu gestartet (14:52), GPU-Queue lief ununterbrochen weiter. G5b (calib) war bereits bestanden: F2-Einzelpass 30,24 gegen o2m+NMS 30,41 (−0,17 AP). | Fehlerbehebung; kein Einfluss auf Ergebnisse. |
| 2026-10-23 | **Boxen auf das Bild geclippt (alle Fusionsvarianten):** F0 aus dem Cache nutzte ungeclippte Pass-Boxen (GREEDYNMM scheiterte an SAHIs Prüfung auf nichtnegative Koordinaten); der Ultralytics-Predictor (B-MZ) und SAHI clippen. Jetzt: Pass-Detektionen und Endausgaben aller Varianten (F0–F5) auf [0, W] × [0, H] geclippt (`run.pass_dets`, `run.clip_xywh`). Die bis dahin gelaufene B-FULL-1280-Fusion (19 von 90 Tuning-Aufgaben, Kalibrierung, G5b) wurde verworfen (archiviert unter `runs/dev/p5_unclipped/`, nicht ausgewertet) und vollständig neu gerechnet. | Konsistenz mit dem Predictor; keine Mischung zweier Zustände. |
| 2026-10-23 | **Seed-Streuung und Rauschschwelle:** B-FULL 640 (o2o @ 640, val, Toolkit-Port) Seeds 0/1/2: 22,861 / 22,627 / 22,683 AP → σ_seed = 0,122 AP (Stichproben-SD, ddof 1) → **ε = max(0,3; 2·σ_seed) = 0,300 AP**. Folgen: G3c-Toleranz = max(1,0; 3·σ_seed) = 1,0 AP; Seeds Phase 3: 1 Seed für T0/T1/T3, 3 Seeds für T2/T4 (σ_seed ≤ 0,3). In der Präregistrierung als Ergänzung eingetragen (Commit b4340c5, automatisch vor dem ersten Phase-3-Training). | AUTORUN_BRIEF 1.2. |
| 2026-10-23 | **Nutzerentscheide nach Phase 2 (B-FULL 1536 35,75 > R 35,00):** (1) Präregistrierungs-Nachtrag „zweite Referenz R2 = stärkste Baseline“, Kriterien (a)–(c) gegen R und R2 (Commit c8b8aff9, 17:13; bis dahin keine val-Fusionszeile eines T-Modells; val-Fusionszeilen nur für B-FULL-1280-Gewichte, nicht eingesehen). (2) Neue Referenzzeilen in der GPU-Queue nach T0/T2/T4 und deren Caches, vor T1/T3 und Zusatz-Seeds: `bmz_native1536` (B-FULL-1536-Pass + z = 2, N = 1024, NMS/IOS auf calib, o2o/o2m), `bfull2048_v10` (gleiches Rezept wie 1536, **Batch 3** aus Speichergründen: 1536 lief mit Batch 6 bei bis zu 23 GB), `reliability_2048` (Kollaps val + calib, Degradation). Budgetprognose vor der Aufnahme: 5,7 GPU-h verbraucht (seit 11:30), Rest ≈ 18,2 h inkl. 2048 (≈ 5 h, Schätzung 2,66 h × 1,78 Pixel) → ≈ 23,9 von 35 GPU-h → 2048 aufgenommen. (3) Phase-2-Report um Größenbin-Vergleich 1280/1536/B-MZ-native/B-MZ-1536-native, Einzelpass-Auflösungskurve und kompakte Degradationstabelle erweitert. (4) Cache und Fusion auch für B-FULL-1536-Gewichte (Stapel native1536, S5; Cache-Overlap wie alle = 0,3). | Nutzervorgaben 1–4. |
| 2026-10-23 | **Queue-Runner liest die Queue-Datei vor jedem Job neu** (Einfügen/Umordnen ohne Neustart); zusätzlich queue-spezifische Stopp-Dateien `STOP_GPU`/`STOP_CPU` (geteiltes `STOP`/`HALT` bleibt). Einmaliger Neustart beider Queues nach T0 bzw. nach der B-FULL-1280-Fusion. | Betriebssicherheit. |
| 2026-10-23 | **Gate G4 gerissen (P3T2_s0, 19:36):** G4a-Parität max. |Δ Box| = 1,2207·10⁻⁴ px > Toleranz 1·10⁻⁴ (Score-Δ 0, Klassen 100 % gleich, Cache vollständig). Diagnose (`runs/p4/20261023_P3T2_s0_cache/gate_G4.json` + Nachmessung): die Abweichung ist an der Maximalstelle genau **1 float32-ulp** der Koordinate (1094,7 px → ulp 2⁻¹³ = 1,22·10⁻⁴); bei 640-px-Eingang (Original-Phase-4-Test) 6,1·10⁻⁵ px → bestanden; B-FULL 1280 bestand nur, weil sein Maximum bei 804 px lag (ulp 6,1·10⁻⁵). Kein Dekodierfehler, sondern eine absolute Toleranz unterhalb der float32-Auflösung für Koordinaten ≥ 1024 px. **Halt von Phase 4/5 bis zum Nutzerentscheid** (CPU-Queue gestoppt, `STOP_CPU`); vorgeschlagen: G4a-Toleranz = max(1·10⁻⁴ px, 1 ulp der Koordinate) oder Paritätstest bei 640 px Eingang wie in Phase 4. Bis dahin laufen nur Cache-unabhängige GPU-Jobs weiter (`configs/queue/phase3_g4hold.txt`: B-MZ-1536-native, B-FULL 2048, reliability_2048, T1, T3, Zusatz-Seeds) – Phase-3-Trainings und Referenzzeilen hängen nicht vom Cache ab. | AUTORUN_BRIEF 4 (G4); konservativ: kein Weiterlaufen der Gate-abhängigen Schritte. |
| 2026-10-24 | **Gate-Korrektur G4a (Nutzerentscheid Option 1):** Paritätstoleranz je Box-Koordinate = max(1·10⁻⁴ px, 1 float32-ulp der Koordinate) – ein absolutes 1·10⁻⁴ liegt für Koordinaten ≥ 1024 px unter der float32-Auflösung. Geprüft an allen Eingangsgeometrien von Phase 4/5 (Ganzbild lange Seite 640/1248/1280/1536/1920, Kacheln 640 und 1024; 2048, sobald gecacht) für T2, T4 (Seed 0), B-FULL 1280, B-FULL 1536: **alle bestanden**, maximale Abweichung überall genau 1,00 ulp (6,1·10⁻⁵ px unter 1024 px, 1,22·10⁻⁴ px darüber), Score-Δ 0, Klassen 100 % (`runs/p4/20261023_parity_g4a_v2.json`; Gate-Datei von B-FULL 1280 um die neue Prüfung ergänzt, alte Prüfung erhalten). Keine Änderung der Präregistrierung; bestehende Caches bleiben gültig. `STOP_CPU` wird entfernt, sobald die Cache-Jobs T2, T4, B-FULL 1536 mit der neuen Regel bestanden haben. | Gate-Fix wegen float32-Auflösung. |
| 2026-10-24 | **Finale Latenzen:** Störungen der CPU-Exklusivität durch die IDE (VS Code + Git-Integration, Spitzen bis ≈ 2,5 Kerne bei Dateiänderungen im Repo) und durch eine parallele Analyse (`phase6.py`) markierten drei Ergänzungszeilen zunächst als informational. Regel unverändert; stattdessen pro Zeile Reset der Exklusivitätsstatistik und bis zu 3 Messversuche, Messung ohne Repo-Schreibzugriffe. Ergebnis: alle 15 Latenzzeilen `final` (`runs/p5/20261023_latency/latency.json`). | Saubere Messung ohne Lockerung der Regel. |
| 2026-10-24 | **Phase 6 abgeschlossen** (`reports/EXCHANGE_VAL_FINAL_20261024.md`, `docs/FREEZE_PROPOSAL.md`): Hauptkriterium (a)/(b) gegen R und R2 nicht erfüllt ((a) +0,92 [+0,52; +1,31] < +1,0), (c) nicht prüfbar; H1 nicht erfüllt, H2 erfüllt (+1,10 [+0,84; +1,37]), H3 o2m > o2o. Empfehlung Analyse-Paper. **Halt vor test-dev** (AUTORUN_BRIEF 6.8). Optionale Phase 7 (person_vehicle T2/T4) nicht gestartet. | AUTORUN_BRIEF. |
| 2026-10-24 | **Nutzerentscheide nach Phase 6:** (1) Präregistriertes Ergebnis exakt wie berechnet berichten, keine Nachjustierung. (2) Paper-Form: Analyse-Paper mit dichter Skalenraum-Fusion als positivem Methodenbaustein (`docs/FREEZE_PROPOSAL.md` §5). (3) **Explorative Zeilen** (Kennzeichnung „exploratory, defined after val results“, eigene Tabelle, zählen nie für die Präregistrierung): B-FULL-2048-Pass + z = 2 (512 px, N = 1024) und B-FULL-1536-Pass + z = 2, je F0-NMS (o2o) und beste dichte Variante F2 (o2m), Tuning nur auf calib; Cache B-FULL 2048 mit G4 (Parität inkl. 2048-px-Eingang) und G5b (−0,85 AP) bestanden; Budgetprognose vorab ≈ 0,4 GPU-h. Ergebnis: 2048 + z = 2 F2 39,34 AP, +0,87 [+0,62; +1,09] über den 2048-Einzelpass (`runs/p6/20261024_exploratory/explore.json`). (4) Pareto (AP vs. Latenz/GFLOPs): keine präregistrierte Multi-Pass-Zeile Pareto-optimal; nur die explorativen 2048 + z = 2-Zeilen liegen über der Einzelpass-Kurve (≈ 4,5× GFLOPs). (5) H1-Diagnostik deskriptiv (`runs/p6/20261024_h1_diagnostic.json`, `scripts/h1_diagnostic.py`). | Nutzervorgaben 1–6. |
| 2026-10-24 | **Plan B (volle Ressourcen):** Fusion B-FULL 2048 mit 20 Workern (RAM-Spitze < 100 GB), danach `p6_explore_latency` exklusiv (alle Zeilen final). YIELD-Mechanismus nicht benötigt (kein paralleler Fremdjob). **GPU frei seit 24.10. 13:48:44.** GPU-Budget gesamt ≈ 22,8 von 35 GPU-h. | Nutzervorgabe. |
| 2026-10-24 | **Abbildungen und Tabellen** (`scripts/paper_figures.py`, `scripts/paper_tables.py`, `reports/figures/paper/`): alle Zahlen aus `runs/` generiert; TikZ-Methodenübersicht `tex/fig_method.tex`; Sammeldokument `tex/paper_figures.pdf` (LuaLaTeX, Inter + Fira Math). Farbslots fest (blau #2a78d6 Einzelpass/Referenz, orange #eb6834 Box-Fusion, aqua #1baf7a dichte Fusion; Palette validiert). Explorative Zeilen in Abbildungen offen markiert. | Nutzervorgabe Visualisierung. |
| 2026-10-24 | **Freeze approved** (user): `configs/final/frozen_rows.json` (36 rows, generated by `scripts/freeze_final.py` from the val artifacts, weights SHA-256) + `configs/final/README.md` (provenance per row incl. „selected on val“ for R2's role and for the X1–X4 row definition / o2m restriction; seeds as mean ± std, preregistered seed rule unchanged; reporting rules fixed before the run; val latencies). Run code `scahi/testdev.py`; gate G7 = val replay of the test-dev code path (|ΔAP| ≤ 1e-6) before any test-dev inference. **person_vehicle (Phase 7) not pursued for now; remaining budget (≈ 12 GPU-h) kept as reserve.** | User conditions 1–5. |
| 2026-10-24 | **test-dev abgeschlossen** (`reports/EXCHANGE_TESTDEV_20261024.md`): G7 (val-Replay, 29 Zeilen, max \|ΔAP\| = 0) bestanden, alle 36 eingefrorenen Zeilen auf 1610 Bildern. Präregistrierung auf test-dev exakt wie auf val: (a) erfüllt (+1,81 [+1,52; +2,10]; Seed-Mittel +1,73), (b) nicht erfüllt, (c) nicht prüfbar, R2-Kriterien nicht erfüllt, H1 nicht erfüllt (+0,286 < ε), H2 erfüllt, H3 o2m > o2o. Val-Urteile unverändert; (a) ist der einzige abweichende Punkt. Zusätzlich gemessen (Bedingung 4): Fusionsanteil an der E2E-Latenz auf val (`td_latency_split`, alle final). Analyse-Bootstrap: Metrikvektoren je Zeile einmal über dieselben 1000 Resampling-Gewichte (identisch zu `bootstrap.paired`, val-Werte reproduziert). Fehler unterwegs: B-MZ-Stapelname falsch geparst (erster G7-Versuch abgebrochen, vor jedem test-dev-Zugriff), `start_queue.sh` traf per tmux-Präfix die CPU-Session (exakter Match). GPU gesamt ≈ 24,2 von 35 GPU-h. | Freeze-Bedingungen 1–5. |
| 2026-10-26 | **Erweiterungsstudie E** (`docs/EXTENSION_BRIEF.md`) übernommen; Vorrang bei Widersprüchen. **E0:** Tag `paper-draft-v1` = c1a81ca (lokal, kein Remote); `exports/paper_draft_v1/` (Appendix-PDF – ein separates Paper-PDF existiert im Projekt nicht –, alle Abbildungen inkl. LaTeX-Quellen, Austauscharchiv; nicht versioniert). Zentrale Stildatei `scripts/figstyle.py` (Farben, Schriften, Größen, dpi; schreibt `tex/figstyle.tex` für die TikZ-Abbildung), Einstieg `scripts/make_figures.py`; neben jeder Abbildung `<name>.data.json` (geplottete Werte, Eingabedateien, bei Bildern Ausschnitt und Detektionen). Test `make_figures.py --check`: alle Abbildungen gelöscht und neu erzeugt → alle 14 PNG pixelidentisch. **Replay-Gate E0** (`runs/ext/e0_replay/gate_E0.json`): val-AP von B-FULL 1280, B-FULL 2048, R, M_ab mit aktuellem Code exakt reproduziert (ΔAP = 0,000). | Brief §1. |
| 2026-10-26 | **Modelle in ultralytics 8.4.183:** COCO-Gewichte für YOLOv10m, YOLO11s/m, YOLO12s/m vorhanden (Release v8.4.0, per `attempt_download_asset` nach `weights/`; SHA-256 in `runs/ext/heads.json`). **Kopfstruktur verifiziert:** YOLOv10 = `v10Detect` mit o2m (`cv2`/`cv3`) und o2o (`one2one_cv2`/`one2one_cv3`); YOLO11 und YOLO12 = `Detect` mit einem dichten Kopf (`cv2`/`cv3`/`dfl`, reg_max 16, Strides 8/16/32), strukturell identisch zum o2m-Kopf von YOLOv10 → wird im Code als Zweig „o2m“ geführt (Dekodierung + NMS). | Brief §2. |
| 2026-10-26 | **Kostenprognose E1** (`runs/ext/cost_forecast.json`, Smoke-Test `scripts/ext_smoke_cost.py`): ≈ 70,8 GPU-h > freigegebene 60 → **Halt** (Brief §5/§10), Kürzungsoptionen in `reports/EXCHANGE_EXT_E0_20261026.md`. Hinweis zur Kalibrierung: die `time`-Spalte von `results.csv` beginnt bei Resume neu; maßgeblich sind die Wanduhrzeiten aus dem Queue-Status (YOLOv10s 2048: 5,02 h auf host-b). | Brief §5. |
| 2026-10-26 | **Nutzerentscheid Budget:** Erweiterung auf ≈ 75 GPU-h erhöht, Plan unverändert (5 Modelle × {1280, 2048} + 2 Zusatz-Seeds YOLO11s 2048). **Präregistrierung der Erweiterung** `docs/PREREGISTRATION_EXT.md` = Commit **e5eca5bd** (vor dem ersten Training). Festlegungen dort u. a.: YOLOv10m E-H2 im selben Zweig (o2m gegen o2m), E-MZ-NMS mit fester Metrik IOS und Overlap 0,3 (12er-Raster), F2-Raster wie bisher (24), Fenster a/b wie bisher, OOM-Regel (einmalige Batch-Reduktion 8→6 / 3→2). Code: `CalibSelector(selection_branch=…)` (Default o2o unverändert; Ein-Kopf-Modelle wählen über ihren einen Kopf), Jobs `scahi/ext.py`, Queue `configs/queue/ext_e1.txt`. | Nutzerentscheid; Brief §4/§5. |
| 2026-10-26 | **G-E1 definiert und vorab getestet** (COCO-Gewichte aller neuen Modelle): (1) eigene DFL-Dekodierung des dichten Kopfes gegen Ultralytics' Funktionen (`dist2bbox`, `make_anchors`, `Detect.dfl`) auf demselben Vorwärtslauf, Ganzbild bei der Trainingsgröße und z = 2-Kachel 1024; (2) YOLOv10m zusätzlich o2o-Parität wie G4a; (3) eigener Cache-Pfad + NMS gegen den Ultralytics-Predictor (gefuste Gewichte → ≥ 98 % Treffer bei IoU ≥ 0,99, Δscore ≤ 0,01). Kein strikter NMS-gegen-NMS-Vergleich, weil Ultralytics' NMS ein Wanduhr-Zeitlimit hat (Warnung „NMS time limit exceeded“; in allen bisherigen Queue-Logs nie aufgetreten, wird in E-Logs überwacht). Befund: 2 ulp durch DFL-Summationsreihenfolge (+ TF32 in Ultralytics' conv1d) → **Nutzerentscheid: Toleranz 2 ulp**, TF32 aus für die Prüfung (Präregistrierungs-Ergänzung). | Gate-Klärung vor Anwendung. |
| 2026-10-26 | **E3 (Datensatz-Audit)** `reports/EXT_DATASET_OPTIONS_20261026.md`: xView und DOTA v1.5 liegen nicht lokal vor (Spiegel-Laufwerk nicht eingehängt); Audit aus offiziellen Quellen, Verteilungen (Größenbins, lange Seite, Klassen) offen bis zum Download; kein Download eigenmächtig (xView-Registrierung, Lizenz nichtkommerziell). **Auslegung „Halt nach E3“:** Halt des Datensatz-Strangs bis zur Wahl durch den Nutzer; die bereits freigegebenen E1-Trainings auf VisDrone laufen weiter (unabhängig von der Datensatzwahl, Budget 75 GPU-h freigegeben). | Brief §7/§10; konservativ bzgl. Daten, keine Unterbrechung freigegebener GPU-Arbeit. |
| 2026-10-26 | **Nutzerentscheide zu den Designnotizen** → Präregistrierungs-Ergänzung **0f6fecbc** (vor jedem E-H3-Ergebnis): (1) R3 = B-FULL 2560 für YOLOv10s (3072 nur, wenn Batch 3 passt; Entscheidung automatisch im Job `e_smoke_r3` → `runs/ext/cuts.json`), neben E-H3 berichtet, ohne Einfluss auf das Urteil; (2) YOLO12m 2048 gestrichen, feste Kürzungsreihenfolge bei Prognose > 75 GPU-h (12m 2048 → YOLO11s-Seeds → 11m 2048), automatisch in `e_smoke_r3` angewendet, Jobs prüfen `cuts.json`; (3) effektiver Batch je Lauf in `training.json` (nbs 64, accumulate = round(64/batch)); (4) YOLOv10m Einzelpass o2m+NMS als deskriptive Zeile markiert; (5) G-E1 = ≤ 4 ulp **und** calib-Toolkit-AP aus eigener vs. Ultralytics-Dekodierung identisch auf 4 Dezimalen (entscheidend), Predictor-Vergleich nur berichtet; o2o von YOLOv10m direkt gegen Ultralytics' End-to-End-Ausgabe mit derselben Regel; (6) Latenz: Spalten Referenz / optimiert, optimierter Pfad für jede Zeile mit gefustem Modell und float32 (Umsetzung im Latenzjob); (7) Holm-adjustierte Urteile im Anhang, gepoolter Effekt nur deskriptiv; (8) E-H3 auf VisDrone-val nur teilweise unabhängig – Bestätigung auf dem zweiten Datensatz. | Nutzerentscheid. |
| 2026-10-26 | **Pipeline-Probelauf** (COCO-Gewichte, 20 Bilder): GPU-Teil YOLO11s vollständig (G-E1 bestanden: 0,61 der Toleranz, calib-AP eigene = Ultralytics-Dekodierung, E-1280/E-2048, E-SAHI, Caches); der Fusionsteil brach im Queue-Job an den 80 COCO-Klassen ab (Fusion ist auf 10 Klassen gebaut – Artefakt des Probelaufs) und lief danach auf der CPU mit auf 10 Klassen beschnittenen Wahrscheinlichkeiten vollständig durch (G-E3, Tuning NMS-IOS/F2, alle val-Zeilen). YOLOv10m-GPU-Teil (o2o-Zweig von G-E1) wird erst im echten Job `e_gpu_yolov10m` erprobt (GPU belegt). **R3-Smoke:** 2560 und 3072 passen mit Batch 3 (22,5 / 26,7 GB) → 3072 läuft. Die Smoke-Zeit für 3072 war nicht plausibel (schneller pro Iteration als 2048) → Prognose konservativ = max(Smoke, Pixelskalierung der gemessenen 2048-Wanduhrzeit): 2560 8,4 h, 3072 11,3 h → Gesamtprognose 79,2 GPU-h > 75 → **feste Kürzungsreihenfolge angewendet: YOLO11s-Zusatz-Seeds entfallen** (Prognose 71,4 GPU-h); damit ε = 0,30 ohne σ_seed-Prüfung (`runs/ext/cuts.json`). | Ergänzung 0f6fecbc, Punkte 1–2. |
| 2026-10-26 | **Nachtpause (Nutzer):** `runs/queue/STOP_GPU` gesetzt – das laufende Training YOLOv10m 2048 läuft zu Ende (≈ 20:30), danach beendet sich die GPU-Queue; GPU-lastige Auswertung (`e_gpu_yolov10m` usw.) wird erst auf Kommando fortgesetzt (`rm runs/queue/STOP_GPU && scripts/start_queue.sh configs/queue/ext_e1.txt`). Die CPU-Queue wartet nur auf den Cache (keine Last). Beobachtung: Bei jeder calib-Auswertung während des 2048-Trainings meldet der CUDA-Allocator „allocation failed with OOM … 3,2 GB“ (Warnung, Cache wird freigegeben und neu versucht); alle Auswertungen liefen durch, die calib-Kurve ist lückenlos. | Nutzerwunsch. |
| 2026-10-26 | **GPU-Leistungslimit (Nutzerregel):** aktives und Standard-Limit werden bei jedem Schreiben frisch abgefragt (`scahi.machine.gpu_power`) und stehen in jedem `metrics.json` (machine-Block), `env.json`, jeder calib-Auswertung des Trainings (`gpu_power_limit_w`) und als Leaderboard-Spalte. Latenzen sind nur „final“, wenn das Standardlimit aktiv ist (`latency_validity`); Latenzjobs prüfen `require_default_power` und werden unter Limit zurückgestellt (Exit 77, Datei `runs/queue/LATENCY_DEFERRED`, Hinweis in QUEUE_STATUS). Training und Genauigkeitsläufe dürfen unter Limit laufen; ETA aus gemessenem Durchsatz: `scripts/ext_eta.py`. ε bleibt 0,30 (YOLOv10s-Seeds der abgeschlossenen Studie) – im Bericht als Einschränkung zu nennen. | Nutzerregel 1–3. |
| 2026-10-26 | **Tag-/Nachtmodus GPU (Nutzer):** nachts (≈ 20–8 Uhr) Leistungslimit ≈ 400 W, tagsüber Standard 575 W, bei Lärm Rückmeldung des Nutzers. Umschalten durch den Nutzer (sudo nötig, z. B. root-Crontab); der Agent hat keine sudo-Rechte. Wirkung nur auf Laufzeit; Limit wird je Zeile/calib-Auswertung protokolliert; Latenzjobs nur bei Standardlimit (sonst zurückgestellt). | Nutzer. |
| 2026-10-26 | **Stolperfalle „wartender Job mit altem Code“:** `e_fuse_yolov10m` startete 11:02 und wartete ~10 h auf den Cache; der Hauptprozess lief mit dem Code von 11:02, die erst um 21:32 gestarteten Worker (spawn) mit dem neuen Code (val-Zeilen mit GFLOPs-Argument) → Abbruch der val-Zeilen („expected 7, got 6“). calib-Tuning und G-E3 waren vollständig und gültig (unveränderter Code). Val-Zeilen manuell mit aktuellem Code nachgeholt (Tuning übernommen). Regel: Code-Änderungen an `scahi/ext.py` während wartender Jobs vermeiden bzw. wartende Jobs danach neu starten. | Fehlerbehebung, ohne Einfluss auf Ergebnisse. |
| 2026-10-26 | **Nachtstopp (Nutzer, 23:07):** `HALT` + `STOP` gesetzt: YOLO11s 2048 sauber am Epochenende angehalten (Fortsetzen aus `last.pt`), CPU-Queue beendet, manuelle val-Zeilen YOLOv10m abgebrochen (werden mit `e_fuse_yolov10m` nachgeholt; Tuning liegt vor). Fortsetzen: `rm runs/queue/HALT runs/queue/STOP && scripts/start_queue.sh configs/queue/ext_e1.txt && SCAHI_QUEUE_TAG=_CPU scripts/start_queue.sh configs/queue/ext_e1_cpu.txt` (CPU-Queue: `e_fuse_yolov10m` und `e_fuse_yolo11s` stehen auf failed und werden im zweiten Durchlauf bzw. per Neustart wiederholt). | Nutzer. |

- 2026-10-27 06:40: Queues nach Nachtpause fortgesetzt (Power-Limit 575 W = Standard). e_dryrun aus ext_e1.txt auskommentiert: scheitert nur am bekannten COCO-80-Artefakt des Probelaufs; die echten 10-Klassen-Modelle sind über G-E1/G-E3 verifiziert. e_fuse_yolov10m läuft neu mit aktuellem Code.

- 2026-10-27 ~14:30: **Explorative Zerlegung von E-H3 (post hoc)**, auf Wunsch des Nutzers nach Sicht der Zwischenergebnisse definiert; ändert kein E-H-Urteil, keine neuen GPU-Läufe. Je Modell und s ∈ {1280, 2048} auf val: (A) Einzelpass mit Standard-Nachbearbeitung (= E-s-Zeile; YOLOv10m o2o), (B) Einzelpass + F2 nur auf der Ganzbild-Ebene des Caches (Zweig o2m wie die DENSE-Zeilen; YOLOv10m zusätzlich o2o), (C) = E-MZ-DENSE-s. Deltas B−A („bessere Deduplizierung“) und C−B („Kacheln“) mit gepaartem Bootstrap (1000, Seed 0); dup/frag-Raten, AP je Bin <8/8–16/16–32 px. Festlegung von mir: (B) nutzt die calib-getunten F2-Parameter von (C) im selben Zweig (kein eigenes Tuning), damit sich B und C nur durch die Kacheln unterscheiden. (B)-Zeilen liegen in `runs/ext/decomp/` (nicht in `rows/`). Skript `scripts/ext_decomp.py`, Ausgabe `runs/ext/decomp.json`; CPU-Jobs `e_decomp_<model>` nach jeder Fusion. Jede Tabelle trägt das Label „exploratory decomposition (post hoc)“.

- 2026-10-27 ~15:30: **Zwei weitere explorative Diagnostiken (post hoc)** auf Nutzerwunsch, nur CPU, bestehende Caches, kein Einfluss auf E-H-Urteile: (1) AP50/AP75 (Toolkit-Port) für A, B, C mit gepaarten KIs für B−A und C−B (`bootstrap.port_ap` hat dafür `iou_index` bekommen; Standard unverändert, gegen metrics.json exakt geprüft). (2) Box-Quellen-Ablation Bt/Ct: Box des stärksten Cluster-Mitglieds (max w·p_max, das Mitglied, um das der Cluster gebildet wird) statt des gewichteten Mittels; Scores und Peaks unverändert (`scalespace.fuse_image(box_source="top")`, Standard „mean“ unverändert). Läuft im Job `e_decomp_<model>` mit.
  **Ergebnis (YOLOv10m, YOLO11s):** die Best-Member-Box ist überall schlechter (AP Bt−B −0,06…−0,23, Ct−C −0,49…−1,07, alle KI < 0); das gewichtete Mittel ist also **nicht** die Ursache des B-Verlusts. Die Bedingung für den Eintrag „hybrid fusion (dense scores, best-member box)“ als Kandidat für den zweiten Datensatz ist **nicht erfüllt**; nicht eingetragen. Wird nach YOLO12s/YOLO11m/YOLO12m erneut geprüft.

- 2026-10-27 ~15:30: **YOLO12s 2048: OOM, Lauf neu gestartet.** Ablauf: Batch 3 OOM im ersten Schritt → das eingebaute Auto-Retry von Ultralytics 8.4 (`trainer._oom_retries`) halbierte **ohne Rückmeldung** auf Batch 1 und trainierte ~55 min weiter (nicht präregistriert); bei der calib-Auswertung (predict, Batch 16, 2048 px, Attention: 16 GiB Anforderung) OOM → unser präregistrierter Fallback 3 → 2 startete **im selben Prozess** (20 GB noch belegt) → erneut OOM → Job failed. Kein Ergebnis aus diesem Lauf wird verwendet (Ordner gelöscht). Infrastruktur-Korrekturen (Rezept unverändert): (1) `FixedBatchGuard` (`on_train_batch_end`) bricht ab, sobald Ultralytics die Batch-Größe selbst ändert → präregistrierter Fallback statt stiller Reduktion; (2) der Fallback startet den Job in einem **frischen Prozess** (`os.execv`); (3) `predict_full` halbiert bei OOM die Inferenz-Batch-Größe für den Rest des Aufrufs (betrifft nur Läufe mit OOM; alle bisherigen Läufe unverändert; bei YOLO12 kann die calib-Auswahl dadurch mit kleineren Predict-Batches laufen – Letterbox-Padding je Batch kann minimal abweichen, dokumentiert). Neustart als `e_retry_train_yolo12s_2048` mit Batch 2 (OOM_FALLBACK, präregistriert) + `e_retry_gpu_yolo12s` direkt nach YOLO11m 1280, damit die CPU-Fusion nicht bis Donnerstag blockiert. Verlust ≈ 0,9 GPU-h. YOLO12s 2048 läuft damit mit effektiver Batch 64 (2 × 32), als Rezeptabweichung zu berichten.

- 2026-10-27 ~16:00: **E3 vorgezogen** (Nutzerauftrag): Audit xView vs. DOTA v1.5 in `reports/EXT_DATASET_OPTIONS_20261027.md`; Empfehlung xView; Halt für beide Datensätze (Registrierung bzw. manueller Download durch den Nutzer). GPU-Smoke `e_e3_smoke` (synthetisch, ≤ 15 min) in der GPU-Queue nach YOLO11m 1280. Hybrid-Zeile nicht vorgeschlagen (Diagnostik negativ).

- 2026-10-27 ~16:15: **Nutzerentscheid Priorität und zweiter Datensatz.**
  - Reihenfolge: (1) laufende Erweiterungs-Queue (YOLO12s 2048 Retry, YOLO11m, YOLO12m 1280, R3 2560/3072); (2) E2 (faire Latenz) und E4 (val-Auswertung, `FREEZE_PROPOSAL_EXT`, dann **Halt vor test-dev** für die Freigabe); (3) Austauscharchiv der Erweiterung mit FOR_CLAUDE.md wie zuvor.
  - Zweiter Datensatz grundsätzlich entschieden, **keine Arbeit daran, bis VisDrone abgeschlossen ist** (`e_e3_smoke` deshalb aus der Queue genommen; Skript bleibt).
  - Datensatz: **DOTA v1.5, horizontale Boxen**; xView zurückgestellt (Download nicht möglich).
  - Training auf allen 16 Klassen. Primärmetrik: All-Class-AP nach dem offiziellen HBB-Protokoll (DOTA_devkit Task 2). Präregistrierte Sekundärmetrik: AP über small vehicle + large vehicle. Diagnostik: klassenagnostische AP.
  - Eigener Split aus den gelabelten train/val-Daten (Test-Labels nicht öffentlich). Nach dem Download prüfen, dass die Instanzzahl v1.5 entspricht (≈ 403k; DOTA-Seite 403 318, DOTA-v2-Paper 402 089), nicht v1.0 (≈ 188k).
  - Eigenes Budget (≈ 20 GPU-h) und eigene Präregistrierung, geschrieben erst nach Abschluss von VisDrone.

- 2026-10-27 ~16:30: **Notiz für die spätere DOTA-Phase (Nutzer; keine Aktion jetzt).**
  - Kandidatenquelle: Ultralytics `DOTAv1.5.zip` (≈ 2 GB, 1411/458/937 Bilder, 16 Klassen, YOLO-OBB-Labels).
  - Vor Verwendung Audit im Stil von Phase 0: (a) Bildformat und Auflösung gegen das offizielle DOTA (neu kodiert?); (b) Instanzzahl gegen offizielles v1.5 (≈ 403k); (c) ist das „difficult“-Flag vorhanden, das die offizielle Auswertung braucht?
  - Fehlen das difficult-Flag oder die v1.5-Labels kleiner Objekte: Labels aus den offiziellen v1.5-Annotationsdateien; Ultralytics-Bilder nur, wenn das Audit Gleichwertigkeit mit den Originalen zeigt.

- 2026-10-27 ~18:20: **Nutzerentscheide DOTA-Phase** (ersetzt die vorläufigen Punkte von 16:15/16:30, wo abweichend; keine DOTA-Arbeit vor Abschluss von VisDrone).
  - Datensatz DOTA v1.5, horizontale Boxen (offizielle DOTA-HBB-Aufgabe; HBB = min/max der OBB-Ecken wie im Devkit; „difficult“ wie in der offiziellen Auswertung).
  - Modelle: nur YOLOv10s und YOLOv10m (Konsistenz mit Hauptstudie und Erweiterung).
  - Training auf allen 16 Klassen; Rezept fest 60 Epochen; Modellauswahl auf calib.
  - Metriken: primär All-Class-AP (offizielles HBB-Protokoll); präregistriert sekundär AP über small vehicle + large vehicle; Diagnostik klassenagnostische AP.
  - Daten: offizielle v1.5-Annotationen (≈ 403k Instanzen prüfen). Bilder aus Ultralytics `DOTAv1.5.zip` nur, wenn ein Audit im Stil von Phase 0 Gleichwertigkeit mit den Originalen zeigt (Format, Auflösung, Neukodierung).
  - Eigener Split aus den gelabelten Daten (Test-Labels nicht öffentlich), festgelegt vor jedem Ergebnis.
  - Eigenes Budget (≈ 20 GPU-h, per Smoke-Test neu geschätzt) und eigene Präregistrierung, erst nach Abschluss von VisDrone.
  - Einschränkung fürs Paper: HBB auf gedrehten, länglichen, dicht geparkten Objekten; OBB = future work.

- 2026-10-27 ~18:20: **Zeitplan host-b bis Sonntag, danach host-a (Nutzer).**
  - Auf host-b endet alles sauber bis **So 01.11. 12:00** (kein laufendes Training, alles committet). `runs/queue/DEADLINE` = 2026-11-01T09:00 (12:00 − 1 h Regel − ≈ 2 h Migrationsvorbereitung): der Runner startet keinen Job, dessen Schätzung später endet.
  - Priorität: (1) restliche Erweiterungsläufe, (2) CPU-Fusion/Zerlegung dieser Modelle, (3) E2 faire Latenz – **alle** Latenzzeilen der Erweiterung auf **einer** Maschine; passt der Job nicht vollständig vor die Deadline (bei Standardlimit), läuft er komplett auf host-a, (4) E4 + `FREEZE_PROPOSAL_EXT`, dann Halt; test-dev auf host-b nur bei Freigabe vor Sonntag, sonst auf host-a.
  - Sonntag vor 12:00: Stopp-Datei, Queue leer, Commits; Transferpaket (git-Bundle oder Push, falls der Nutzer ein Remote einrichtet; Gewichte, metrics.json, für offene Jobs nötige Caches, `configs/local/`, `data/splits/`, `env_export/host-b/`) mit Manifest (Größen, Prüfsummen); HANDOFF-Abschnitt „Maschinenwechsel host-b → host-a“.
  - Auf host-a: Env gegen `env_export/host-b/` (ultralytics, sahi, torch versionsgleich, sonst Halt); `core.fileMode=false`, `YOLO_CONFIG_DIR`, `configs/local/host-a.yaml`; Prüfsummen; Replay-Gate val-AP B-FULL 2048 (YOLOv10s), E-2048 (YOLOv10m), E-MZ-DENSE-2048 (YOLO11s) exakt (ΔAP = 0,000), sonst Halt; Leistungslimit vor jedem Latenzjob protokollieren; dann Queue fortsetzen.

- 2026-10-27 ~18:20: **Prognosekorrektur YOLO12s 2048 und Budget.** `scripts/ext_eta.py` erkannte den Retry-Job (`e_retry_train_…`) nicht als Training und setzte 7,5 h an; gemessen läuft Batch 2 mit ≈ 9,5 min/Epoche → ≈ 15,9 h (Ende ≈ Mi 09:05). Korrektur: laufende Trainings werden aus der eigenen gemessenen Wanduhrzeit hochgerechnet; die Retry-Geschwindigkeit fließt nicht in den Leistungslimit-Faktor (andere Batch). **Budgetprognose damit ≈ 76 GPU-h > 75** (verbraucht ≈ 24 inkl. Fehllauf 0,9; Rest ≈ 52 mit den konservativen Schätzungen: 12s-Rest 14,9, drei e_gpu 2,4, 11m 2048 6,6, 12m 1280 3,6, R3 8,4 + 11,3, e_gpu_r3 0,3, E2 1,5, test-dev 3,0). Nach der festen Kürzungsreihenfolge wäre der nächste Schritt das Streichen von YOLO11m 2048 → **Nutzerentscheid erbeten** (Streichen vs. Budget +≈ 2 GPU-h). Bis dahin YOLO11m 2048 + `e_gpu_yolo11m` in beiden Queues hinter R3 verschoben (Entscheidung erst ≈ Do 29.10. 09:40 nötig; Gesamtende unverändert).

- 2026-10-27 ~21:15: **Nutzerentscheid Budget 75 → 77 GPU-h**, YOLO11m 2048 bleibt (datierte Ergänzung in `docs/PREREGISTRATION_EXT.md`). Geprüft: Die Präregistrierung regelt nicht prüfbare Modelle bereits (Abschnitt 3 „Fehlende Läufe“, Ergänzung 0f6fecbc Punkt 2: zählen als „unklar“, nicht als erfüllt, „≥ 4 von 5“ mit dieser Zählung) → wie geschrieben angewendet, keine neue Regel, keine Auslegung nötig. YOLO12m: E-H1, E-H2(2048), E-H3 = „unklar (nicht prüfbar, präregistrierte Kürzung)“. Die Queue-Reihenfolge (YOLO11m 2048 hinter R3) bleibt; das ändert das Gesamtende nicht. Sonntags-Deadline unverändert.

- 2026-10-27 ~21:30: **Nutzerentscheid: YOLO12m 2048 wiederhergestellt, Budget nach Bedarf** (Ergänzung in `docs/PREREGISTRATION_EXT.md`, ersetzt die 77-GPU-h-Ergänzung). Umsetzung:
  - `scahi/ext.py`: `sizes("yolo12m")` = (1280, 2048), außer `cuts.json` enthält `drop_yolo12m_2048`. Neuer Job `e_smoke_yolo12m_2048` (`scripts/ext_smoke_mem.py`): 200 dichteste Trainingsbilder (≥ 152 GT je Bild), 80 Iterationen, `FixedBatchGuard`, Batch 3, sonst 2, jeweils in frischem Prozess → `runs/ext/smoke_yolo12m_2048.json`. Das Training verlangt diese Datei; bei Batch 2 schreibt es `OOM_FALLBACK = 2` (präregistrierter Rückfall, durch den Test ausgelöst). Batch 1 ist ausgeschlossen: OOM bei Batch 2 (Test oder Lauf) → `drop_yolo12m_2048` in `cuts.json`, der Lauf entfällt.
  - Grund für den neuen Test: Der Kosten-Smoke vom 26.10. meldete YOLO12s 2048 mit Batch 3 als passend (29,4 GB), der echte Lauf nicht. Vermutlich hat Ultralytics dort die Batch stillschweigend halbiert. Derselbe Smoke meldete YOLO12m 2048 Batch 3 mit 29,3 GB.
  - Queue (GPU): YOLO12s-Retry → e_gpu 12s → Speichertest 12m 2048 → YOLO11m 2048 → e_gpu 11m → YOLO12m 1280 → **YOLO12m 2048 (Schätzung 28 h)** → e_gpu 12m → R3 2560 → R3 3072 → e_gpu_r3. R3 steht zuletzt, weil es kein Urteil beeinflusst. Zwischen Speichertest (Mi ≈ 10:20) und Start von 12m 2048 (Mi ≈ 21:25) liegen ≈ 11 h für den Bericht an den Nutzer. CPU-Queue: Fusion 11m vor 12m.
  - Budgetprognose ≈ 105 GPU-h: verbraucht ≈ 23 (ohne Retry) + Retry 15,9 + Rest ≈ 66 (davon YOLO12m 2048 ≈ 28 + e_gpu 1, R3 8,4 + 11,3, E2 1,5, test-dev 3).

- 2026-10-27 ~21:15: **Drei explorative Post-hoc-Aufgaben (Nutzer; nur CPU, kein Einfluss auf E-H-Urteile).** Läuft parallel zum YOLO12s-2048-Training in tmux `scahi_posthoc`. Kerne 16–23 (taskset, 8 Kerne), nice 19, ionice idle, ein Thread je Prozess. Durchsatz des Trainings geprüft: 5,02 it/s gegenüber 5,1–5,2 vorher (≈ −2…−4 %, Grenze 5 %). CPU-Queue unverändert, keine Latenzmessung. Skript `scripts/ext_posthoc.py`.
  1. **Kopf-Definition, wörtlich aus PREREGISTRATION_EXT §2/§3:** E-1280/E-2048 „Einzelpass bei der Trainingsgröße (YOLOv10m: o2o; sonst Standardausgabe)“. E-H2: „Gleiche Gewichte, gleiche Pässe, **gleicher Zweig** (YOLOv10m: o2m gegen o2m; die o2o-NMS-Zeile wird zusätzlich berichtet).“ E-H3: „(YOLOv10m: DENSE o2m gegen Einzelpass o2o.)“ Nach der Nutzerregel gilt der Text: E-H2 YOLOv10m = o2m gegen o2m (+1,91 / +1,55). Die frühere Korrektur (+1,00 / +0,47, DENSE o2m − NMS o2o) widerspricht dem Text; diese Werte bleiben deskriptiv, zweigübergreifend. E-H3 = DENSE o2m − Einzelpass o2o (+0,26). Die Tabellen entsprachen bereits dem Text; es ändert sich nichts.
  2. **Over-Merging-Diagnostik** (s = 2048, A/B/C der Zerlegung): „crowded“ = nicht ignoriertes GT mit GT derselben Klasse bei IoU > 0,3 im selben Bild, sonst „isolated“. Recall@0,5 per greedy COCO-Matching je Klasse, (i) über alle Detektionen (max 500, wie in der Auswertung) und (ii) über die Top-k Detektionen je Bild (k = Anzahl nicht ignorierter GT, unabhängig von der Score-Skala). Merges: Detektion ist bestes Match (max IoU, gleiche Klasse) von ≥ 2 crowded GT bei IoU ≥ 0,5 bzw. ≥ 0,3. Nachbarpaare: in A getrennt erkannt, in B/C von einer Detektion geteilt. → `runs/ext/overmerge.json`.
  3. **Faire „nur F2“-Zeile B′:** F2 auf der Ganzbild-Ebene allein, auf calib getunt mit dem festen 24er-F2-Raster von C, Auswahl max calib-AP wie bei C; val-Zeilen `runs/ext/decomp/D_<m>_single<s>_F2tuned_<br>`; B′−A, B−A, B′−B, C−B′ mit gepaartem Bootstrap → `runs/ext/posthoc_tuneB.json`.
  **Ergebnis (21:31, exploratory, post hoc):**
  - B′ = B in allen 6 Kombinationen: Das eigene calib-Tuning wählt dieselbe Rasterkonfiguration wie C (tau_splat 0,005, tau_peak 0,001, cell_factor 5, bins_per_oct 5). B−A ist innerhalb des präregistrierten Rasters also keine bloße Untergrenze, sondern der faire Wert.
  - Das Optimum liegt bei 3 von 4 Parametern am Rand des Rasters.
  - Der Cluster-Parameter `iou_cluster` = 0,5 ist fest und nicht Teil des Rasters; das Over-Merging lässt sich mit dem Raster also nicht abstellen.
  - Durchsatz des Trainings unter Last: Epochenzeit 591 s (ohne calib-Auswertung) bzw. 613 s (mit) gegenüber 576–585 bzw. 613 s vorher → ≤ 2 %.

- 2026-10-28 ~07:25: **Clustering-Diagnostik F2 (exploratory, post hoc; kein Einfluss auf E-H-Urteile, alle präregistrierten Zeilen bleiben F2 mit `iou_cluster` = 0,5).**
  - **Korrektur zum Eintrag 21:15:** Das calib-Optimum liegt bei **allen 4** Rasterparametern am Rand (tau_splat 0,005 und tau_peak 0,001 jeweils unterer Rand, cell_factor 5 und bins_per_oct 5 jeweils oberer Rand), nicht bei 3 von 4.
  - Lauf: `scripts/ext_cluster_diag.py` (1b0115b), 64 Varianten = `iou_cluster` {0,5; 0,6; 0,7; 0,8} × je ein Schritt über den Rand (tau_splat {0,005; 0,00125}, tau_peak {0,001; 0,0001}, cell_factor {5; 7}, bins_per_oct {5; 6}); Zeilen B und C; YOLOv10m (o2m, o2o) und YOLO11s, 1280 und 2048; Auswahl max calib-AP. Sanity: die Referenzvariante reproduziert die calib-AP aus `tuning.json` in allen 6 C-Kombinationen exakt. Ergebnisse `runs/ext/cluster_diag.json`, Log `runs/ext/cluster_diag.log`, val-Zeilen `runs/ext/clusterdiag/`.
  - **Befund:** `iou_cluster` selbst ist praktisch wirkungslos (calib-AP über 0,5–0,8 bei sonst gleichen Parametern innerhalb ±0,1; bei C senkt 0,8 um bis zu 0,5). Der Gewinn stammt fast ganz von **cell_factor 5 → 7** (+0,6…+0,9 calib-AP, einzeln), dazu bins_per_oct 5 → 6 (+0,1…+0,2); die Schwellen tragen nichts bei. Die gewählte Konfiguration liegt in allen 12 Fällen wieder am (neuen) Rand (tau_splat 0,00125, tau_peak 0,0001, cell_factor 7, bins_per_oct 6).
  - val (gewählte Variante X): X − F2(0,5) = +0,86…+1,05 AP in allen 12 Fällen (CIs ohne 0), crowded-Recall@0,5 +7,7…+12,4. **Der crowded-Recall-Verlust gegenüber dem Einzelpass A bleibt aber deutlich** (X − A = −5,4…−16,9, alle CIs unter 0), ungefähr halbiert. Nahtfragmente kehren nicht zurück (frag_rate 0,004–0,005, FP-Rate 0–8 px am Kachelrand unverändert).
  - **Kriterium „beseitigt den crowded-Recall-Verlust klar, ohne Nahtfragmente“: nicht erfüllt** → kein DOTA-Kandidat für `iou_cluster`, keine zusätzliche test-dev-Zeile von selbst. Die Beobachtung „gröberes Raster (cell_factor) hilft, Optimum am Rand“ ist offen und wird dem Nutzer vorgelegt (Entscheidung beim Nutzer).
  - Durchsatz des Trainings unter Last: Epochenzeit 574–582 s gegenüber 576–585 s vorher, kein messbarer Verlust.

- 2026-10-28 ~07:30: **Nutzerentscheid nach der Clustering-Diagnostik (alles exploratory, post hoc; keine präregistrierte Zeile und kein E-H-Urteil ändert sich).**
  1. **Interpretation:** Der crowded-Recall-Verlust von F2 kommt vor allem von der räumlichen Auflösung der Peak-Suche (cell_factor), nicht von `iou_cluster`. Ein feineres Raster trennt Nachbarn.
  2. **Erweiterte calib-Suche („X-grid“)**, nur CPU, gleiche Auflagen (Kerne 16–23, nice 19, ionice idle, Trainingsdurchsatz ≤ 5 % Verlust): `scripts/ext_grid_ext.py`, Log `runs/ext/grid_ext.log`, Ergebnis `runs/ext/grid_ext.json`, val-Zeilen `runs/ext/gridext/`. Runde 1: cell_factor {7, 9, 11, 13} × bins_per_oct {6, 7, 8} × tau_splat {0,00125; 0,0003125} × tau_peak {1e-4; 1e-5} = 48 Varianten, dazu die bisherige Referenz. `iou_cluster` bleibt bei 0,5 (Diagnostik: wirkungslos). Abbruch, sobald das calib-Optimum in allen Fällen innen liegt oder die Fusionszeit pro Bild 2× die des jetzigen F2 übersteigt. Umsetzung: „am Rand“ nur, wenn der Randwert den nächsten inneren um > 0,01 calib-AP schlägt; sonst gilt der Fall als konvergiert. Fälle am Rand ohne Überschreitung bekommen eine weitere Runde (Randparameter einen Schritt weiter nach außen), höchstens 4 Runden. Auswahl: max calib-AP unter den Varianten mit Zeit ≤ 2× F2; das unbeschränkte Optimum wird mitberichtet. Kosten je Variante: Fusionszeit pro Bild als Verhältnis zur Referenz im selben Prozess (CPU geteilt, nur indikativ; die maßgebliche Messung ist E2), Spitzenspeicher (tracemalloc) über die 25 calib-Bilder mit den meisten Detektionen. Vorab-Messung (40 calib-Bilder, YOLO11s 2048 C): F2 80 ms/Bild, X aus der Diagnostik 145 ms (×1,8), cell_factor 13 / bins 8 181–196 ms (×2,3–2,4) → die 2×-Grenze wird voraussichtlich greifen. Bericht: calib/val AP, crowded/isolated Recall, dup/frag, Rand-FP, Zeit, Speicher; gepaarte CIs gegen F2(0,5), A und X. Spätere Modelle laufen nach, sobald ihre Caches da sind (`--models …`).
  3. **Exploratorische test-dev-Zeile:** Die calib-gewählte X-grid-Variante kommt in `FREEZE_PROPOSAL_EXT` als exploratorische Zeile (wie X1–X4), für alle verfügbaren Gewichtssätze, Zeilen C (MZ-DENSE) und B (Einzelpass + Fusion). Kennzeichnung: „selected on calib after val results were known“. Einmalige Auswertung auf test-dev zusammen mit den eingefrorenen Zeilen.
  4. **Kandidat für die DOTA-Präregistrierung (noch nicht präregistriert):** ein vorab festgelegtes, verbreitertes Raster für cell_factor und bins_per_oct, getunt auf dem DOTA-calib-Split. Die konfirmatorische DOTA-Hypothese zur dichten Fusion verwendet dieses Raster. Die konkreten Werte werden aus dem X-grid-Ergebnis abgeleitet und in der DOTA-Präregistrierung festgeschrieben, bevor ein DOTA-Ergebnis vorliegt.
  5. **E2:** X-grid wird in die Latenzmessung aufgenommen (optimierter Pfad), damit der AP-Gewinn gegen die Fusionskosten abgewogen werden kann. Der E2-Job ist noch nicht gebaut; die Zeile wird beim Bau ergänzt.

- 2026-10-28 ~07:50: **Qualitative Abbildung v2 – Regel, festgelegt vor dem Rendern (Nutzerauftrag; keine Auswahl nach Augenschein).**
  - Aufbau wie v1: YOLOv10s-Gewichte B-FULL-1280, Pässe von R (Pass bei 1280 + z = 2-Kacheln, Cache-Overlap 0,3). Box-Fusion = F0-NMS (o2o), dichte Fusion = F2 (o2m, mit Kalibrierung), Parameter jeweils calib-getunt aus `runs/p5/20261023_BFULL1280_fusion/tuning.json`.
  - **Arbeitspunkt je Methode:** Score-Schwelle bei Präzision 0,80 auf calib. Alle calib-Detektionen (≤ 500 je Bild, nach der Drop-Regel des Protokolls) werden je Bild und Klasse in Score-Reihenfolge gierig bei IoU ≥ 0,5 zugeordnet; ein Treffer auf ignoriertes GT zählt weder als TP noch als FP. Schwelle = kleinster Score s, bei dem die gepoolte Präzision aller Detektionen mit Score ≥ s noch ≥ 0,80 ist.
  - **Färbung auf val:** TP grün, FP rot, FN (nicht ignoriertes GT ohne Treffer am Arbeitspunkt) gelber Umriss. Treffer auf ignoriertes GT werden nicht gezeichnet.
  - **Bildauswahl (val):** zulässig sind Bilder mit ≥ 10 nicht ignorierten GT. ΔAP je Bild = Toolkit-Port-AP (dicht) − AP (Box) auf diesem Bild allein. Sortieren, in vier Quartile teilen, je Quartil das Bild, dessen ΔAP dem Median des Quartils am nächsten liegt (Gleichstand: kleinere Bild-ID). Dazu das Bild mit den meisten nicht ignorierten GT mit σ = √(w·h) < 16 px (Gleichstand: kleinere ID; ist es schon gewählt, das nächste).
  - **Ausschnitt:** je Bild ein Fenster von einem Drittel der Bildbreite und -höhe mit den meisten GT-Mittelpunkten (Raster-Schritt 30 px, Gleichstand: oben links zuerst), für beide Methoden gleich. Er hängt nur vom GT ab, nicht von den Detektionen.
  - Skript `scripts/qualitative_v2.py` → `reports/figures/paper/fig_qualitative_v2.*` (PDF, PNG, data.json, CSV mit Auswahl, Schwellen und Zählungen). v1 bleibt unverändert.

- 2026-10-28 ~08:25: **DOTA-Quelle heruntergeladen (Nutzerauftrag; sonst keine DOTA-Arbeit).**
  - `DOTAv1.5.zip` von der `download`-URL in `ultralytics/cfg/datasets/DOTAv1.5.yaml` der installierten Version 8.4.183 (`https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.5.zip`). Nur das Zip, ohne Ultralytics' Auto-Download oder Konvertierung, mit nice 19 / ionice idle. Freier Platz vorher: 3,3 TB auf `the data disk`.
  - Ablage auf host-b: `/path/to/datasets/DOTA/DOTAv1.5.zip` (Datenwurzel wie VisDrone in `configs/local/host-b.yaml`). 1 994 705 603 Bytes, SHA-256 `d8a4da6d6e3a6ddd4fd56d7a76e468e31dfbea6cce3b06e598f3cda57b1361b0`; 6554 Einträge, nicht entpackt. Daneben `README_SOURCE.txt` (Quelle, Datum, Version, Prüfsumme, „images to be audited against official DOTA before use“).
  - Endgültiger Ort: host-a, `/path/to/datasets/DOTA/`; in `configs/local/host-a.yaml` als DOTA-Wurzel einzutragen. Zip und README stehen mit Prüfsumme und Zielpfad im Migrationsmanifest (`runs/migration/manifest.tsv`, Kategorie `external`; `--verify` prüft am Zielpfad).
  - Die offiziellen v1.5-Annotationen (Google Drive) lädt der Nutzer selbst herunter; kein eigener Versuch.
  - Die leeren Ordner `train/`, `val/`, `test/` in `/path/to/datasets/DOTA/` (angelegt 08:23, nicht von mir) bleiben unberührt.
  - **Audit** (Bildformat und Auflösung, Neukodierung, Instanzzahl gegen offizielles v1.5 ≈ 403k, „difficult“-Flag) **und jede weitere DOTA-Arbeit erst nach Abschluss von VisDrone.**

- 2026-10-28 ~08:35: **Offizielle DOTA-v1.5-Annotationen abgelegt (vom Nutzer manuell geladen; nicht entpackt).** In `/path/to/datasets/DOTA/`:

  | Datei | Bytes | SHA-256 | Einträge |
  |---|---|---|---|
  | `train/DOTA-v1.5_train.zip` (OBB) | 3 931 525 | `72229b80f528289fa6fbaa263f562ead9940a9f7f8f17fd65827f959afbd3326` | 1411 |
  | `train/DOTA-v1.5_train_hbb.zip` | 3 271 683 | `7ec2edefd2758b459841f98a7ad796429972abe217ccdfb5415bbe274b04e70d` | 1411 |
  | `val/DOTA-v1.5_val.zip` (OBB) | 1 283 040 | `d63ccdad7be3e1a2f6df6e5df3c506a3b7ad519aab4e49c4104b7befd1c6a724` | 458 |
  | `val/DOTA-v1.5_val_hbb.zip` | 1 074 961 | `fab5149d7f7ec7167978630ec1c505ae8b3fa18d4939f1c3abccbc985706104a` | 458 |
  | `test/test_info.json` | 4 229 815 | `1bc04f945f45823fb6115ebde859b360a199e8a3ce24bb37bd02639862727b27` | – |

  Die Zahl der Annotationsdateien passt zu den Bildzahlen 1411 / 458. Alle Dateien unter `/path/to/datasets/DOTA/` stehen mit Prüfsumme und gleichem Zielpfad auf host-a im Migrationsmanifest. Hinweis für den Audit: Neben den OBB-Dateien gibt es offizielle HBB-Dateien. Ob wir HBB aus den OBB-Ecken bilden (Entscheid oben, wie im Devkit) oder die offiziellen HBB-Dateien nutzen, prüft der Audit (Gleichheit); entschieden wird nichts vor Abschluss von VisDrone.

- 2026-10-28 ~08:35: **DOTA-Split-Plan (Nutzer; später, keine Aktion jetzt).**
  - DOTA-test hat nur Bilder (Labels nicht öffentlich) und wird nicht verwendet; der DOTA-Evaluationsserver wird nicht genutzt.
  - **Offizielles DOTA-val (458 Bilder) = unser zurückgehaltener Testsatz**, genau einmal nach dem Freeze ausgewertet. Dafür sind die offiziellen v1.5-Labels mit „difficult“-Flag nötig.
  - Aus DOTA-train: train_main + calib (≈ 10 %, größenstratifiziert, fester Seed) für Modellauswahl und jedes Tuning. Kein separater Entwicklungs-Split.
  - Die Hypothesen sind auf dem zurückgehaltenen Testsatz definiert.
  - ε = 0,30 AP aus VisDrone übernommen (ein Seed je Lauf; als Annahme ausgewiesen).

- 2026-10-28 ~08:45: **DOTA HBB-Quelle (Nutzer; später, keine Aktion jetzt; ersetzt „HBB aus den OBB-Ecken“ als Quelle).**
  - Training und Auswertung nutzen die **offiziellen HBB-Annotationsdateien** (`DOTA-v1.5_*_hbb.zip`); sie definieren die offizielle HBB-Aufgabe.
  - Der Audit vergleicht sie zur Kontrolle mit Boxen aus den OBB-Ecken (min/max wie im Devkit). Bei Abweichungen gelten die offiziellen HBB-Dateien; die Unterschiede werden dokumentiert.
  - Der Audit prüft außerdem, dass die HBB-Dateien das „difficult“-Flag tragen.

- 2026-10-28 ~11:45: **YOLO12m 2048 bleibt gestrichen; X-grid Zeile C erweitert; DOTA-Kandidat „dichte Fusion als NMS-Ersatz“ (Nutzer).**
  1. **YOLO12m 2048:** Der Worst-Case-Speicher-Smoke (09:44, `runs/ext/smoke_yolo12m_2048.json`) ergab bei Batch 3 und bei Batch 2 je einen OOM vor der ersten Iteration (Ultralytics halbierte die Batch; Spitze reserviert 29,3 bzw. 25,9 von 31,3 GB). Nach der vorab festgelegten Regel (Addendum 2026-10-27, nur Batch ≥ 2) ist der Lauf gestrichen (`cuts.json`: `drop_yolo12m_2048`). Keine Regel wird geändert. Konservative Lesart: Für YOLO12m gelten E-H1, E-H2 (s = 2048) und E-H3 als **„nicht erfüllt“** (nicht testbar; Holm p = 1; kein Delta, nicht im gepoolten Effekt). E-H2 (s = 1280) bleibt testbar. `scripts/ext_e4.py` setzt das um und nennt im Bericht, wenn „≥ 4 von 5“ nicht mehr erreichbar ist. Stand jetzt (Dry-Run): **E-H3 kann „≥ 4 von 5“ unabhängig von YOLO11m nicht mehr erreichen**, weil YOLOv10m unter ε liegt (+0,26, CI [+0,01; +0,51]) und YOLO12m nicht testbar ist (höchstens 3 von 5). Das steht so im E4-Bericht.
  2. **X-grid, Zeile C, eine weitere Runde** (exploratorisch, post hoc, CPU, gleiche Auflagen). Anlass: In Runde 1 lagen bei vier C-Fällen (YOLOv10m 1280 o2m/o2o, YOLOv10m 2048 o2o, YOLO11s 1280) alle Varianten über 2× F2-Zeit, X-grid fiel auf die Referenz zurück und lag ≈ 1 AP unter X. Für Zeile C jetzt cell_factor {7, 9, 11, 13} × bins_per_oct {6, 7, 8} mit jeder Schwelle sowohl auf dem Referenzwert als auch abgesenkt: tau_splat {0,005; 0,00125; 0,0003125} × tau_peak {1e-3; 1e-4; 1e-5} = 108 Varianten (vollständig gekreuzt, also auch eine Schwelle auf Referenz und eine abgesenkt). Vorhandene Fälle rechnen nur die fehlenden 60 Varianten. Zeile B bleibt bei Runde 1. Die eingefrorene exploratorische test-dev-Zeile bleibt die calib-gewählte Variante unter der 2×-Grenze. **„converged“ korrigiert:** converged = Optimum ohne Zeitgrenze liegt innen (FLAT-Regel); zusätzlich „converged_under_cap“ für die gewählte Variante; beide werden berichtet (bisher bezog sich das Flag nur auf die gewählte Variante). Bericht je Fall: volle calib-Kurve „AP vs. Fusionszeit pro Bild“ mit allen Varianten, auch über der Grenze (`runs/ext/grid_ext_calib.csv`, Abbildung `reports/figures/ext/fig_ext_xgrid_cost.*`), dazu crowded/isolated Recall und dup/frag. Gleiches Verfahren (Runde 1 für B, 108er-Raster für C) für YOLO12s (läuft mit), YOLO11m und YOLO12m 1280 (CPU-Queue-Jobs `e_xgrid_<m>` nach `e_figs_<m>`). Ergebnis von Runde 1 gesichert in `runs/ext/grid_ext_round1.{json,log}`.
  3. **Kandidat für die DOTA-Präregistrierung (exploratorischer Befund, noch nicht präregistriert):** Mit dem feineren Raster erreicht die Fusion eines einzelnen Passes (Zeile B, X-grid) auf val die Standard-Dekodierung und übertrifft NMS bei YOLO11s: B(X-grid) − A = +1,24 [+1,09; +1,38] AP (1280) und +1,22 [+1,08; +1,37] (2048); bei YOLOv10m (NMS-freier o2o-Kopf als Standard) −0,13 … −0,05 AP, also gleichauf. Kandidat: dichte Fusion als NMS-Ersatz für Modelle mit einem einzigen (o2m-)Kopf. YOLO12s, YOLO11m und YOLO12m werden ergänzt, sobald ihr X-grid vorliegt.

- 2026-10-28 ~15:05: **X-grid eingefroren; Aufbau des E4-Berichts; DOTA-Kandidat dichte Fusion (Nutzer).**
  1. **X-grid-Suche beendet:** keine weiteren Runden, keine weiteren Parameter. Die calib-gewählte Konfiguration je Gewichtssatz (Zeilen B und C, 16 Gewichtssätze von YOLOv10m, YOLO11s, YOLO12s) ist eingefroren in `configs/ext/xgrid_frozen.json` (`scripts/ext_freeze_xgrid.py`; ein bestehender Eintrag wird nie geändert, eine abweichende Auswahl bricht ab). Eingefroren ist auch das Verfahren: Raster Zeile B = Runde 1 (48 Varianten), Zeile C = 108 Varianten, Auswahl max calib-AP bei Fusionszeit ≤ 2× F2. YOLO11m und YOLO12m 1280 bekommen genau dieses Verfahren ohne Änderung (Queue-Jobs `e_xgrid_<m>`, danach Freeze-Eintrag). `scripts/ext_grid_ext.py` bricht ab, wenn Raster oder Zeitgrenze vom eingefrorenen Stand abweichen.
  2. **FREEZE_PROPOSAL_EXT:** X-grid-Zeilen B und C für alle Gewichtssätze als exploratorische test-dev-Zeilen. Kennzeichnung: „grid found in a multi-round post-hoc search after val results were known; selection on calib only“. Eigene Tabelle, nie für ein E-H-Urteil.
  3. **Aufbau des E4-Berichts** (`scripts/ext_e4_report.py` → `reports/E4_EXT.md`, bis zum finalen E4 als INTERIM markiert, wird von `e_figs_<m>` und `e_xgrid_<m>` neu erzeugt): (a) präregistrierte Urteile E-H1–E-H4 genau wie definiert; (b) Diagnose: Zerlegung A/B/C und Over-Merging; (c) exploratorischer Fix X-grid mit val-CIs, klar gekennzeichnet, daneben R3 (YOLOv10s 2560/3072) als Rechenkontrolle (erreicht ein größerer Einzelpass dieselbe AP bei geringeren Kosten? AP und GFLOPs/Bild); (d) Platzhalter für die Bestätigung außerhalb der Stichprobe: VisDrone test-dev (exploratorische Zeilen) und DOTA (präregistriert).
  4. **Kandidat für die DOTA-Präregistrierung (noch nicht festgeschrieben):** dichte Fusion mit dem eingefrorenen X-grid-Verfahren (gleiche Zeitgrenze, Auswahl auf DOTA-calib; **Raster für B und C gleich und vollständig, siehe Nachtrag 2026-10-29 ~18:30**), sowohl als Mehrpass-Fusion (Zeile C gegen Einzelpass) als auch als NMS-Ersatz (Zeile B gegen A). Ergänzung zum Eintrag ~11:45 Punkt 3: YOLO12s Zeile B − A = +1,35 [+1,22; +1,51] AP (1280) und +1,46 [+1,30; +1,63] (2048).

- 2026-10-28 ~17:05: **G-E1 gerissen (YOLO11m 2048), Ursache, Gate-Korrektur für alle Modelle (Nutzer; gleiche Behandlung wie die G4a-Korrektur vom 2026-10-24).**
  - **Befund (15:53):** `e_gpu_yolo11m` scheiterte an G-E1. Dekodierung bei 0,61 der 4-ulp-Toleranz bestanden; die AP-Bedingung „calib-Toolkit-AP aus eigener vs. Ultralytics-Dekodierung identisch auf 4 Dezimalen“ nicht: 42,358175 vs. 42,358728 (Δ = −0,00055 AP). Das Gate setzte `STOP`. Reproduzierbar (zweimal identisch nachgerechnet).
  - **(a) Ursache** (`scripts/ge1_rootcause.py`, `runs/ext/ge1_rootcause/`): Detektionen beider Dekodierungen genau wie im Gate (calib, 594 Bilder). Auf der GPU sind GT, Anzahl (198 284; nach dem Verwerfen in Ignore-Regionen 184 974), Reihenfolge, Scores und Klassen identisch; die Koordinaten weichen um höchstens 3,3·10⁻⁴ px ab. Die Toolkit-Zuordnung über alle 10 IoU-Schwellen 0,50–0,95 unterscheidet sich in **genau einem Eintrag**: Bild `0000339_01177_d_0000167`, Klasse motor, Box 11,7 × 19,8 px, Schwelle **0,55**; IoU 0,5499966 (eigene, kein Treffer) vs. 0,5500025 (Ultralytics, Treffer), Abstand zur Schwelle ≤ 3,4·10⁻⁶. Ursache ist die rechte Boxkante, die um 1,6·10⁻⁴ px abweicht (≈ 2,7 float32-ulp der Koordinate bei x ≈ 729 px); bei einer so kleinen Box verschiebt das die IoU um ≈ 6·10⁻⁶. **Vollständig erklärt:** Ersetzt man in den Zuordnungen der eigenen Dekodierung nur diesen Eintrag durch den von Ultralytics, ergibt sich exakt die Ultralytics-AP (42,358727587); die nachgerechnete AP stimmt mit der regulären Auswertung exakt überein. Auf der CPU (andere Numerik) gibt es keinen Unterschied (ΔAP = 0). **Bestätigungsbedingung (präzisiert durch den Nutzer):** „eine Zuordnung kippt an einer der zehn IoU-Schwellen; das Ersetzen genau der betroffenen Zuordnung(en) reproduziert die Ultralytics-AP“ – erfüllt (eine Zuordnung, Schwelle 0,55, Ersetzen reproduziert die Ultralytics-AP exakt). Die ursprüngliche Formulierung „wenige ulp von der IoU-Schwelle 0,5“ war ungenau (die AP mittelt über 0,50–0,95; der Abstand liegt bei ≈ 2,7 ulp der Koordinate, nicht des IoU-Werts); Nutzer: Ursache vollständig erklärt, Gate-Korrektur bleibt.
  - **(b) Gate-Korrektur für alle Modelle:** AP-Bedingung neu **|AP_eigen − AP_Ultralytics| ≤ 0,01 AP** (weit unter ε = 0,30); die Koordinatenbedingung (≤ 4 ulp, TF32 aus) bleibt. Die Bedingung „identisch auf 4 Dezimalen“ war ein **Konstruktionsfehler, der alle Modelle betraf** (empfindlich für einzelne Zuordnungen, die genau an einer IoU-Schwelle kippen), kein Dekodierproblem: Auch YOLO12s 1280 (Δ = 0,000021) und YOLOv10m 1280 o2m (Δ = 0,000002) wichen ab und bestanden nur, weil die 4. Dezimale zufällig gleich gerundet wurde. **Neuprüfung aller bisher geprüften Modelle** aus den gespeicherten Gate-Werten (`runs/ext/ge1_recheck_20261028.json`): YOLOv10m, YOLO11s, YOLO12s, YOLO11m bestehen alle (größtes |ΔAP| = 0,00055, YOLO11m 2048; Koordinaten ≤ 0,92 der Toleranz). Gate-Dateien um die neue Prüfung ergänzt, alte Prüfung erhalten (`gate_GE1_strict_4dp.json`). Keine Änderung der Präregistrierung; Code `scahi/ext.py` (`GE1_AP_TOL`).
  - **(c) Unabhängigkeit der Urteile** (vereinfacht, Nutzer): Sobald die YOLO11m-Zeilen existieren, wird für jedes YOLO11m-Urteil (E-H1, E-H2 bei 1280 und 2048, E-H3, Holm) der Abstand zu den Entscheidungsgrenzen berechnet (Punktschätzer zu ε, CI-Grenzen zu 0). Liegt jeder Abstand über 0,05 AP (≈ 90× die größte Dekodierdifferenz), ist dokumentiert, dass die Wahl der Dekodierung kein Urteil ändern kann, und es gibt keinen zweiten Cache; sonst wird ein zweiter val-Cache mit Ultralytics-Boxen gebaut. Hinweis: Die Einzelpass-Zeilen E-1280/E-2048 laufen ohnehin über den Ultralytics-Predictor; betroffen wären nur die MZ-Zeilen aus den Caches.
  - **(d)** `e_retry_gpu_yolo11m` (eigene Dekodierung wie bisher) läuft nach YOLO12m 1280 und `e_gpu_yolo12m`.

- 2026-10-28 ~17:05: **Prüfung auf das Ultralytics-NMS-Zeitlimit und andere Kappungen (Nutzer).** Alle 97 Logs unter `runs/` (Hauptstudie und Erweiterung: Einzelpass-Zeilen über den Predictor, SAHI-Zeilen, Gate-Läufe, test-dev) nach „NMS time limit“ durchsucht (`runs/ext/nms_time_limit_check.tsv`). Treffer nur in den CPU-Fusionsjobs `e_fuse_yolov10m` (30×), `e_fuse_yolo11s` (41×), `e_fuse_yolo12s` (37×), aus der NMS je Pass (`pass_detections`, Box-Fusions-Zeilen inkl. MZ-NMS-IOS); alle übrigen Logs: nein. **Keine Zeile ist betroffen:** Ultralytics prüft das Zeitlimit erst, nachdem das Ergebnis des Bildes gespeichert ist; alle unsere Aufrufe haben Batch 1. Empirisch: Zeitlimit bei jedem Aufruf erzwungen → 1389/1389 echte dichte Pässe (252 218 Detektionen) bitgleich zum normalen Aufruf. Die zweite, stille Kappung (`max_nms` = 30 000 Kandidaten vor der NMS) greift nie: Maximum 10 668 Kandidaten über 72 607 geprüfte Pässe.

- 2026-10-28 ~17:05: **YOLO12m 1280 trainierte 24 min auf der CPU (Fehler im Wrapper, behoben).** 16:33 OOM bei Batch 8 → vorab festgelegte Reduktion 8 → 6; der Neustart im selben Prozess (`os.execv`, gleiche PID, CUDA-Kontext des gescheiterten Laufs noch belegt) konnte CUDA nicht initialisieren (Error 304), Ultralytics wich still auf die CPU aus (18 s/it). Um 16:57 von mir beendet (nur ein Teil von Epoche 1, keine Gewichte). Korrektur: `train_full` bricht ab, wenn CUDA fehlt oder der Trainer nicht auf der GPU läuft; nach einem OOM wird nur die Reduktion eingetragen und der Job beendet, das Training startet in einem frischen Prozess als Queue-Retry (`e_retry_train_yolo12m_1280`). Speicher-Smoke Batch 6 vorab (dichteste Bilder): bestanden, Spitze 29,8 / 31,3 GB, 0,23 s/it. Retry seit 17:01 auf der GPU.

- 2026-10-28 ~17:20: **YOLO12m 1280: Rezeptabweichung Batch 6 statt 8 (zu berichten wie YOLO12s 2048).** Das Standardrezept bei 1280 ist Batch 8 (`BATCH[1280]`). Nach dem OOM bei Batch 8 greift die präregistrierte einmalige Reduktion 8 → 6 (`OOM_FALLBACK`). Mit nbs 64 ist accumulate = round(64/6) = 11, effektive Batch **66** statt 64 (YOLO12s 2048: 2 × 32 = 64). Alles andere unverändert (COCO-Init, 100 Epochen, calib-Auswahl). In `training.json` stehen batch, accumulate, effective_batch und oom_fallback; in Tabellen und im Paper als Rezeptabweichung zu kennzeichnen (YOLO12s 2048: Batch 2, YOLO12m 1280: Batch 6; jeweils weniger Bilder pro BatchNorm-Schritt).

- 2026-10-28 ~21:40: **YOLO12m 1280: zweiter OOM (Batch 6, Epoche 66), Fortsetzung aus `last.pt` – vom Nutzer genehmigt (~21:45).** 21:07 brach der Lauf nach 65 abgeschlossenen Epochen (`last.pt`, `calib_curve.json` bis Epoche 65 vorhanden) im Backward mit OOM ab: 1,83 GiB angefordert, 6,42 GiB von PyTorch reserviert, aber unbelegt (Fragmentierung). Davor 159 abgefangene Allokator-Warnungen (Ultralytics weist dann pro Bild zu; Modell-Batch unverändert). Die Präregistrierung sieht nur eine einmalige Reduktion vor (8 → 6); eine zweite Reduktion erfolgt nicht. Vorgehen: **Fortsetzung bei Batch 6 aus `last.pt`** (gleicher Mechanismus wie HALT/Resume), in einem frischen Prozess mit `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (betrifft nur die Speicherverwaltung, nicht die Rechnung); höchstens 3 Versuche, jeder ab dem jeweils letzten `last.pt` (`e_resume_train_yolo12m_1280`). Danach `e_retry2_gpu_yolo12m`. Rezept, Batch 6 / effektive Batch 66 und die calib-Auswahl bleiben unverändert; die Unterbrechung wird im Bericht wie die Batch-Abweichung vermerkt. GPU-Reihenfolge: `e_retry_gpu_yolo11m` (läuft) → YOLO12m-Fortsetzung (~2,5 h) → `e_retry2_gpu_yolo12m` → R3. Die CPU-Queue wartete in `e_fuse_yolo12m` auf YOLO12m-Caches und hätte die YOLO11m-Kette blockiert: wartenden Job beendet, YOLO11m-Kette vor die YOLO12m-Kette gestellt (`e_retry_fuse_yolo12m`).
  - **Genehmigung (Nutzer, ~21:45):** Fortsetzung aus `last.pt` angenommen. Optimizer, LR-Schedule, EMA und Epoche werden wiederhergestellt (`last.pt`: Epoche 65 abgeschlossen, batch 6, 6250 Updates); nur der RNG-Zustand der Augmentierung weicht ab Epoche 66 vom ununterbrochenen Lauf ab. `expandable_segments` ändert die Speicherverwaltung, nicht die Rechnung.
  - **Vor dem nächsten Versuch festgelegt:** (1) Scheitern alle 3 Fortsetzungsversuche, gilt YOLO12m 1280 als **nicht testbar** (konservativ „nicht erfüllt“ für die betroffenen Hypothesen; YOLO12m hat dann keine Trainingsgröße mehr, alle YOLO12m-Zeilen nicht testbar). Umsetzung: `_resume_fresh` schreibt `drop_yolo12m_1280` nach `runs/ext/cuts.json`; `sizes()` lässt die Größe dann weg; `e_gpu`, `e_fuse`, `e_xgrid` laufen für YOLO12m nicht; `ext_e4._untestable` meldet das ganze Modell. Ein teilweise trainierter Checkpoint wird nie verwendet (`e_gpu` verlangt `TRAIN_DONE`, das nur ein vollständiger Lauf schreibt). Mit simuliertem Eintrag geprüft (`cuts.json` unverändert). (2) Batch 6 ist als Rezeptabweichung erfasst (Eintrag ~17:20: einzige erlaubte Reduktion 8 → 6, effektive Batch 66, BatchNorm-Hinweis); die Unterbrechung bei Epoche 65/66 wird dort mitberichtet. (3) Batch-Wächter: `FixedBatchGuard(6)` ist auch im Resume-Zweig von `train_full` registriert (`on_train_batch_end`; bricht mit `BatchReduced` ab, falls Ultralytics die Batch selbst ändert; dass er greift, zeigte der Lauf bei Batch 8 am 2026-10-28 16:33). Ein solcher Abbruch zählt als gescheiterter Versuch; mit reduzierter Batch wird nie weitertrainiert.

- 2026-10-29 ~08:00: **Nacht 28./29.10.: YOLO12m 1280 fertig; zwei Folgefehler behoben; G-E1 (c) für YOLO11m abgeschlossen.**
  - **YOLO12m 1280:** Die Fortsetzung lief im ersten Versuch bis Epoche 100 durch (21:35–23:51, Batch 6, kein weiterer OOM, kein Batch-Wächter-Abbruch). calib-Auswahl: beste Epoche 65 (38,27 AP). Die gewählte Epoche ist die letzte vor der Unterbrechung; die 35 fortgesetzten Epochen haben die calib-AP nicht verbessert. `drop_yolo12m_1280` wurde nicht gesetzt. Im Bericht als Rezeptabweichung (Batch 6) mit Unterbrechung bei Epoche 65/66 zu vermerken.
  - **Fehler 1:** `e_retry2_gpu_yolo12m` (00:15) – `caches()` lief über beide Stacks und versuchte die gestrichenen 2048-Gewichte zu laden. Gate G-E1, Einzelpass- und SAHI-Zeilen waren schon geschrieben. Behoben: nur Größen aus `sizes(model)`. Neu `e_retry3_gpu_yolo12m`.
  - **Fehler 2:** `e_figs_yolo11m` (22:04) – der Forest-Plot erwartete für jeden Eintrag ein CI, auch für nicht testbare Einträge (YOLO12m 2048). Behoben: Diese Einträge werden als „untestable“ beschriftet. Neu `e_retry_figs_yolo11m`. Die Urteilslogik ist nicht betroffen.
  - **G-E1 (c) YOLO11m – Abstände zu den Entscheidungsgrenzen** (val, Dry-Run nach `e_retry_fuse_yolo11m`, ε = 0,30):

    | Hypothese | Δ AP [95 %-CI] | Δ − ε | CI-Untergrenze − 0 | p (einseitig) |
    |---|---|---|---|---|
    | E-H1 | +2,955 [+2,548; +3,380] | +2,655 | +2,548 | 0,001 |
    | E-H2, s = 1280 | +1,988 [+1,730; +2,242] | +1,688 | +1,730 | 0,001 |
    | E-H2, s = 2048 | +1,827 [+1,561; +2,080] | +1,527 | +1,561 | 0,001 |
    | E-H3 | +1,533 [+1,288; +1,788] | +1,233 | +1,288 | 0,001 |

    p liegt jeweils am Minimum des Bootstraps (1/1001). Der Holm-Schritt wird damit unabhängig vom Rang abgelehnt. Kleinster Abstand: 1,23 AP (auch CI-Untergrenze − ε: ≥ 0,99 AP). Vergleich: der Dekodierungsunterschied beträgt 0,00055 AP. Alle Abstände liegen weit über 0,05 AP → **kein zweiter Cache** (Regel (c), Nutzer). Die YOLO11m-Urteile hängen nicht von der Dekodierung ab.

- 2026-10-29 ~08:45: **NMS-Zeitlimit: Suche wiederholt, jetzt als Skript (Nutzer, vor E4).** `scripts/nms_log_check.py` durchsucht alle 113 Logs unter `runs/` und `reports/` und ordnet jedes Log einem Lauftyp zu (Ergebnis je Lauf ja/nein: `runs/ext/nms_time_limit_check.tsv`). Warnung = ja nur in den fünf CPU-Fusionsjobs der Erweiterung: `e_fuse_yolov10m` (30×), `e_fuse_yolo11s` (41×), `e_fuse_yolo12s` (37×), neu `e_retry_fuse_yolo11m` (27×) und `e_retry_fuse_yolo12m` (3×). Alle Einzelpass-, SAHI-, Gate-, Cache-, Latenz-, test-dev- und Trainingsläufe beider Studien: nein. Das Zwangstest-Verfahren vom 2026-10-28 ist jetzt reproduzierbar (`scripts/nms_limit_forced_check.py`). Für die beiden neuen Modelle: YOLO11m 1280 / 2048 und YOLO12m 1280, je 50 val-Bilder, alle Pässe der Stacks (471 / 470 / 471 Pässe). Ergebnis mit erzwungenem Zeitlimit bitgleich, kein Pass weicht ab. Maximal 3 658 / 4 722 / 3 698 Kandidaten, weit unter der stillen Kappung `max_nms` = 30 000 (`runs/ext/nms_limit_forced_yolo11m_yolo12m.json`). **Keine Zeile ist betroffen; für E4 sauber.**

- 2026-10-29 ~08:45: **E2-Slot auf host-b (Nutzer).** Neuer Job `e_latency_ext` (`scahi/ext_latency.py`) in der GPU-Queue nach `e_train_yolov10s_3072` und `e_gpu_r3`, Queue-Schätzung 4,0 h.
  - **Ablauf:** (1) GPU-Leistungslimit = Standard, sonst zurückgestellt (Exit 77). (2) `STOP_CPU`; der Job wartet, bis der CPU-Runner beendet ist und die GPU-/CPU-Exklusivitätsprüfung sauber ist. (3) Smoke über alle Zeilen und Pfade (3 Bilder). (4) Alle Latenzzeilen der Erweiterung in einem Lauf auf val (548 Bilder). Eine Zeile mit unsauberer Exklusivitätsprüfung wird höchstens dreimal gemessen. (5) Danach wird `STOP_CPU` entfernt und die CPU-Queue neu gestartet.
  - **Zeilen:** alle E-Zeilen (Einzelpass, SAHI, MZ-NMS, MZ-DENSE; YOLOv10m beide Zweige), R3, die X-grid-Zeilen B und C (eingefrorene Auswahl), sowie R, R2, M_ab und X2 der alten Studie. Stand heute 60 Zeilen; mit den YOLO12m-Fusions- und X-grid-Zeilen und R3 etwa 66.
  - **Zwei Spalten:** Box-Fusions-Zeilen laufen über den Ultralytics-Predictor (gefuste Gewichte, ein Zweig). Das ist schon der Standardpfad; eine Messung, berichtet in beiden Spalten. Dichte Zeilen: Referenzpipeline (ungefust, CPU-Fusion) und optimierter Pfad (ein Kopf, GPU-F2/F3e, Parität `runs/ext/e2_pipeline_parity.json`), aufgeteilt in Pässe und Fusion.
  - **Ergebnis** nur dann als `runs/ext/e2/latency.json`, wenn alle Zeilen in diesem einen Lauf gemessen sind. Eine Teildatei eines abgebrochenen Laufs wird beiseitegelegt und nie gemischt.
  - **Smoke vorab:** `e_latency_ext_smoke` (3 Bilder je Zeile und Pfad, ohne Zeitwert) läuft zwischen R3 2560 und R3 3072, damit Fehler vor dem Slot sichtbar werden.
  - **Deadline-Regel:** Passt der Slot nicht vor Sonntag 09:00, stellt der Runner ihn zurück (Schätzung + Startzeit > Deadline → Queue-Halt, Status „deferred“). Dann wird auf host-b nichts gemessen; E2 läuft vollständig auf host-a. Dasselbe gilt, wenn der Lauf auf host-b abbricht.

- 2026-10-29 ~13:50: **CPU-Vorbereitung bis E2 (Nutzer; ≤ 8 Kerne 16–23, nice 19, ionice idle, kein GPU-Zugriff).**
  - **E4-Dry-Run (INTERIM)** mit allen fünf Modellen über die finale Pipeline (`ext_e4.py --dry-run` → `ext_e4_report.py`): Urteile je Modell mit CI, Zählung „≥ 4 von 5“ mit konservativer Lesart, Holm (jetzt mit adjustiertem p), gepoolte Effekte; neu Größenklassen (AP und AP50 je GT-Größe, Zeilen a/b jeder gezählten Hypothese, deskriptiv) und ein Latenzabschnitt (Platzhalter bis E2). Nach E2 kommen nur R3-Zeilen und Latenzspalten hinzu.
  - **Over-Merging zusätzlich bei s = 1280** (`ext_posthoc.py overmerge --s 1280` → `runs/ext/overmerge_s1280.json`), weil nur dort alle fünf Modelle existieren (YOLO12m 2048 gestrichen); `fig_ext_recall` zeigt jetzt beide Größen; `e_figs` rechnet beide.
  - **FREEZE_PROPOSAL_EXT (Entwurf):** `scripts/ext_freeze_proposal.py` → `configs/ext/frozen_rows_ext.draft.json` + `docs/FREEZE_PROPOSAL_EXT.md`: 60 Zeilen (38 präregistriert, 22 X-grid exploratorisch), je Zeile exakte Inferenzkonfiguration, Gewichte mit SHA-256, Herkunft jeder Wahl (calib / fest nach Design / exploratorisch). R, R2, M_ab, X2 nicht erneut (configs/final). R3 kommt nach `e_gpu_r3` dazu.
  - **test-dev-Codepfad der Erweiterung:** `scahi/ext_testdev.py` (liest nur den Freeze-Eintrag; Gewichte per SHA-256 geprüft; test-dev verweigert einen nicht committeten Freeze).
  - **Replay-Gate, Cache-Teil:** alle 44 Cache-Zeilen (MZ-NMS, MZ-DENSE, X-grid B/C) über den test-dev-Codepfad auf val neu gerechnet: **44/44 exakt (max |ΔAP| = 0)** (`runs/ext/testdev_replay/gate_replay.json`). GPU-Teil (Einzelpass, SAHI, R3; 16 Zeilen) als Job `e_replay_predictor` in der GPU-Queue **nach** `e_latency_ext`.
  - **Migration:** Manifest um die Caches von YOLO11m und YOLO12m ergänzt (Kategorie `cache`, 6852 Dateien, 1,50 GiB; calib + val aller Trainingsgrößen); Gewichte beider Modelle enthalten. Manifest jetzt 7519 Dateien, 5,72 GiB.
  - **Archiv-Probelauf:** `scripts/make_exchange_zip_ext.py --dry-run` → `exports/scahi_ext_exchange_20261029_<commit>_dryrun.zip` (7,9 MB, 186 Dateien, FOR_CLAUDE.md 68 Zeilen, alle gelisteten Pfade vorhanden). Nicht verschickt. Das echte Archiv erst nach E4 final + FREEZE_PROPOSAL_EXT; dann wird `reports/FOR_CLAUDE_EXT.md` committet (im Archiv als `FOR_CLAUDE.md`).

- 2026-10-29 ~16:30: **R3 B-FULL 3072 entfällt (Addendum 0f6fecbc wörtlich angewandt).**
  - `e_train_yolov10s_3072` lief um 15:48 im ersten Trainingsschritt bei Batch 3 in einen OOM (26,7 GB belegt, 4,1 GB angefordert). Der Smoke-Test (120 Iterationen, Spitze 26,7 GB) hatte gepasst, das echte Training nicht.
  - Der generische Code schrieb daraufhin die Rückfall-Batch 2 (`OOM_FALLBACK`). Das Addendum erlaubt 3072 aber nur „ohne Rezeptänderung (Batch 3)“; Batch 2 wäre eine Rezeptänderung. Deshalb **kein Neustart mit Batch 2**: `runs/ext/cuts.json` → `fits_3072: false` mit Begründung; `e_train_r3(3072)` und `sizes("yolov10s")` überspringen 3072 damit.
  - `e_gpu_r3` scheiterte nur an den fehlenden 3072-Gewichten; die R3-Zeile 2560 (`E_yolov10s_full2560_o2o`, val AP 39,51, AP50 64,42; Auswahl o2o auf calib, Epoche 60) war vorher geschrieben. R3 besteht damit aus einer Zeile.
  - Folge: Der E2-Slot startete schon um 15:50 statt nach ~11 h Training. E2 misst alle 65 Zeilen inkl. R3 2560 bei Standardleistung 575 W, CPU-Queue pausiert.
  - Umkehrbar: Wenn der Nutzer 3072 mit Batch 2 doch will, als gekennzeichnete Abweichung (~12–14 GPU-h, passt noch vor So 09:00).

- 2026-10-29 ~17:00: **Nutzerentscheide nach R3 3072.**
  1. **R3 3072 bleibt gestrichen** („keine Rezeptänderung“, konsistent mit YOLO12m 2048; Batch 2 brächte den BatchNorm-Vorbehalt). Die Rechenkontrolle zu E-H3 besteht nur aus 2560 und wird so berichtet.
  2. **Manifest:** Caches aller fünf Erweiterungsmodelle (calib + val jeder Trainingsgröße; vorher nur YOLO11m / YOLO12m) sowie die YOLOv10s-B-FULL-2048-Caches der abgeschlossenen Studie (Gewichte von R2 / X2; Nachtrag des Nutzers). Läuft als CPU-Job `e_manifest` nach E2.
  3. **Like-for-like-Rechenkontrolle (exploratorisch, nur CPU, nach E2):**
     - Das eingefrorene X-grid-Verfahren unverändert (gleiches Raster, gleiche 2×-Zeitgrenze, Auswahl auf calib) auf die Caches von YOLOv10s B-FULL 2048 der abgeschlossenen Studie (Gewichte von R2 / X2), Zeilen B und C, beide Zweige wie YOLOv10m. Referenzen: A = R2; B / C = F2 mit den präregistrierten Parametern, hier neu gerechnet (C o2m muss X2 reproduzieren). Code: `E.weights("yolov10s", 2048)` zeigt auf diese Gewichte; `sizes("yolov10s")` bleibt (2560,).
     - Vergleich auf val in `scripts/ext_r3_compare.py` → `runs/ext/r3_compare.json`: R2, X2, X-grid B/C, R3 mit gepaarten CIs gegen R3, MP und GFLOPs je Bild (exakt je Eingabeform), E2-Latenzen.
     - **Latenz:** Die YOLOv10s-X-grid-Zeilen werden in einem Zusatzlauf gemessen (`e_latency_ext_v10s` → `runs/ext/e2/latency_supplement.json`): gleiche Maschine, Prozedur und Leistungsgrenze, CPU-Queue pausiert. Ankerzeilen des Hauptlaufs (R2, X2, R3, XG_yolov10m_2048_o2m_C) werden im selben Lauf erneut gemessen und zeigen die Drift. Der Hauptlauf bleibt unverändert.
     - Die YOLOv10s-X-grid-Zeilen kommen über `xgrid_frozen.json` als exploratorische test-dev-Zeilen in FREEZE_PROPOSAL_EXT. Das Cache-Replay (`e_replay_cache`) prüft sie mit.
  - Reihenfolge: CPU-Queue (`e_xgrid_yolov10s` → `e_r3_compare` → `e_manifest`) startet, sobald E2 sie freigibt. GPU-Queue: `e_replay_predictor` → `e_latency_ext_v10s` (wartet auf die leere CPU-Queue) → `e_replay_cache` → `e_r3_compare_final`.

- 2026-10-29 ~18:30: **Nutzerentscheide zum X-grid-Raster, zu GFLOPs und zum Bericht.**
  1. **Zeile B bleibt für VisDrone unverändert** (keine weitere Post-hoc-Runde). Zeile B lief bei allen Gewichtssätzen, auch YOLOv10s 2048, nur mit dem Raster der ersten Runde: cell_factor 7–13 × bins 6–8 × **nur gesenkte Schwellen**, 48 Varianten plus F2-Referenz. Zeile C lief mit 108 Varianten (Referenz- und gesenkte Schwellen). Die X-grid-Werte von Zeile B sind daher **wahrscheinlich eine untere Schranke** ihres Potenzials. So steht es im E4-Bericht und in FOR_CLAUDE.
  2. **DOTA-Präregistrierung (Kandidat, ersetzt das Raster im Eintrag ~15:05 Punkt 4):** Zeilen B und C nutzen **dasselbe vollständige Raster**: cell_factor {7, 9, 11, 13} × bins_per_oct {6, 7, 8} × τ_splat {0,005; 0,00125; 0,0003125} × τ_peak {0,001; 1e-4; 1e-5} (108 Varianten) plus F2-Referenz, iou_cluster 0,5, Zeitgrenze 2× F2 (im selben Prozess gemessen), Auswahl = max. AP auf DOTA-calib unter der Zeitgrenze.
  3. **GFLOPs:** Abweichung bei R2 klären (128 nach der Methode „exakt je Eingabeform“ gegenüber 143,6 in der metrics.json der abgeschlossenen Studie). Danach GFLOPs für **alle** Zeilen der Hauptstudie und der Erweiterung mit einer Methode neu rechnen (exakt je Eingabeform, Kacheln eingeschlossen) und die Methode im E4-Bericht nennen. Alte Werte bleiben archiviert, kommen aber in keine Tabelle.
  4. **E4-Bericht / FOR_CLAUDE:** Tabelle zur YOLOv10s-Rechenkontrolle (R2, X2, X-grid B/C, R3) mit den E2-Latenzen, sobald gemessen, dazu Pareto-Plots AP vs. GFLOPs und AP vs. Latenz. Vermerk: R3 3072 passte nicht in den Speicher, eine Grenze des Einzelpass-Wegs.

- 2026-10-29 ~20:10: **Freeze der Erweiterung genehmigt (Nutzer), mit zwei Ergänzungen vor test-dev.**
  - Genehmigt aus dem Archiv `scahi_ext_exchange_20261029_712b91877521_pre-test-dev.zip`.
  - **Berichtsregeln** stehen fest in `configs/ext/frozen_rows_ext.json` (`reporting_rules`) und in `docs/FREEZE_PROPOSAL_EXT.md`. Nach dem Lauf werden sie nicht mehr geändert:
    - E-H1–E-H4 bleiben auf val wie berechnet. Auf test-dev werden sie genauso ausgewertet (Definitionen, Zweige, CIs, Holm, „≥ 4 von 5“ konservativ) und als Bestätigung neben val berichtet.
    - Jede eingefrorene Zeile wird berichtet. Kein Nachtunen, keine Neuauswahl, keine zusätzlichen Zeilen.
    - Primär visdrone_ignore (Toolkit-Port, Cap 500), plain als Nebenspalte.
    - Konsistenztabelle val/test-dev je Zeile: AP, Rang, Urteil.
    - Größenklassen prominent (AP und AP50, auch < 8 px und 8–16 px).
    - Exploratorische Zeilen (X-grid, Zerlegung B, R3-Rechenkontrolle) in eigenen Tabellen, nie in einem Urteil.
    - Deskriptiv: crowded- und isolated-Recall für A, B, C (F2) und X-grid; dup- und frag-Raten.
  - **Zerlegungszeile B** (Einzelpass + F2 mit den calib-getunten F2-Parametern der zugehörigen Zeile C, keine neue Auswahl) kommt als exploratorische eingefrorene Zeile hinzu: 11 Zeilen `D_*_single*_F2_*` und YOLOv10s 2048 o2m (Zeile C = X2, gleiche Parameter geprüft). Die auf B nachgetunten Varianten „F2tuned“ bleiben draußen, weil sie eine neue Auswahl wären.
  - **Replay-Gate:** alle 77 Zeilen exakt (max |ΔAP| = 0).

## 2026-10-29 ~22:00 – VisDrone test-dev run once (freeze e766d120)

- All 77 frozen rows evaluated once on test-dev (caches 20:11–20:40, predictor rows until 21:31, fusion until 21:20, analysis 21:56); no retuning, no re-selection, no added rows.
- Verdicts with the val logic (ε = 0.30 from val): E-H1 4/5, E-H2 s=1280 5/5, E-H2 s=2048 4/5 — identical to val. E-H3: val 3/5 (does not hold) vs. test-dev 4/5 (YOLOv10m +0.51 [+0.35; +0.65] on test-dev vs. +0.26 [+0.01; +0.51] on val). Under reporting rule 1 the pre-registered verdict is the val one; test-dev is reported next to it as non-confirming, no re-decision. E-H4 (no threshold): +0.23 [+0.14; +0.31] on test-dev.
- Spearman val ↔ test-dev over the pre-registered rows: 0.964. Reports: reports/TESTDEV_EXT.md, runs/ext/e4/e4_testdev.json, runs/ext/testdev_analysis.json.

## 2026-10-29 ~22:45 – exploratory test-dev contrasts (user request after the test-dev results; post hoc, no verdict changes)

- Added (CPU only): paired bootstrap contrasts per weight set on val and test-dev, same procedure as the verdicts (scripts/ext_testdev_explore.py → runs/ext/testdev_explore.json, TESTDEV_EXT.md §6): X-grid C − F2 C, C − A, B − A; decomposition B − A, C − B. Recomputed val deltas identical to the stored ones (61/61). YOLOv10s o2o C − F2 C has no test-dev value (the o2o F2 C reference was never frozen); for o2m X2 stands in (val-identical).
- Size bins (AP, AP50, all six bins) of the X-grid and decomposition-B rows on test-dev added to TESTDEV_EXT.md §3; GFLOPs of D_yolov10s_single2048_F2_o2m filled (128, the same single 2048 pass as R2; network only, as for every row).
- Descriptive (verified against the numbers):
  - The X-grid gain over F2 (C − F2 C) on test-dev is about half its val value for every testable weight set: ratio 0.49–0.59 over 12 sets (val +1.18 … +1.38, test-dev +0.61 … +0.81 AP; all CIs exclude zero on both splits).
  - Gain of tiles over R3 (YOLOv10s 2048 weights): X2 − R3 −0.17 (val) → +0.55 [+0.29; +0.79] (test-dev); X-grid C − R3 +1.21 → +1.30 (o2m) and +1.06 → +1.08 (o2o). So clearly larger for X2, only marginally for X-grid C. The single pass 2048 also closes on R3 (R2 − R3 −1.03 → −0.51), i.e. R3's lead over the 2048 rows shrinks on test-dev in general.
  - test-dev has about twice the share of GT < 8 px (13.0 % vs. 6.0 % on val; 8–16 px 23.4 % vs. 24.8 %). This is a co-occurrence, not a tested cause.
  - Not part of the request, for completeness: C − A of the X-grid is mostly smaller on test-dev (ratio 0.65–1.05), and decomposition B − A (F2 alone, no tiles) is positive on test-dev for YOLO11/12 (+0.93 … +1.12, val +0.04 … +0.41) and less negative for YOLOv10 (−0.36 … −0.76, val −0.89 … −1.37).

## 2026-10-30 ~08:00 – migration manifest updated early (user: "mach dies jetzt")

- scripts/migration/manifest.py now also lists the test-dev cache levels of every cache row of the extension freeze (scahi.ext_testdev.cache_plan). runs/migration/manifest.tsv: 55 869 files, 16.58 GiB (cache 55 040 files / 12.26 GiB, of which test-dev 32 200 files). Nothing is running; on Sunday only STOP, commit, git bundle and HANDOFF remain (re-run the manifest then; unchanged files are not re-hashed).

## 2026-10-30 ~08:20 – Phase D (DOTA v1.5) started on host-b (user: docs/DOTA_BRIEF.md)

- Brief committed as `docs/DOTA_BRIEF.md`; on conflict it wins. Conflicts with earlier entries, resolved in favour of the brief:
  1. Images (2026-10-27 ~16:30 / ~18:20: Ultralytics images "only if the audit shows equivalence with the originals"): re-encoded images are **not blocking**; they are recorded as a limitation (all methods see the same images).
  2. Budget (≈ 20 GPU-h earlier): up to **30 GPU-h** approved without asking; forecast after the smoke tests.
  3. HBB source: official HBB files (already decided 2026-10-28 ~08:45; OBB only for the audit) – no conflict.
- VisDrone stays frozen; nothing in Phase D touches VisDrone results. Stops: (a) blocking audit problem, (b) PREREGISTRATION_DOTA + cost forecast, (c) freeze proposal before DOTA-val.

## 2026-10-30 ~08:30 – Phase D0 done (audit, evaluation port, split)

- Data unpacked to `/path/to/datasets/DOTA/unpacked` (configs/data/dota.yaml; host-b override in configs/local): `DOTAv1.5/` (Ultralytics images), `hbb/<split>` (official HBB, label source), `obb/<split>` (audit only). No Ultralytics conversion.
- Audit `reports/D_audit.md`: **no blocking problem**. 1411 / 458 images, all paired; JPEG, IJG quality 70 (originals are PNG → re-encoded, limitation per brief); resolution unchanged. 210 631 / 69 565 instances (2.19× v1.0 → v1.5 labels), difficult flag present (45 % / 51 %). Official HBB = min/max of the OBB corners for every box. **All objects < 8 px and most 8–16 px are flagged difficult** and are therefore ignored by the official evaluation (input for the preregistration). One train image (P1872) does not match its annotation (84/231 boxes up to 7749 px outside, also in Ultralytics' labels) → excluded from train_main and calib; P0334: 2 boxes ≤ 121 px outside → clipped for training. Detection cap = 2 × p99 (1814) → **4000** per image.
- Evaluation port `scahi/dota/evaluate.py` of DOTA_devkit task 2 (reference copy `third_party/DOTA_devkit/`, commit 9938855): identical to the original on synthetic and real-looking detections incl. a cap and the full val set (|Δ mAP| ≤ 6e-17; `runs/dota/eval_validation.json`), unit tests `tests/test_dota_eval.py`. 16 classes (devkit lists the 15 of v1.0; v1.5 adds container-crane). Secondary metrics use the same devkit matching (AP@[.5:.95] with COCO 101-point interpolation; size bins with COCO area-range ignore rule).
- Split `data/splits/dota/` (seed 20261030): calib 141 / train_main 1269 images, stratified by objects per image × median object size; no scene grouping possible (file names carry no scene id). Every class has non-difficult GT in calib (helicopter 4 instances, container-crane 11 in one image: calib AP of these two is noisy).

## 2026-10-30 ~08:45 – resolution crossing, algorithm spec, release (user tasks of the day; DOTA keeps priority)

- **Train/inference resolution crossing** (VisDrone val, exploratory, post hoc; `scripts/ext_res_crossing.py`, `runs/ext/res_crossing.json`): YOLOv10s weights of 2048 (R2) and 2560 (R3) at 2048 / 2560 / 3072 / 3584 px, o2o, conf 0.001 (calib-tuned at the training size). AP: R2 weights 38.47 / 38.84 / 38.08 / 36.91; R3 weights 38.44 / 39.51 / 39.64 / 39.34 (the diagonal reproduces the frozen rows exactly). Peak inference memory 0.3 / 0.6 / 1.0 / 1.8 GB. **Training memory:** the recipe batch 3 OOM at 3072 (addendum 0f6fecbc), but **batch 2 fits at 3072 (25.3 GB) and at 3584 (26.7 GB)** (smoke, 120 iterations, random batches; not a worst-case density test). So the limit of the compute control was the fixed recipe batch, not memory as such; the inference limit lies far above. Latencies informational (CPU work in parallel).
- **Algorithm specification** `docs/SCAHI_ALGORITHM.md`, generated from the code and the frozen configurations by `scripts/scahi_algorithm_doc.py`.
- **SCAHI cell keys:** 13 → 15 bits per spatial index for images whose long side exceeds 8189 px (DOTA, up to 13 383 px); VisDrone keeps 13 bits. Checked: four frozen VisDrone SCAHI rows (pre-specified and refined grid, B and C) reproduce their val detections bit-identically; all unit tests pass.
- **Release** (user: code link in the conference submission): tooling and first build, see `reports/RELEASE.md` (0 forbidden terms, clean-room test passed). Licence draft AGPL-3.0 (Ultralytics dependency) – decision with the user.

## 2026-10-30 ~08:50 – Stopp b: PREREGISTRATION_DOTA freigegeben mit Addendum 1 (Nutzer)

- Freigabe mit Änderungen, festgehalten als Addendum 1 in `docs/PREREGISTRATION_DOTA.md` (vor dem ersten Training): primäre Metrik = offizielle mAP50 **ohne** Cap (Cap 4000 sekundär; Pipeline-Grenzen 20 000 je Bild/Pass, dürfen nicht greifen); vorab festgelegte Sekundäranalyse „alle Objekte“ (difficult als normale GT) für D-H3 mit KI und Größenbins < 8 / 8–16 px; S_max = 2048 für beide Modelle (DOTA: echte Speichergrenze durch Objektdichte, keine Rezeptänderung); Inferenz-Kreuzung der 2048er Gewichte bei 2048 / 2560 / 3072 (beste auf calib, sekundär neben D-H3); z = 2 gestrichen; Holm-Regel wie im Entwurf.
- Dichte Bilder (> 4000 GT): calib 0/141, DOTA-val 1/458 (12,4 % der val-GT), train_main 2/1269 (8,0 %).
- Training mit difficult-GT als Positive bestätigt; nur zwei Boxen von P0334 fehlen in den Labels (vollständig außerhalb des Bildes).
- **P1872:** Ausschluss aus train_main und calib vom Nutzer bestätigt (Audit `reports/D_audit.md`, Eintrag 2026-10-30 ~08:30).
- Modellauswahl auf calib nutzt jetzt die Primärmetrik ohne Cap (`scahi/dota/train.py`); Prädiktor `max_det` 20 000 mit Zählung der Bilder an der Grenze.
- **VisDrone (deskriptiv, Nutzer):** Die Trainingsgrenze der VisDrone-Rechenkontrolle war die feste Rezept-Batch, nicht der Speicher (Batch 2 passt bei 3072 und 3584 px). Inferenz bei 3072 px mit den 2560er Gewichten ergibt 39,64 AP auf val (bester Einzelpass der Kreuzung). Das öffnet R3 bei 3072 nicht wieder; ein bei 3072 trainierter Einzelpass (Batch 2) ist ein Kandidat für einen explorativen Lauf nach DOTA, nur nach Freigabe durch den Nutzer.
- **Konsistenz:** Die vorab festgelegte SCAHI-Einstellung (F2) nutzt nur den Rand-Taper (kein Validity-Fenster) – so in `docs/SCAHI_ALGORITHM.md`, `release/GLOSSARY.md`, `release/README.md` und `README.md`.

## 2026-10-30 ~12:00 – DOTA D3/D4: Implementierungsdetails der Fusion (keine Methodenänderung)

- **SAHI-Backend der Box-Fusion: numba (CPU), fest.** VisDrone lief mit SAHIs „auto“, das auf einer GPU-Maschine torchvision wählt. Auf DOTA lief torchvision neben dem Training aus dem Speicher (IoS: volle N × N-Matrix auf der GPU). Gleicher Greedy-Algorithmus; numba gegen numpy auf 40 calib-Bildern identisch. **Vergleich numba gegen torchvision (11:54, GPU sonst frei, `d_backend_cmp`, `runs/dota/fusion/backend_compare.json`):** YOLOv10s 1024, erste 40 calib-Bilder, alle 12 NMS-Konfigurationen × beide Köpfe: behaltene Boxen in 40/40 Bildern identisch für alle 24 Kombinationen, |ΔmAP50| = 0 (torchvision auf CUDA, Spitze 120 MB). Hinweis aus SAHIs Code: torchvision rechnet IoS nur bei < 2000 Boxen je Klasse auf der GPU (float32), darüber den numpy-Sparse-Pfad auf der CPU; numba rechnet in float64.
- **NMS je Pass (o2m) für DOTA:** Ultralytics' `non_max_suppression` mit `max_wh = 2^15` (globale Koordinaten bis 13 383 px > Standard-Klassenversatz 7680) und ohne Zeitlimit. VisDrone unverändert.
- **SCAHI mit 16 Klassen:** `fuse_image(..., nc=16)` statt des VisDrone-Standards 10 (sonst sofortiger Abbruch).
- **Laufzeitfehler behoben (12:00):** Der erste calib-Tuning-Job (YOLOv10s 1024) hing ~1,5 h in der o2m-Box-Fusion. Ursache: Ultralytics nutzt den torchvision-NMS-Kernel nur, wenn torchvision schon importiert ist, sonst die reine torch-Variante `TorchNMS.nms`. Dazu kam die fehlende Threadgrenze (7 Worker × 24 torch-Threads auf 8 Kernen). Geprüft: `TorchNMS.nms` und `torchvision.ops.nms` behalten auf 512 o2m-Kacheln (29 calib-Bilder, 539 267 Kandidaten) identische Boxen. Fix: `scahi/infer/cache_mz.pass_detections` importiert torchvision; die DOTA-Worker nutzen 1 torch-Thread (wie die VisDrone-Fusion mit `OMP_NUM_THREADS=1`). Ergebnisse unverändert, NMS_o2o für YOLOv10s 2048 jetzt 11 s statt 178 s. Der Job YOLOv10s 1024 wurde abgebrochen und wird wiederholt (fertige Stufe NMS_o2o bleibt).
- **Pipeline-Grenze 20 000 greift bei SCAHI (12:15, nur calib):** Im calib-Tuning von YOLOv10s 2048 erreichen SCAHI-Konfigurationen die Grenze (`top_k` = 20 000) auf 1–5 calib-Bildern (F2: 3 von 24 Konfigurationen; verfeinertes Raster: 109/109 in Zeile C, 108/109 in Zeile B), auch die ausgewählten (F2 1, RGC 2, RGB 1 Bild); die Box-Fusion nie. **Korrektur 13:00:** Auch der Einzelpass (o2o) bei 2048 erreicht die Grenze auf 1–2 calib-Bildern (YOLOv10s 2, YOLOv10m 1; auch bei 2560 / 3072); mit max_det 200 000 ist seine calib-mAP50 identisch (73,695 / 72,021; ohne Grenze höchstens 25 287 Detektionen je Bild). Prüfung mit den ausgewählten Konfigurationen und Grenze 100 000 bzw. 400 000: calib-mAP50 und mAP50 (Cap 4000) auf drei Nachkommastellen identisch (F2 75,727; RGC 76,017; RGB 74,901), ohne Grenze höchstens 34 350 Detektionen je Bild. Das Addendum verlangt, dass die Grenze nicht greift; ob sie für die DOTA-val-Läufe angehoben wird (Vorschlag 100 000), entscheidet der Nutzer im Freeze-Vorschlag (Stopp c). Kein Tuning wird deswegen wiederholt.
- **Gemessene Kosten (Prognose-Nachtrag zu PREREGISTRATION_DOTA §6, wie bei der Freigabe verlangt):** CPU-Tuning je Gewichtssatz (7 Worker, Kerne 16–23) 16–23 min (Box-Fusion je 11–18 s, F2 55–79 s, verfeinertes Raster Zeile C 606–1024 s, Zeile B 244–268 s); SAHI-Tuning je Modell 5–6 min; Replay aller Zeilen 7 min (CPU) bzw. < 1 min (GPU). GPU bis zum Freeze ≈ 3 h (vier Trainings 2,6 h, Gates / Einzelpässe / Caches 0,25 h, SAHI-Rohvorhersagen 0,07 h). Freeze-Vorschlag 13:40 (`docs/FREEZE_PROPOSAL_DOTA.md`), Stopp c.

## 2026-10-30 ~15:10 – Stopp c freigegeben (Nutzer); Pipeline-Grenze 200 000; Plausibilitätscheck 2048 (vor DOTA-val)

- **Freigabe** von `docs/FREEZE_PROPOSAL_DOTA.md` (32 Zeilen, Berichtsregeln) mit **Option 1**: interne Detektionsgrenze für alle Zeilen auf DOTA-val **200 000** je Bild/Pass. Das setzt die präregistrierte Primärmetrik „ohne Cap“ um; die Grenze 20 000 war technisch, nicht präregistriert. Auswahl bleibt eingefroren (bei 20 000 getroffen), kein Nachtuning. `configs/dota/frozen_rows.json`: `approved: true`, `pipeline_limit: 200000`, `calib_selection_limit: 20000`.
- **Neue Replay-Baseline** (calib, Auswertungspfad von DOTA-val, Grenze 200 000; `runs/dota/freeze/replay_baseline_{cpu,gpu}.json`): 28/32 Zeilen identisch zur calib-Auswahl, kein Bild an der Grenze; abweichend nur RG_yolov10s_1024 70,1081 → 70,1081 (−7,1e-5), RG_yolov10s_2048 76,0173 → 76,0172 (−7,6e-5), B-RG_yolov10s_2048 74,9014 → 74,9012 (−1,1e-4), RG_yolov10m_1024 72,1764 → 72,1761 (−3,4e-4) mAP50-Punkte. Identisch mit dem Replay im Freeze-Vorschlag. Auf DOTA-val wird je Zeile die Zahl der Bilder an der Grenze 200 000 berichtet (erwartet: keines).
- **Plausibilitätscheck der 2048er Trainings (nur Trainingslogs und calib; Anlass: v10m 2048 < v10s 2048 auf calib, bei 1024 umgekehrt): kein technischer Fehler.**
  - Batch 3 in allen 60 Epochen (423 Iterationen = 1269 Bilder / 3; Batch-Wache nie ausgelöst, kein OOM-Fallback, kein Resume); effektive Batch 63 (Ultralytics nbs 64, Akkumulation 21; bei 1024: 8 × 8 = 64). Spitzenspeicher (reserviert) v10s 29,5 GB, v10m 29,6 GB von 32 GB. Keine NaN/Inf, keine übersprungenen Batches.
  - **Trainingsbedingung (für Supplement/Bericht):** Bei 2048 meldet der CUDA-Allokator in jeder der 60 Epochen fehlgeschlagene Allokationen (insgesamt v10s 423, v10m 1008 Meldungen auf je 25 380 Iterationen; bei 1024: 10 bzw. 26). Sie werden entweder vom Allokator selbst (Cache freigeben, erneut versuchen) oder von Ultralytics' OOM-Rückfall der Label-Zuordnung (`TaskAlignedAssigner`: dieselbe Zuordnung Bild für Bild auf der GPU, Forward-Batch unverändert; je Instanz nur einmal geloggt, daher nicht zählbar) abgefangen. Laut Code ist die Zuordnung identisch, die Ergebnisse sind nicht betroffen; ein nicht abgefangener OOM nach Epoche 1 hätte den Lauf beendet.
  - calib-Kurven (Epoche: mAP50): beste Epoche v10s 47 (73,70), v10m 48 (72,02); Mittel ± SD der Epochen 41–60: v10s 2048 71,6 ± 1,5, v10m 2048 70,1 ± 1,2 (v10s 1024 63,2 ± 0,7; v10m 1024 67,4 ± 0,4).
  - **Befund:** v10m liegt bei 2048 über die ganze Spätphase rund 1,6 Punkte unter v10s (calib, 141 Bilder); die 2048er Kurven schwanken stärker. Mögliche Erklärungen, **nicht geprüft**: Batch 3 (Batch-Norm-Statistik) trifft das größere Modell stärker; Kapazität/Überanpassung von v10m bei 2048; Rauschen der kleinen calib-Menge.
- Ablauf DOTA-val (einmal): `d_val_gpu` (Einzelpass-Zeilen, X = A, weil die Kreuzung 2048 wählte; SAHI-Rohvorhersagen), `d_val_cpu` (Cache-Zeilen, SAHI-Merge), `d_val_eval` (Metriken, gepaarter Bootstrap 1000 / Seed 0, Holm über 2 Modelle), danach `d_val_latency` exklusiv bei 575 W. „D-H1/D-H2 mit dem anderen Kopf“: A ist o2o-, RG o2m-gebunden; nur der Box-Fusions-Arm wechselt den Kopf (sekundär).

## 2026-10-30 ~16:30 – DOTA-val einmal ausgewertet (nach Freigabe Stopp c)

- Lauf 15:17–16:27 (Code-Commit 2251d77): alle 32 eingefrorenen Zeilen, Grenze 200 000; Ergebnisse `runs/dota/val/results.json`, Bericht `reports/D_RESULTS.md` (`scripts/dota/d_results.py`), Abbildungen `reports/figures/dota/`. Kein Nachtuning, keine Neuauswahl.
- Primär (mAP50 ohne Cap): **D-H1 gilt nicht** (YOLOv10s +2,31 [−0,31; +4,95], Holm-p 0,060; YOLOv10m erfüllt); **D-H2a, D-H2b und D-H3 gelten** (beide Modelle; D-H3: YOLOv10s +5,50 [+3,55; +7,77], YOLOv10m +4,34 [+3,02; +6,19]).
- **Pipeline-Grenze:** 1 Bild an der Grenze 200 000 (P1384, 304 GT, Zeile RG_yolov10m_2048; erwartet: keines). Explorativ, post hoc (`scripts/dota/d_limit_check.py`): ohne Grenze 215 568 Detektionen auf diesem Bild, mAP50 der Zeile identisch (Δ = 0).
- Ablauf: `d_val_cpu` brach im ersten Versuch wie vorgesehen ab, weil die SAHI-Rohvorhersagen für val noch auf der GPU liefen; der Wiederholungslauf schrieb nur die fehlenden Zeilen (Detektionen deterministisch, je Zeile einmal geschrieben). Latenz der D-H3-Zeilen danach exklusiv (GPU und CPU frei, 575 W; Status final; Parität der Live-Pipeline auf 3 calib-Bildern).

## 2026-10-30 ~18:30 – Qualitative DOTA-Abbildung: Regel vor dem ersten Rendern (Nutzerauftrag; explorativ, post hoc)

- **Methoden:** YOLOv10s, 2048er Gewichte, dieselben Pässe (dota2048: ganzes Bild bei 2048 + native 1024er Kacheln), Kopf o2m: „Kacheln + Box-Fusion“ (eingefrorene Zeile NMS-o2m_2048) gegen „SCAHI, verfeinertes Raster“ (RG_2048) – das Paar der Fusions-Frage bei 2048 (identische Pässe und Kopf). Detektionen = die eingefrorenen DOTA-val-Detektionen (`runs/dota/val/dets/`), Grenze 200 000.
- **Arbeitspunkt je Methode** (wie VisDrone v2): kleinster Score s mit gepoolter calib-Präzision (Score ≥ s, greedy IoU-0,5-Zuordnung je Bild und Klasse, Treffer auf difficult-GT weder TP noch FP) ≥ 0,80; calib-Detektionen über denselben Pfad (`scahi.dota.final.row_dets`).
- **Bildauswahl (DOTA-val):** Bilder mit ≥ 10 nicht-difficult-GT; je Bild ΔAP = mAP50 (Devkit-Port auf dem einzelnen Bild; Klassen ohne nicht-difficult-GT im Bild fallen aus dem Mittel) von SCAHI-RG minus Box-Fusion; je Quartil von ΔAP das Bild, das dem Median des Quartils am nächsten liegt (4 Bilder); dazu das Bild mit den meisten nicht-difficult-GT der Klassen small vehicle + large vehicle (falls schon gewählt: das nächste). Gleichstand: kleinerer Bildname.
- **Darstellung:** TP grün, FP rot, FN gelb (nicht-difficult-GT ohne Treffer am Arbeitspunkt); difficult-GT und Treffer darauf werden nicht gezeichnet. Ausschnitt je Bild: Fenster von 1/3 × 1/3 des Bildes mit den meisten nicht-difficult-GT-Zentren (Schrittweite 32 px), für beide Methoden derselbe.
- Ausgabe `reports/figures/dota/fig_dota_qualitative.{pdf,png,csv}` + Tabelle ΔAP je Bild; Beschriftung explorativ, post hoc.

## 2026-10-30 ~19:00 – SCAHI-Bausteine: Ablation (explorativ, post hoc; Nutzerauftrag)

- `reports/ABLATION.md`, `scripts/ablation_components.py`, `scripts/ablation_coverage.py`; Daten `runs/ablation/`. Nur CPU aus den Caches; VisDrone val (5 Weight-Sets bei 2048), DOTA-val (beide Modelle bei 2048, Post-hoc-Analyse auf dem bereits ausgewerteten zurückgehaltenen Satz). Nie Grundlage eines Urteils.
- Neue Variante nur für (a): F2n = F2-Gewichte (Rand-Taper) ohne Division durch C (Nutzer bestätigt). Alle eingefrorenen Zeilen unverändert (F2); Kontrolle: Sweep-Punkt cell_factor 5 = (c) reproduziert die bestehenden Zeilen exakt (VisDrone 5/5, DOTA 2/2).
- **Befund Normalisierung:** In den eingefrorenen Stacks ist C an jeder fusionierten Position im Bild genau 2 (F2; DOTA und VisDrone), bei (b) ohne Taper 2–10 (DOTA) bzw. 2–7 (VisDrone). (a) − (c) ≈ 0 (VisDrone +0,001 bis +0,002 AP; DOTA −0,00 mAP50). Die Normalisierung skaliert dort nur; „ein Pass ohne Vorhersage zählt als Evidenz dagegen“ folgt aus der Summe über die Pässe. Wirksam wäre sie nur bei Teilabdeckung, die die eingefrorenen Zeilen nicht enthalten; kein neues Experiment.
- (b) ohne Taper: VisDrone −0,74 bis −1,01 AP, DOTA −1,12 / −0,99 mAP50 (alle KI < 0). (e) verfeinertes Raster: VisDrone +1,32 bis +1,38 AP (Raster post hoc); DOTA +0,36 / +0,41 (Raster auf DOTA vorab festgelegt). cell_factor-Sweep: VisDrone steigend bis 13, DOTA Maximum bei 9.
- (d) Validity-Fenster: VisDrone −1,6 bis −1,8 AP; DOTA −18,7 / −20,4 mAP50 – Fenster auf VisDrone-Größen ausgelegt, nicht nachträglich angepasst; nur der Vollständigkeit halber berichtet, kein Argument (Nutzer).
- DOTA-Abbildungen in Paper-Begriffen (Q1 = D-H1, Q2 = D-H2a/b, Q3 = D-H3, Nutzer bestätigt; Q4 auf DOTA nicht präregistriert); das verfeinerte Raster heißt dort „SCAHI (refined grid, pre-specified on DOTA)“. Qualitative DOTA-Abbildung nach der Regel vom 18:30: Arbeitspunkte calib-Präzision 0,80 (Box-Fusion s ≥ 0,499, SCAHI s ≥ 0,305); Bilder P0199, P0454, P0749, P0683 (Quartile 1–4) und P1478 (meiste Fahrzeuge).

## 2026-10-30 ~20:50 – Rechenkontrolle VisDrone: fehlende Werte der Kreuzung (Nutzer, Punkt 1)

- Zeile „2560er Gewichte bei 3072 px“ (explorativ, post hoc) und alle übrigen Kreuzungszeilen: **GFLOPs** mit der einheitlichen Methode (`scripts/gflops_unified.py`, `_count` auf die metrics.json der Kreuzung → `runs/ext/res_crossing_gflops.json`): identisch mit den bereits gespeicherten Werten (298,36 GFLOPs bei 3072 px). **Latenz** im E2-Verfahren (`scahi.ext_latency`: GPU und CPU exklusiv, 575 W Standard-Leistungsgrenze, Smoke zuerst, bis zu 3 Versuche; Ultralytics-Predictor, Batch 1, wie R2/R3) → `runs/ext/e2/latency_crossing.json`: alle 10 Zeilen final; 2560er Gewichte bei 3072 px **19,1 ms** (2048 / 2560 / 3584 px: 9,8 / 14,0 / 30,6 ms). Anker im selben Lauf: R2 9,87 ms (E2-Hauptlauf 9,75), R3 2560 14,19 ms (13,77). Einige Versuche wurden wegen ~1 Kern Fremdlast (Virenscanner-Dienst) wiederholt. Beides in `runs/ext/res_crossing.json` (`latency_e2_*`, `gflops_unified_per_img`); die früheren informativen Latenzen bleiben als `latency_ms_mean` erhalten.
- Code: `e_latency_ext(..., extra=...)` (zusätzliche Zeilen, bestehendes Verhalten unverändert), `ext_res_crossing.py gflops|latency`.

## 2026-10-30 ~21:20 – Abbildungen in Paper-Begriffen (Nutzer, Punkt 2)

- Neu erzeugt aus vorhandenen Dateien (keine neuen Läufe): alle Erweiterungsabbildungen (`fig_ext_forest`, `_waterfall`, `_recall`, `_xgrid_cost`, `_r3_pareto`, `_scaling`, `_calib_curves`) und alle Hauptstudien-Abbildungen von `scripts/paper_figures.py`, dazu `fig_method` (TikZ) und `fig_qualitative_v2` (gleiche Bilder und Arbeitspunkte wie zuvor). Begriffe: single pass, tiles + box fusion, SCAHI (pre-specified setting), SCAHI, refined grid, SCAHI on the single pass (no tiles); Q1 = E-H1, Q2 = E-H2, Q3 = E-H3, „YOLOv10 head (o2m vs. o2o)“ statt E-H4. Hauptstudie (Begriffe von mir gewählt, zur Bestätigung): F3e → „SCAHI + validity windows“, T2/T4 → „loss-masked / scale-normalised training“, Stacks als Pässe ausgeschrieben („1229 + z2 tiles“ = S4, „1920 + z2 tiles“ = S5, „640 + z1 + z2 tiles“ = S1), R → „tiles + box fusion“, M_ab → „SCAHI + validity windows“, X2 → „SCAHI, 2048 + z2 tiles“.
- `fig_ext_r3_pareto` enthält zusätzlich die Zeile „2560er Gewichte bei 3072 px“ (nur val; explorativ, post hoc). Daten: `.data.json` neben jeder Abbildung, bei den Erweiterungsabbildungen zusätzlich CSV.

## 2026-10-31 ~14:50 – Analyse pro Bild „Wann lohnen sich Kacheln?“ (Nutzer; explorativ, post hoc; Regel vor der ersten Rechnung)

- Nur vorhandene Detektionen der eingefrorenen Zeilen, keine neue Auswertung über diese hinaus. Paare je Detektor und Kopf (gleicher Kopf auf beiden Seiten): **VisDrone test-dev** Einzelpass 2048 (`E_<m>_full2048_<h>`; YOLOv10s: R2, o2o) gegen Kacheln + SCAHI, verfeinertes Raster 2048 (`XG_<m>_mz2048_…_<h>`): YOLOv10s o2o, YOLOv10m o2m und o2o, YOLO11s, YOLO11m, YOLO12s (o2m); YOLO12m hat keine 2048er Zeilen. **DOTA-val** `A_<m>_2048` gegen `RG_<m>_2048`, YOLOv10s und YOLOv10m.
- AP pro Bild: VisDrone wie `fig_qualitative_v2` (Toolkit-Port, AP@[.5:.95], Ignore-Regionen, Cap 500, `bootstrap.port_ap` mit Einheitsgewicht auf dem Bild); DOTA wie `d_qualitative.py` (Devkit-Port, mAP50 des Bildes, difficult ignoriert). ΔAP = SCAHI − Einzelpass. Bilder ohne gültiges GT (nicht ignoriert / nicht difficult) oder mit undefiniertem AP entfallen.
- Kovariaten je Bild über das gültige GT: Anteil mit sqrt(w·h) < 16 px; Anteil „crowded“ = Nachbar derselben Klasse mit IoU > 0,3 (Definition aus `ext_posthoc.py` bzw. `dota/final.crowd_gt`).
- Statistik: Spearman ρ(ΔAP, Anteil) je Paar, 95-%-Perzentil-CI aus 2000 Bild-Resamples (Seed 0). Abbildung: Mittelwert von ΔAP in festen Klassen des Anteils [0, 0,1), [0,1, 0,25), [0,25, 0,5), [0,5, 0,75), [0,75, 1] mit Bootstrap-CI; Klassen mit < 10 Bildern werden nicht gezeichnet (Werte trotzdem in der CSV).
- Ergebnis (`reports/tiles_payoff/TILES_PAYOFF.md`, `reports/figures/ext/fig_ext_tiles_payoff.*`): DOTA-val ρ(ΔAP50, Anteil < 16 px) = +0,21 [+0,11, +0,30] (YOLOv10s) und +0,15 [+0,06, +0,24] (YOLOv10m), crowded ≈ 0; VisDrone test-dev |ρ| ≤ 0,09 in beiden Richtungen (o2o-Zeilen leicht positiv, o2m-Zeilen leicht negativ).

## 2026-10-31 ~15:00 – Latenz-Aufteilung, DOTA-Tabellen (Nutzer; explorativ)

- `scripts/latency_breakdown.py`: ein exklusiver Lauf auf host-b (575 W, E2-Verfahren, Smoke zuerst, ≤ 3 Versuche) für VisDrone YOLOv10s (Einzelpass 2048 / 2560, Kacheln + SCAHI vorab festgelegt und verfeinertes Raster) und die DOTA-D-H3-Zeilen; je Bild Pässe (Referenzpfad), SCAHI CPU und SCAHI GPU auf denselben Pass-Ausgaben → `reports/tab_latency_breakdown.{md,csv}`, `runs/latency_breakdown/latency.json` (alle Zeilen final). Ein erster Lauf (`latency_run1_not_all_final.json`) wird nicht verwendet: Desktop-Fremdlast 1–2,5 Kerne, und der DOTA-Einzelpass rief die Exklusivitätsprüfung nicht auf (behoben).
- **Parität GPU- vs. CPU-Fusion auf allen val-Bildern** (`parity.json`): verfeinertes Raster YOLOv10s identisch (249 999 Detektionen); X2 **1 von 217 492 Detektionen** weicht ab (ein Bild, Box bis 1,9 px bei gleichem Score und gleicher Klasse; vermutlich Cluster-Zugehörigkeit an der Grenze: KD-Baum vs. Chebyshev-Schranke; der frühere E2-Paritätstest prüfte nur 40 Bilder). **DOTA: GPU-Fusion nicht verfügbar** – die dichte Zugehörigkeitsmatrix Spitzen × Vorhersagen bräuchte bei top_k 200 000 ≈ 90 GiB; eine Anpassung wäre eine neue GPU-Implementierung (nur mit Freigabe). Bis zum Abbruch (P1384) waren die DOTA-Ergebnisse identisch.
- `scalespace_torch`: Zellschlüssel wie im CPU-Kern über `key_bits(W, H)` (13 Bit unverändert, 15 Bit nur bei langer Seite > 8189 px); VisDrone bitgleich.
- DOTA-Tabellen aus vorhandenen Auswertungen: AP50 je Klasse (`reports/dota/tab_dota_perclass_map50.md`; Box-Fusion = NMS auf o2m wie in Q2) und D-H3 unter vier Metriken mit gepaarten KIs (`reports/dota/tab_dota_dh3_metrics.md`).
- `fig_method`: Validitäts-Panel entfernt (nicht Teil von SCAHI), w = b̂_t, C = Anzahl abdeckender Zoomstufen, Cap VisDrone 500 / DOTA keiner.
- Migrationsmanifest um alle DOTA-Artefakte (Gewichte, konvertierte Daten, Caches) und alle Ergebnisdateien unter runs/dota, runs/ablation, runs/ext, runs/testdev erweitert (77 045 Dateien, 28,7 GiB, vorgehasht).

## 2026-10-31 ~15:55 – DOTA Addendum 2: Speicher-Smoke YOLO11s / YOLO12s, S_max nach der präregistrierten Regel

- `runs/dota/smoke.json` (48 dichteste train_main-Bilder): YOLO11s passt bei 1024 / Batch 8 und 2048 / Batch 3, nicht bei 3072 (Batch 3 und 2 → Ultralytics halbiert auf 1) → **S_max = 2048**. YOLO12s passt bei 1024 / Batch 8, **nicht bei 2048** (Batch 3 und 2) und nicht bei 3072 → **S_max = 1024**.
- Folgen nach Abschnitt 2 / 4 (keine neue Entscheidung): YOLO12s D-H3 = RG_1024 − A_1024; D-H1 (braucht A_2048) und D-H2b (braucht 2048) für YOLO12s nicht prüfbar → nicht erfüllt (Holm-p = 1); Inferenz-Kreuzung für YOLO12s entfällt (keine 2048er Gewichte).
- GPU-Plan damit ≈ 2,5 h (Trainings YOLO11s 1024 / 2048, YOLO12s 1024 nach Smoke-Prognose 0,15 / 0,55 / 0,35 h; Gates, Einzelpässe, Caches, Replay). Queues `configs/queue/dota_new.txt` (GPU) und `dota_new_cpu.txt` (CPU-Tuning, wartet auf die Caches; danach Replay).

## 2026-10-31 ~19:28 – Freigabe Freeze Addendum 2, DOTA-val der neuen Zeilen, Latenz-Nachtrag (Nutzer)

- **Freigabe (Nutzer):** Freeze der 19 neuen Zeilen (YOLO11s, YOLO12s) mit den Berichtsregeln des ursprünglichen DOTA-Freeze: alle Zeilen, kein Nachtuning; primär ohne Cap; sekundär Cap 4000, „alle Objekte“, AP@[.5:.95]; Größenbins; D-H1–D-H3 mit gepaarten KIs, Holm über die zwei neuen Modelle; deskriptive Zählung über vier Modelle; ursprüngliche Urteile unverändert. YOLO12s: S_max = 1024 nach der Regel, D-H1 und D-H2b nicht prüfbar → nicht erfüllt, D-H3 auf dem 1024er Pass. Eingetragen in `configs/dota/frozen_rows_new.json` (`approved: true`, Commit fd0a2c2) **vor** der Auswertung.
- DOTA-val einmal ausgewertet (Queue `configs/queue/dota_val_new.txt`, 19:29–20:05) → `runs/dota/val/results_new.json`, Bericht `reports/D_RESULTS_NEW.md` (`scripts/dota/d_results_new.py`). Ergebnis primär: D-H1 YOLO11s erfüllt (+3,05 [+0,55; +5,81], Holm-p 0,036), YOLO12s nicht prüfbar; D-H2a beide erfüllt (+3,12 / +3,09); D-H2b YOLO11s erfüllt (+3,12), YOLO12s nicht prüfbar; D-H3 beide erfüllt (YOLO11s RG_2048 − A_2048 +4,62 [+2,76; +6,62], YOLO12s RG_1024 − A_1024 +9,62 [+7,10; +12,22]). Zählung über vier Modelle (deskriptiv): D-H1 2, D-H2a 4, D-H2b 3, D-H3 4.
- Feld in `results_new.json` umbenannt (nur Beschriftung, nach der Auswertung): `original_two_model_verdict_unchanged` → `original_two_model_holds`; der Wert war immer „holds“ aus `results.json` (D-H1 = false), der alte Name las sich wie „Urteil geändert“. Code in `final.val_eval_new` entsprechend.
- Latenz-Nachtrag der neuen D-H3-Zeilen auf host-b (`scripts/latency_breakdown.py measure_new`, 20:07–20:17, 575 W, exklusiv, alle Zeilen final; RG_yolov10s im ersten Versuch wegen 1,3 Kernen Fremdlast verworfen und wiederholt): YOLO11s A_2048 26,4 ms, RG_2048 286,3 ms; YOLO12s A_1024 22,2 ms, RG_1024 340,5 ms. Anker im selben Lauf: A_yolov10s_2048 27,8 ms (Hauptlauf 29,7), RG_yolov10s_2048 295,5 ms (306,4). In `reports/tab_latency_breakdown.{md,csv}` ergänzt.
- Begriffe für das Hauptpaper (Nutzer): „single pass“, „tiles + box fusion“, „SCAHI“ (vorab festgelegte Einstellung), „SCAHI, refined grid“, Q1–Q3, „YOLOv10 head“-Vergleich bestätigt. Weitere Begriffe des Eintrags vom 30.10. ~21:20 dem Nutzer zur Bestätigung vorgelegt.
- `fig_method`: Detektorblock generisch („dense detector, e.g. YOLOv10s“); Ausgabe-Cap aus der Abbildung entfernt (Auswertungsdetail, steht im Studiendesign).
- Migrationsmanifest: Caches von YOLO11s (1024, 2048) und YOLO12s (1024) aufgenommen (28 DOTA-Cache-Ordner).

## 2026-10-31 ~20:40 – Begriffe der Hauptstudie (Nutzer, verbindlich für alle Abbildungen und Tabellen)

- Muster: „⟨Pass⟩ + tiles, ⟨Zusammenführung⟩“, z. B. „2048 + tiles, SCAHI“, „1280 + tiles, box fusion“.
- S4 / S5 / S1 nach absolutem Zoom, nicht in Pixeln (Pixelgrößen variieren je Bild; die Stacks sind über den Zoom definiert, `w0.64` / `w1.0` / `w0.333` in `cache_mz.STACK_LEVELS`): „z0.64 + z2 tiles“, „z1 + z2 tiles“, „z1/3 + z1 + z2 tiles“ (S2: „z1/3 + z1 + z2 tiles (1024)“). Die bisherigen Beschriftungen „1229 / 1920 / 640 + …“ waren für Bilder ungleich 1920 px Breite falsch.
- R: „1280 + tiles, box fusion“. F3e: „SCAHI + validity windows (estimated on calib)“. M_ab: „selected main-study configuration (scale-normalised training, SCAHI + validity windows)“.
- Bestätigt: T2 / T4 = „loss-masked training“ / „scale-normalised training“; X2 = „2048 + tiles, SCAHI“; „SCAHI on the single pass (no tiles)“.
- Umgesetzt in `scripts/paper_figures.py` (`STACK_TERM`, `VALIDITY`, `MAB`, Annotationen, Achsenbeschriftungen, Forest-Plot), alle Abbildungen neu erzeugt; in `fig_testdev` (b) ist der M_ab-Name dreizeilig umbrochen, damit die Legende im Panel bleibt.
- DOTA: Bei YOLO12s steht neben dem D-H3-Wert, dass er auf dem 1024er Pass (S_max) gemessen ist (`reports/D_RESULTS_NEW.md`, `reports/dota/tab_dota_dh3_metrics.md`). D-H3-Tabelle und Klassentabelle umfassen jetzt alle vier Modelle (jeweils bei S_max).

## 2026-10-31 ~20:48 – Begriffe auch in den LaTeX-Tabellen der Hauptstudie; Austauscharchiv für das Paper-Update

- Nachtrag zu ~20:40: Die Tabellen `reports/figures/paper/tex/tab_*.tex` (`scripts/paper_tables.py`) trugen noch interne Namen (T4 + S4, Multi-zoom + NMS, F3e, $M_{ab}$, Seed-IDs). Jetzt nach demselben Muster; CSVs mit Spalte `term` neben der internen `id`. Die Symbole $R$ und $R_2$ bleiben als Spaltenbezüge („Δ vs. $R$“), die Zeilen nennen den Begriff.
- **Zur Bestätigung durch den Nutzer (von mir gewählt):** T0 / T1 / T3 = „SAHI-chip training“ / „masked-pixel training“ (Ignore-Regionen grau) / „class-specific loss masks“; Spalten der Fusionstabelle F1 / F3 / F4 / F5 = „uniform“ / „+val. (analyt.)“ / „+val. (analyt.) +unc.“ / „+val. (analyt.) +unc. +temp.“ unter den Gruppen „box fusion“ / „dense fusion“.
- Neue Tabelle `reports/dota/tab_dota_rows_all_models.{md,csv}` (`scripts/dota/d_rows_table.py`): alle 51 DOTA-Zeilen der vier Detektoren in Paper-Begriffen, aus `results.json` / `results_new.json`.
- Austauscharchiv für das Paper-Update: `scripts/make_exchange_zip_paper.py` → `exports/scahi_paper_exchange_<Datum>_<Commit>.zip`; committet wird davon nur `reports/FOR_CLAUDE_PAPER.md`.
- Korrektur (20:52): Die Überschriften der drei Einträge von heute Abend trugen geschätzte statt tatsächlicher Uhrzeiten (~19:35 / ~20:45 / ~21:30); korrigiert nach `approved_at` in `frozen_rows_new.json` (19:28) und den Commit-Zeiten (20:42, 20:49) auf ~19:28 / ~20:40 / ~20:48. Inhalt unverändert.

## 2026-10-31 ~20:52 – Endgültige Begriffe: Trainingsvarianten und Spalten der Fusionstabelle (Nutzer)

- T0 „ignore regions dropped“; T1 „ignore regions painted grey“; T2 „loss-masked training“; T3 „loss-masked + truncation ignore“; T4 „scale-normalised training“. T0–T3 nutzen dieselben Chips und unterscheiden sich nur im Umgang mit Ignore-Regionen.
- Fusionstabelle: F0 „box fusion“ (Gruppe: NMS, GNMM, WBF); F1 „uniform weights“; F2 „SCAHI“; F3 „+ validity (analytic)“; F4 „+ uncertainty“; F5 „+ temperature“; F3–F5 kumulativ auf F2 (steht in der Tabellennotiz). F3e als Spalte „+ validity (calib)“ (= „SCAHI + validity windows (estimated on calib)“; ersetzt die analytischen Fenster, nicht kumulativ zu F3).
- Umgesetzt in `scripts/paper_tables.py` (Tabellen neu erzeugt) und in der Begriffsliste des Archivskripts. Kein neues Austauscharchiv (Nutzer); kommt ins Abschlussarchiv von host-a. Damit sind die am ~20:48 offenen Begriffe entschieden.

## 2026-11-01 ~18:00 – Migration host-b → host-a: checks (task 1 of the host-a brief)

- New brief (user, 2026-11-01): host-a is the only machine from now on; tasks 0–6 in order (orientation, migration checks, GitLab, blocked GPU fusion on DOTA with parity gate, DOTA latency table on host-a, figures / release / archive, FCOS spike). Binding: no new training and no new held-out evaluation (exception: FCOS smoke tests), accuracy results never change, new implementations only after a parity gate, host-a latencies only in complete host-a tables (never mixed with host-b), paper terms everywhere.
- **Environment:** pip freeze identical to `env_export/host-b/pip_freeze.txt` (ultralytics 8.4.183, sahi 0.12.8, torch 2.14.1+cu130, torchvision 0.29.1+cu130); conda env var `YOLO_CONFIG_DIR=<repo>/.ultralytics`; `core.fileMode=false`; branch `master`. host-a: RTX 5090, driver 595.99.02, power limit 575 W, 24 cores, 125 GB RAM.
- **Machine config:** `configs/local/host-a.yaml` (gitignored; VisDrone original / official ZIPs / toolkit under `/path/to/datasets/VisDrone/`, DOTA `/path/to/datasets/DOTA/unpacked`, restricted copy `null`). `scahi.machine` allows only one file in `configs/local/`, so the copied `host-b.yaml` moved unchanged to `env_export/host-b/configs_local_host_b.yaml` (SHA-256 = manifest entry).
- **Manifest:** first `--verify`: 12 158 files missing, all from the DOTA folder `/path/to/datasets/DOTA` (not copied to host-a; only `DOTAv1.5.zip` under `/path/to/DOTA`, SHA-256 correct) and the 1 869 canonical image links pointing into it. The complete folder was on the mirror drive (`<external drive>/Datasets/Datasets/DOTA`, 10 289 files = manifest category "external"); copied with `rsync -a` to the planned target. Second `--verify`: **all 80 764 files OK** (size + SHA-256; the only report was the moved `host-b.yaml`, verified at its new path).
- **Replay gate on host-a** (`scripts/migration/host_a_replay.py`, writes only to `runs/migration/host_a_replay/`, never into frozen result folders): VisDrone val through the frozen extension code path – single pass `E_yolov10m_full2048_o2o` 41.02882344 and refined grid `XG_yolov10s_2048_o2m_C` 40.71666962: |ΔAP| = 0 for both. DOTA calib at the approved limit 200 000 against `runs/dota/freeze/replay_*_limit200000*.json` – `A_yolov10s_2048` 73.69503, `RG_yolov10s_2048` 76.01722, `A_yolo11s_2048` 72.83480, `RG_yolo11s_2048` 74.01494: |Δ mAP50| = 0 for all four, no image at the limit. **Migration checks passed.**

## 2026-11-01 ~20:30 – Task 2 (GitLab) prepared, waiting for the URL; task 3 blocked GPU fusion: parity gate passed

- **GitLab (task 2), before the first push:** `.gitignore` already excluded `data/*` (except `data/splits/`), `runs/`, `cache/`, `weights/`, `*.pt`, `*.npz`, `exports/`, `configs/local/`; added `*.log` (one tracked 865-byte summary `reports/tiles_payoff/compute.log` kept by an explicit exception), `*.parquet`, `*.npy`, `*.onnx`, `*.engine`, `*.zip`, `*.bundle`, `nohup.out`. History scan (408 commits, 661 paths): no weights, caches, raw detections, data or archives ever committed; 157 small result JSONs under `runs/` are force-tracked on purpose (largest `runs/ext/grid_ext.json`, 1.7 MB). Push size (full history, packed): **53.5 MB**. Weights (manifest): 54 files, 3.56 GB, of which `best_calib.pt` 2.95 GB (with optimizer state) and `best.pt` 0.61 GB (stripped); includes the restricted bridge-experiment weights (`BFULL640TNO`, trained on the banned processed copy) – LFS only after the user's decision on scope and instance limits. No remote set (URL from the user).
- **Blocked GPU fusion (task 3):** `scalespace_torch.fuse_image_f2` – membership in spatial blocks of the peak centres (1024 px core, halo = 1.5 · max g_J of the block's peaks + 1 px, chosen per block from the data; exact because membership depends only on the centre distance and σ bin), per-pass maximum over occurring (peak, pass) slots, border windows in point chunks, final sort as in the CPU reference. Report `reports/GPU_FUSION.md`.
- **Parity gate (one-to-one matching, tolerance 1e-4, every image):** DOTA-val 28 SCAHI rows of both freezes, 12 824 image×row, 31.0 M detections; VisDrone val 49 frozen SCAHI rows of the extension freeze + X2, 27 400 image×row, 11.4 M detections: **0 differing images**, max |Δbox| 2e-11 px, |Δscore| 4.4e-16; peak GPU memory 1.82 GiB (DOTA; P1384 previously ≈ 90 GiB) / 0.31 GiB; largest halo 1 148 px (DOTA) / 166 px (VisDrone). Sorted-cell-key fallback not needed.
- Two intermediate findings, both resolved and not caused by the blocking: (1) the first comparator (sorted element-wise) flagged 37 DOTA images that are identical on one-to-one matching (equal-score order); (2) 31 VisDrone images had one detection exchanged at the cap of 500 between two detections with exactly equal score and class – CPU (pandas quicksort) and GPU (stable sort) cut the tie differently; the same applied to the old dense GPU version (its X2 difference). The GPU now uses the CPU's final sort; all 31 identical. Accuracy results unchanged (the GPU fusion is used only for latency).
- **DOTA figures with four detectors (task 5, done early):** `scripts/dota/d_figures.py` reads `results.json` + `results_new.json`; YOLO11s / YOLO12s added (fixed categorical slots 3 / 4 of the reference palette: aqua, yellow, plus one marker shape per model); YOLO12s labelled "1024 px only", Q1 / Q2 (2048) shown as "not testable (no 2048 weights)", its Q3 value marked "1024 pass (S_max)"; rows without a counterpart (o2o box fusion, SAHI for single-head models) left out. The palette validator (`validate_palette.js`) could not run on host-a (no node.js); the four slots are the validated reference order, identity is never colour-only (marker shape + legend).

## 2026-11-01 ~19:30 – GitLab postponed (user)

- Task 2 postponed: no remote, nothing pushed until the user says so. The prepared `.gitignore` stays.
- Weights decision for later (user): only `best.pt` of the paper models go to the remote (via LFS if the instance allows it); no `best_calib.pt` (optimizer state), **no restricted bridge-experiment weights** (`BFULL640TNO`, `BFULL640PV`/`BFULL1280PV` are not paper models either unless the user says otherwise).

## 2026-11-01 ~19:40 – Task 4: DOTA latency table measured on host-a

- A first run from inside VS Code (18:41) was aborted and not used: the VS Code GPU process (type C+G) made every row "informational" (`runs/latency_host_a/dota_latency_aborted_20261101T1906.json`). No whitelist of foreign GPU processes (user). The report step writes a table only if all rows are final and includes GPU fusion times only with a passed parity gate (`runs/gpu_fusion/parity_dota.json`: passed).
- Clean run 19:07:03–19:23:05, started by the user from a plain terminal with VS Code closed (`scripts/run_latency_host_a.sh`): all 10 measurements **final** on the first attempt, 575 W, no foreign compute process at any check, foreign CPU load ≤ 0.98 cores (limit 1.0). Graphics processes (type G) during the run: gnome-shell, Xwayland, Firefox and Nautilus (the latter two still running afterwards per process start times, so present during the whole run; the user's note said they were closed after the start). VS Code's GPU process started 19:28, after the run. Full nvidia-smi lists at start/end were not saved by this run; recorded from the per-check fields and process start times (`runs/latency_host_a/setup.json`); the starter script saves them from now on.
- Table `reports/dota/tab_dota_latency_host_a.{md,csv}` (labelled "measured on host-a", 458 DOTA-val images, batch 1): e.g. YOLOv10s 2048 – single pass 28.7 ms; passes of the tiled stack 137.6 ms; box fusion +45.9 ms (183.5 total); SCAHI refined grid on CPU +158.4 ms (296.0), on GPU +12.2 ms (149.8, peak 1.9 GB); SAHI (1024 weights) 191.6 ms. The tile passes are 91–94 % of the GPU-fusion totals (YOLOv10s 91.8, YOLOv10m 93.6, YOLO11s 90.9, YOLO12s 91.3 %). Never combined with the host-b tables.

## 2026-11-01 ~19:50 – Latency table accepted; go for task 5 and the FCOS spike (user)

- The host-a DOTA latency table is accepted as valid. For the paper: DOTA latencies only from this table (complete, one run, incl. GPU fusion), VisDrone latencies stay from host-b; each table from exactly one machine; both have the same GPU model (RTX 5090, 32 GB, 575 W default) – noted in the table.
- Wording rule: box fusion (numba NMS) runs on the CPU, SCAHI in the GPU rows on the GPU – never presented as a speed comparison; the valid statement is that the tile passes dominate (91–94 % with GPU fusion) and SCAHI on the GPU adds about 12 ms. Added to the table text. Adoption into the paper text is the user's decision.
- Go: task 5 (release, clean-room test, final archive), then the FCOS spike (report, then stop).

## 2026-11-01 ~21:00 – Task 5: release with DOTA code and paper regeneration; clean-room passed

- **Release content (manifest):** DOTA code (`scahi/dota/` without the job module, `third_party/DOTA_devkit/`, DOTA configs and split), blocked GPU fusion + its test, `tests/test_dota_eval.py`, ablation scripts, all generators of the paper tables / figures that are computed from result files, `scripts/regenerate_paper.py`, the result files they read (`configs/release/results.txt`, 159 files ≈ 8 MB: JSON / CSV, 16 empty `TRAIN_DONE` markers; no weights, caches, raw detections, images) and the reference outputs (`expected/`, 62 files). Release 319 files, 12 MB, text only.
- **Code changes for the release (behaviour unchanged, 62/62 outputs byte-identical in the research repo before/after):** helpers `rel`, spawn context, ulp comparison and the env record moved unchanged to `scahi/common.py` (old names import from there), so the DOTA code no longer imports the queue drivers; `scripts/dota/d_results.py` finds `scahi` without an installed package; `fig_ext_scaling` no longer drops rows silently when weights are missing – with weights it writes its points and source list to `runs/ext/scaling_points.json`, without them it re-plots exactly these points, otherwise it stops; a comment mentioning the banned copy reworded.
- **Deviation (for the user to confirm):** `scripts/ablation_components.py` (computation from the dense-head caches) and the measurement modes of `scripts/latency_breakdown.py` import the research drivers `scahi.ext` / `scahi.ext_latency`, which are not in the release; they are shipped as documentation of the computation with a per-file exception list in the export check (`research_only_imports`). Their report modes (used by the regeneration) do not need them.
- **Exporter fixes:** `.csv` and `.tex` are now treated as text (substitutions and the forbidden-term check previously skipped them – one machine name in a CSV would have leaked); line endings kept byte-exact; substitutions also apply to file names (e.g. `latency_host_a` → `latency_host_a`); new substitutions for the repo path, host names and the DOTA data path. Anonymisation list regenerated (`scripts/anonymize_terms.py` → `docs/ANONYMIZE_TERMS.txt`, private; 8 new regexes in the release filter: author name variants, host names, host-name pattern).
- **Lock file:** five packages pinned by local conda build paths (`packaging`, `ply`, `sip`, `toml`, `tomli`) replaced by their versions (same versions as installed); README install line with the PyTorch cu130 index.
- **Clean-room (passed):** fresh conda environment (Python 3.11) from the release's `requirements.lock` (all pins present; extra only pip, setuptools, wheel and PyQt5's Qt wheel); `pip install .` works; tests 69 passed / 7 skipped (need data or weights); all modules from the copy; reproduce CLI help ok; no repo path; regeneration without GPU 14/14 generators, **62/62 outputs byte-identical with `expected/`** (VisDrone and DOTA). Not regenerated (need images / raw detections / caches): qualitative figures, volume figure, method figure, per-image tables. `reports/RELEASE.md` updated (pass/fail per output). Release ready for upload by the user.

## 2026-11-01 ~21:15 – Task 6: FCOS feasibility spike – decision recorded before any step (user)

- Detector: **FCOS from torchvision** (`torchvision.models.detection.fcos_resnet50_fpn`, weights `FCOS_ResNet50_FPN_Weights.COCO_V1`) in the existing environment (torchvision 0.29.1). No build of the official maskrcnn-benchmark repository, no mmdetection.
- Scope of the spike (one working day, then report `reports/FCOS_SPIKE.md` and stop for the user's go/no-go): load; differences of the torchvision variant to the paper (table); dense per-position export (class probability × centerness, decoded boxes) into the SCAHI cache format; decoding parity against torchvision's own post-processing on a few images; raised detection / candidate limits for dense DOTA images; memory smoke for training at 1024 and 2048 px on the densest DOTA images; 2-epoch training smoke on a subset with model selection on calib; recipe proposal fixed in advance; cost estimate (DOTA 1024 / 2048, optionally VisDrone 2048). Only smoke tests; **no full training, no evaluation on held-out data** (VisDrone test-dev, DOTA-val) without the user's explicit approval.

## 2026-11-01 ~22:00 – FCOS spike done, report `reports/FCOS_SPIKE.md`; stop for the user's go / no-go

- torchvision FCOS loads in the existing environment; dense export into the SCAHI cache format (`scahi/fcos.py`: probs = p_c × centerness, branch "o2m", std 0); decoding parity vs. torchvision's post-processing bit-identical on 14/14 images (default and raised limits); limits raised for dense DOTA images (score 0.001, 200 000 candidates per level, 200 000 detections per image).
- Training memory on the densest train_main images: 1024 batch 8 fits (20.8 GB), 2048 only batch 1 (20.6 GB; target assignment of P1868 with 10 206 GT ≈ 11 GiB at batch 2). 2-epoch smoke at 1024 (64 images, selection on calib) and end-to-end SCAHI pipeline smoke run; values not meaningful.
- Recipe proposal and cost estimate in the report: DOTA 1024 ≈ 1.6 GPU-h, 1024 + 2048 ≈ 8 GPU-h + ≈ 2 h CPU; VisDrone 2048 not recommended. No full training, no held-out evaluation; waiting for the go / no-go.

## 2026-11-01 ~22:35 – FCOS GO (DOTA 1024 + 2048), score correction, release deviation accepted (user)

- FCOS: GO for DOTA at 1024 and 2048 px; VisDrone: no. Addendum 3 committed before any training (6aa3ebf): recipe as in `reports/FCOS_SPIKE.md`; batch 8 at 1024, batch 1 + accumulation to 8 at 2048 (pre-specified FCOS-only exception to "batch ≥ 2": that rule exists for BatchNorm statistics, FCOS uses GroupNorm in the head and frozen BatchNorm in the backbone); hypotheses and rules unchanged, one model → no Holm, verdict per model, descriptive count over five detectors; no SAHI row.
- **Score correction (user):** every FCOS configuration uses the detector's own score sqrt(p_c × centerness) as in torchvision's post-processing; this replaces the earlier instruction "p × ctr" (spike export used p × ctr; changed in `scahi/fcos.py` before any FCOS training).
- My additions in addendum 3 (implementation of unchanged rules, for the user to see in the freeze proposal): pipeline limit 200 000 also for the calib tuning (with FCOS the earlier calib limit 20 000 would already bind in the single pass; "the limit must not bind" unchanged); box fusion uses the detector's own per-pass detections (torchvision post-processing per pass, boxes clipped to the pass); FCOS version of the parity gate (own post-processing of the dense outputs vs. torchvision's, bit-identical; calib mAP50 of the single pass from the own pass detections vs. the torchvision predictor, |Δ| ≤ 0.01).
- Release deviation accepted: `ablation_components.py` and the host-b latency measurement mode shipped as documentation with the import exception list; README sentence added. Difference table in `FCOS_SPIKE.md` marked "unverified". Final archive rebuilt only after the FCOS evaluation.

## 2026-11-01 ~20:45 – FCOS runs started; restart at an epoch boundary (implementation fixes, no recipe change)

- S_max smoke (addendum 3 item 3, `runs/dota/fcos_smoke.json`): 2048 at batch 1 fits on all 48 densest train_main images (peak 20.6 GB); 3072 at batch 1 does not (P1868, 10 206 GT) → S_max = 2048. Training 1024 started 20:40.
- Before the single pass of the chain ran, I found that the shared DOTA calib-tuning paths default to the YOLO calib limit 20 000 (`T.PIPE_MAX_DET`); addendum 3 fixes 200 000 for FCOS. Fixed in code for FCOS only (`T.predict` dispatch, `pipeline.pipe(model)` for box fusion / SCAHI top-k / limit counts); YOLO paths unchanged (tests, regeneration 62/62, host-a replay exact). Both chains were stopped right after epoch 5 had been saved and restarted; the resume first failed (generator state loaded onto the GPU; fixed with `.cpu()`) and then continued at epoch 6 from `last.pt` (model, optimizer, scaler, step counter and data-order generator restored; the flip of each sample depends only on epoch and index). Logs `runs/dota/fcos_logs/gpu_part1.log`, `gpu_part2_resume_error.log`, `gpu.log`.

## 2026-11-01 ~22:10 – FCOS 1024 done (training, gate, single pass, caches); 2048 training started

- Training 1024 (60 epochs, batch 8): best calib epoch 35, devkit mAP50 42.90 (selection as preregistered); peak 29.3 GiB of 31.3, no OOM.
- FCOS parity gate 1024 passed (`runs/dota/fusion/fcos_1024/gate.json`): own post-processing bit-identical with torchvision's on the gate images (whole image and tiles); calib mAP50 of the single pass own vs. torchvision 42.900 / 42.900 (|Δ| = 0).
- Single pass on calib, conf grid {0.001, 0.01, 0.05, 0.1}: 42.900 / 42.900 / 42.900 / 42.878 → conf 0.001 by the rule. **The pipeline limit 200 000 is reached on 8 of 141 calib images at conf 0.001** (whole image at 1024 gives up to ≈ 195 000 detections after torchvision's NMS). Reported as for every row; no rule changed. To be shown in the freeze proposal for the user's decision; the effect is at most in the 5th decimal (conf 0.05 drops the low-score tail and gives 42.9002).
- Caches 1024 (calib + DOTA-val, whole image + tiles) written; DOTA-val is not evaluated (caches only, as for the YOLO models).
- S_max for FCOS recorded as `T.S_MAX_FCOS = 2048`; `T.S_MAX` (YOLO, read by frozen outputs) left unchanged until the FCOS freeze.
- 2048 training started 22:07 (batch 1, accumulation 8). One allocator OOM warning at its start (5.3 GB request, 4.6 GB free); the allocator recovered and training continues – being watched. CPU calib tuning 1024 (box fusion + SCAHI grids) running.

## 2026-11-01 22:28 – Time stamps corrected (FCOS entries); addendum 3 predates every FCOS training result

- The manual times in the headers of three entries above are wrong; the git commit times are authoritative: "Task 6: FCOS spike decision" was committed 20:17 (9425643), not ~21:15; "FCOS spike done" 20:25 (b85ad3f), not ~22:00; "FCOS GO" 20:34 (3142473), not ~22:35. Content unchanged.
- Addendum 3 was committed 2026-11-01T20:34:36+01:00 (6aa3ebf), not ~22:30 as its heading says; dated correction note appended to the addendum (as for addendum 2), items 1–11 unchanged.
- Order: addendum 3 20:34:36 → FCOS training code 20:40:04 (0a23221, descendant of 6aa3ebf) → first artefact of the addendum-3 run 20:40:20 (S_max smoke), first training log 20:40:51. Before the addendum only the spike's smoke results existed (runs/fcos/*.json, 20:19–20:23). The FCOS freeze proposal computes and states this from git and file creation times (section "Präregistrierung vor dem ersten FCOS-Ergebnis").

## 2026-11-01 23:32 – FCOS pipeline limit: keep 200 000 (user), recorded before the DOTA-val evaluation

- **Decision (user):** the pipeline limit 200 000 stays for FCOS (per image and per pass), the same pipeline as for all YOLO rows. **Reason:** identical pipeline across detectors; only the lowest-scored detections are cut. Addendum 3 item 6 ("the limit must not bind") is therefore not met for FCOS; selection and rows stay as they are (no re-tuning).
- Where it binds on calib (found so far): single pass A_1024 on 8/141 images (per-image limit = torchvision detections_per_img); per-pass detections in the 1024 caches (used by box fusion only; SCAHI reads the dense outputs, no per-pass cap): 8 whole-image passes and 3 tile images on calib, 24 / 9 on DOTA-val (counted from the caches, no evaluation).
- Before the freeze: `scripts/dota/fcos_limit_check.py calib` computes for every FCOS row on calib the images at the per-image limit, the images with a pass at the per-pass limit, and Δ mAP50 without the limit (10 000 000; affected passes re-run, dense outputs checked identical to the caches; the per-level candidate limit 200 000 of addendum 3 item 6 kept) minus with the limit. The same run is the replay of every row on calib (limited detections from `F.row_dets`, the DOTA-val path). The freeze proposal shows the decision, the reason and the per-row table.
- After the single DOTA-val evaluation: per row the number of images at the limit, and as an exploratory check mAP50 without the limit for every affected row (`fcos_limit_check.py val`, as `d_limit_check.py` for P1384).

## 2026-11-02 00:19 – FCOS: rule for the per-pass limit (user, before the calib limit check), input-size check, memory safety

- **Rule (user, fixed before any limit-check number exists):** the per-pass limit 200 000 affects only the box-fusion rows (SCAHI reads the dense outputs). If |Δ mAP50| (without vs. with limit, `scripts/dota/fcos_limit_check.py calib`) is < 0.05 mAP50 points for **every** FCOS row on calib, 200 000 stays. Otherwise 10 000 000 is used for **all** FCOS rows, so that box fusion and SCAHI see the same detections; in that case (my implementation reading, to be confirmed in the freeze proposal) the per-pass detections of the caches are recomputed without the limit and the calib tuning of all FCOS rows is redone at 10 000 000 before the freeze, so that selection and DOTA-val use the same pipeline. The per-row Δ is reported in either case.
- **Input-size check (user, before the freeze; `scripts/dota/fcos_size_check.py` → `runs/fcos/size_check.json`):** torchvision's `GeneralizedRCNNTransform` (default min_size 800, max_size 1333) is set to min_size = max_size = L in every setting; measured on 3 calib images + P1868 (3771 × 10478) for both weight sets: training (model.train) long side 1024 / 2048, single pass 1024 / 2048, crossing 2560 / 3072, whole-image cache pass s, tile passes 1024 (native, scale 1; edge tiles at their letterbox long side 1024): **network long side = intended long side in 46/46 cases; the default 800 / 1333 was never used.** No bug; the crossing result (calib mAP50 60.30 at 2048, 63.08 at 2560) is a property of the model, not of the transform. Note for the paper: unlike the YOLO models, FCOS 2048 gains from a larger inference size on calib (selection stays as preregistered: X is secondary, A_2048 is evaluated at 2048).
- **Incident (2026-11-02 ~00:10):** my equality test of the parallel tuning was started while the 7 tuning workers held ≈ 91 GB of 125 GB; the kernel OOM killer ended the test and VS Code (this session). The GPU chain and the tuning were not affected. **Memory safety from now on (user):** the number of tuning workers is derived from the measured peak memory per worker on the densest images so that ≥ 20 GB stay free; the tuning runs as its own systemd user scope with a hard memory cap (`systemd-run --user --scope -p MemoryMax=…`) and with oom_score_adj 1000 for its processes, so that it is the first OOM victim and never the GPU chain or the editor; no memory-heavy test while the box-fusion stage runs.
- **Parallel tuning (user: all cores after the 2048 training, parallelisation only):** stages with fewer configurations than workers (box fusion, 12) are split by image (each worker loads only its images, all configurations on them; detections put back into the original image order before scoring); SCAHI stages do not load the per-pass detections. No threshold, limit or other setting changes, no pre-filtering. Used only after an equality test against the config-split path / stored tuning values.

## 2026-11-02 01:14 – fig_method: generic dense detector (user)

- Method figure (`reports/figures/paper/tex/fig_method.tex` → `tex/fig_method.pdf`, `fig_method.png`): the two heads (o2m / o2o) replaced by one block "Dense per-position outputs" of a generic dense detector; both paths branch from it – box fusion via the detector's own per-pass postprocessing (top, orange) and SCAHI via the dense outputs (bottom, blue). No head-specific labels (detector label without a model name). No compute; goes into the next exchange archive (already listed by `make_exchange_zip_paper.py`).

## 2026-11-02 01:16 – FCOS input sizes verified at runtime (user, before the tuning): intended sizes everywhere, no bug

- `scripts/dota/fcos_size_check_runtime.py` → `runs/fcos/size_check_runtime.json`: forward hook on `model.transform` (GeneralizedRCNNTransform) inside the real code paths – (a) training step as in `fcos_train.train` (train mode, `TrainSet` / `_to_gpu`; forward + loss only, no optimizer step), (b) single pass and (c) crossing via `fcos_train.predict`, (d) whole-image and tile cache passes via `scahi.fcos.raw_predict_level`; 3 calib images, and for training P1868 + 2 train_main images, both weight sets.
- **38/38 transform calls with min_size = max_size = intended long side and network long side = intended:** (a) training 1024 / 2048 (e.g. P1868 3771 × 10478 → 368 × 1024 / 737 × 2048); (b) single pass 1024 / 2048; (c) crossing 2048 / 2560 / 3072 (e.g. 1139 × 941 → 2048 × 1691 / 2560 × 2114 / 3072 × 2537); (d) whole-image cache pass at s, tile passes min/max 1024 / 1024 (native 1024 tiles unscaled; small edge crops to long side 1024). The default 800 / 1333 occurs nowhere. Confirms the earlier static check (`fcos_size_check.py`, 46/46). The crossing gain at 2560 is a property of the model; no fix, no repeated training.

## 2026-11-02 01:23 – YOLO scale augmentation (descriptive); FCOS crossing comparison in the freeze proposal (user)

- **YOLO training augmentation (read from the Ultralytics configs actually used, `runs/dota/train/*_s0/ultralytics/train/args.yaml`, all 7 DOTA trainings):** `scale: 0.5` (random scale ±50 %, Ultralytics default) active, plus `mosaic: 1.0` (`close_mosaic: 10`), `translate: 0.1`, `fliplr: 0.5`, `multi_scale: 0.0`. FCOS (addendum 3): horizontal flip only.
- **Descriptive, possible explanation (not tested):** the YOLO models saw objects over a range of scales in training and prefer inference at their training size 2048 on calib (YOLOv10s 73.70 / 71.32 / 73.05, YOLO11s 72.83 / 71.18 / 72.54 at 2048 / 2560 / 3072); FCOS saw each image only at long side 2048 and prefers 2560 (60.30 / 63.08 / 62.53) – larger inference enlarges the many small DOTA objects that are only a few pixels large after downscaling and that FCOS' location-based assignment (P3 stride 8, centre radius 12 px) covers poorly. No change to any recipe.
- **Freeze proposal (FCOS):** comparison table added; the pre-specified secondary comparison "tiles + SCAHI vs. best single pass of the inference crossing" uses the size selected on calib (X_fcos_2048 at 2560 px, best of 2048 / 2560 / 3072) and is reported next to the primary D-H3 (RG_2048 − A_2048, single pass at 2048).

## 2026-11-02 01:45 – Parallel FCOS calib tuning: equality test passed, memory-derived worker counts, tuning started

- **Old CPU chain stopped (00:58):** box fusion of the densest FCOS images takes up to ≈ 1 h per configuration and image; with one worker per configuration subset the stage would have run for many more hours. Its 3.4 h were lost (stage results are written only at the end of a stage).
- **Task split** (`scahi.dota.pipeline.tune`, stages without a timing reference: box fusion, F2): one task per (image, configuration), densest images first, each task loads only its image; a task starts only while the summed memory estimate (1 GB + factor × cache size of the image: 15 for box fusion, 45 for F2, from the measured peaks: F2 14.8 GB on P1139 / 356 MB, 10.2 GB on P1869 / 263 MB, box fusion 2.9 GB on P1139) stays within the budget; detections put back into the original image order before scoring. SCAHI stages no longer read the per-pass detections (`load_levels(dets=False)`). Refined-grid stages keep the config split (time ratio measured against the reference in the same worker).
- **Equality test passed** (`scripts/dota/fcos_split_equality_check.py`, `runs/dota/fusion/split_equality_check.json`): YOLO11s box fusion, task split vs. the stored tuning values, 3/3 configurations bit-identical; FCOS F2, config split vs. task split, 3/3 bit-identical (all fields except time). Time per image in the task split ≈ 1.7× higher (21 processes share memory bandwidth); it only breaks ties in the box-fusion / F2 selection; the refined-grid time rule is unaffected (config split).
- **First test run killed by its own cap (01:16, 80 GB scope)** – only the test scope, nothing else; this led to the measurement above.
- **Tuning started** (`scripts/dota/run_fcos_tuning.sh 4 21 95G 2-23`): systemd user scope, MemoryMax 95 GB (≥ 20 GB of 125 GB stay free with ≈ 9 GB used by the desktop / editor), oom_score_adj 1000, cores 2–23; refined grid 4 config-split workers (≈ 6.6 GB data + ≤ 15 GB peak each), box fusion / F2 21 task-split workers within an 85 GB budget. 1024 first, then 2048. No threshold, limit or other setting changed, no pre-filtering.

## 2026-11-02 08:23 – Addendum 4 (D-H5, user): SCAHI on the passes of the frozen SAHI rows; gate rule aligned before any gate result

- **Addendum 4 committed** `17086ff2c7e068f84eb51d60ac0247157d08e44b` at 2026-11-02T08:23:09+01:00, before any cache, gate or tuning result of the new rows (SAHI-F2, SAHI-RG for YOLOv10s / YOLOv10m on SAHI's passes: 640 px slices with the frozen overlap 0.1 / 0.2 plus the standard pass at long side 640).
- **Implementation in a new module** (`scahi/dota/sahi_scahi.py`): cache levels `s640` (SAHI's slice grid via `sahi.slicing.get_slice_bboxes`) and `w640`; the slice grid is put into `cache_mz.level_passes` for that process only. Reason: the running FCOS tuning spawns new workers per stage that import the shared modules from disk; nothing shared is edited while it runs.
- **Gate rule corrected before any gate result over a whole split** (note appended to addendum 4): the stored SAHI raw predictions come from Ultralytics' fused model, the caches from the unfused model, both with TF32 convolutions → different rounding (scores up to ≈ 0.01, boxes p99 ≈ 3 px on 4 calib images; network inputs bit-identical; with TF32 off the forwards agree). "≤ 4 ulp against the stored raw predictions" is therefore not achievable and was not the earlier rule; the gate now follows `pipeline.gate`: (1) decoding on the same forward at both new geometries, (2) |Δ calib mAP50| ≤ 0.01 of the frozen SAHI configuration from the cache passes; the raw comparison is reported descriptively. Part (1) already ran as a function test on these 3 calib images per model and passed (o2o counts identical, ≤ 0.61 of the tolerance, class agreement 1.0).
- **Found in the smoke:** Ultralytics' YOLOv10 o2o post-processing uses k = min(max_det, anchors of the whole input) in both top-k stages; `topk_like_ultralytics` applied to cached anchors (max class probability ≥ 0.001) caps the second stage at the number of cached anchors and drops (anchor, second-class) pairs. The new module takes k from the pass's input shape. The shared function and every frozen row stay unchanged; the effect on the existing o2o box-fusion rows is being measured on calib (reported separately, nothing changed).
- SAHI runs its standard pass only for images with more than one slice; the cache path and the SCAHI rows follow that.

## 2026-11-02 08:45 – Addendum 4: caches built, parity gate YOLOv10s passed, YOLOv10m failed (Δ 0.0151 > 0.01) → stop

- Caches `s640` / `w640` (calib, DOTA-val, both models): ≈ 4 GPU-min. Two gate runs were killed by their own 16 GB scope cap (exact one-to-one matching of the descriptive raw comparison does not fit on dense images); the descriptive comparison now uses nearest neighbours per class (KD-tree, max-norm). FCOS tuning not affected.
- **Gate (`runs/dota/fusion/<m>_1024/sahi_scahi_gate.json`):** (1) decoding at both new geometries: passed for both models. (2) calib mAP50 of the frozen SAHI configuration from the cache passes vs. frozen: **YOLOv10s 61.2981 vs. 61.2912, |Δ| 0.0069 → passed; YOLOv10m 63.3297 vs. 63.3146, |Δ| 0.0151 → failed** (tolerance 0.01).
- Descriptive (raw pass detections vs. the stored SAHI raw): calib 765 325 vs. 765 030 (v10s), 809 070 vs. 808 905 (v10m); DOTA-val 2 817 869 vs. 2 817 382, 2 798 406 vs. 2 797 901; median per-image p99 deviation ≈ 0.5 px box, ≈ 0.0009 score. Consistent with the TF32 / fused-vs-unfused rounding noted above.
- Per addendum 4: no tuning without a passed gate → stopped, reported to the user. Nothing tuned, no SCAHI number of these rows exists.
- Top-k finding (note of 08:23), measured on calib (cache counts only): dropped (anchor, class) pairs with score ≥ 0.01 (the lowest box-fusion threshold): YOLOv10s / v10m 1024: 0 / 0; 2048: 1 / 0 (max dropped score 0.0176). Frozen box-fusion rows unchanged.

## 2026-11-02 10:15 – Gate incident (D-H5, YOLOv10m) and fix; box back-mapping: deviation, what the gates tested, calib checks and decision rule (user, fixed before any check number)

- **Incident:** the parity gate of addendum 4 failed for YOLOv10m (criterion 2, calib mAP50 63.3297 vs. frozen 63.3146, |Δ| 0.0151 > 0.01; entry 08:45). **User decision:** tolerance stays at |ΔAP| ≤ 0.01 (G-E1 fix); rebuild the SAHI-layout caches with the fused model, same numerics as the predictor behind the frozen SAHI rows; record the model state of every cache; re-run the gate with unchanged criteria; if it fails again: root-cause analysis (as for YOLO11m), stop, no tolerance change.
- **Root causes found** (4 calib images per model; a fresh SAHI run reproduces the frozen raw predictions bit for bit):
  1. Unfused (cache) vs. fused (predictor) model, both with TF32 – different rounding.
  2. Box back-mapping: `scahi.infer.preprocess.input_to_source` divides by **one** gain; Ultralytics' `scale_boxes` divides per axis by (rounded resized side / side).
  3. Batching: the cache path ran 16 passes per forward, SAHI one; cuDNN picks other TF32 kernels for batches (scores up to ≈ 0.01).
- **Fix (only in `scahi/dota/sahi_scahi.py`, shared code untouched):** new cache levels `s640f` / `w640f`, built with the fused `DetectionModel` taken from SAHI's own load path (Ultralytics' `Detect.fuse` – removal of the o2m branch – skipped, conv+BN fused exactly as in the predictor), TF32 as there, per-axis back-mapping as `scale_boxes` (float32), batch 1. Each directory records this in `cache_state.json`. Check on 4 calib images per model: counts and scores identical to the frozen raw predictions, boxes ≤ 0.0005 px (≤ 2 float32 ulp). The first build (`s640` / `w640`: unfused, single gain, batch 16) is kept only as a record and is not used. Gate re-run on the new caches: result in the next entry.
- **Single-gain back-mapping in the existing caches (all frozen rows; nothing changed).** The deviation occurs only where rounding the resized side changes the gain. Count over all stored caches (tile and gain arrays):

  | Cache | Passes | Passes with deviation | Max relative deviation | Max shift at the far edge |
  |---|---|---|---|---|
  | DOTA whole image w1024 | 2995 | 89 % | 1.9·10⁻³ | 3.7 px |
  | DOTA whole image w2048 | 2396 | 89 % | 1.4·10⁻³ | 2.0 px |
  | DOTA tiles t1.0n1024 | 79 218 | 0.8 % (edge tiles) | 8·10⁻⁴ | 0.46 px |
  | VisDrone whole image (w0.333 … w2048) | 78 872 | 16–46 % | ≤ 6.4·10⁻⁴ | ≤ 0.5 px |
  | VisDrone tiles (all levels) | 617 229 | 0 | 0 | 0 |

  Not negligible for VisDrone (user): the metric is AP@[.5:.95]; a 0.5 px shift on an 8 × 8 px object gives IoU ≈ 0.88, i.e. a lost match at the thresholds 0.90 / 0.95.
- **What the parity gates tested:** G-E1 (VisDrone, `scahi/ext.py`) and the DOTA gate (`pipeline.gate`): (1) decoding parity in network-input pixels – no back-mapping involved; (2) the AP condition maps **both** sides (own and Ultralytics' decoding of the same forward) with the same `input_to_source` (single gain) – identical AP therefore shows that the decoding agrees and is blind to the mapping; (3) G-E1 only: cache path vs. the real Ultralytics predictor with tolerance (≥ 98 % of the predictor's detections with score ≥ 0.05 matched at IoU ≥ 0.99, Δscore ≤ 0.01) – the only place where the two mappings met; sub-pixel shifts pass by design. No gate tested the back-mapping exactly.
- **Planned checks (after the FCOS tuning, descriptive, calib, no frozen value changes):**
  - DOTA: YOLOv10s 2048, tiles + SCAHI (refined grid, RG) and its box-fusion counterpart (NMS), recomputed with the per-axis Ultralytics mapping; Δ mAP50 against the frozen calib values.
  - VisDrone: YOLOv10s 2048, tiles + SCAHI (refined grid) and tiles + box fusion, same; Δ AP@[.5:.95] and Δ AP50 against the stored values, plus Δ for objects < 16 px.
- **Decision rule (user, fixed before any number of these checks exists):**
  - If |Δ| on calib (VisDrone: AP@[.5:.95]; DOTA: mAP50) is **below 0.05 for both checked rows** of a dataset: no change; the quantified deviation is documented in the supplement as a known small difference.
  - If it is **0.05 or larger for either row**: treat as a bug fix. Recompute all affected rows on the held-out sets (VisDrone val / test-dev, DOTA-val) with the per-axis mapping, **without any re-tuning**, and report original and corrected values side by side. Pre-registered verdicts stay as originally computed; for each, state whether it would change. **Stop and report before writing anything into the paper materials.**

## 2026-11-02 10:29 – Mapping checks: three input/back-mapping details, combined (user, before any check number)

- Besides the single-gain back-mapping, two further differences from the Ultralytics predictor were found while aligning the SAHI-layout caches (5 calib / 8 DOTA-val images with differing counts, all explained): (a) **size rule** – `preprocess.full_image_shape` pads the unrounded scaled side up to a multiple of 32, Ultralytics' LetterBox rounds first (e.g. P1453 2322 × 1742: 512 × 640 vs. 480 × 640); (b) **letterbox gain** – `preprocess.letterbox` resizes with the gain to the padded target shape, Ultralytics with r = min(L/h, L/w) (e.g. P2077 846 × 635: 639 vs. 640 rows); Ultralytics' `scale_boxes` then maps back from the padded input shape. All three are replicated exactly only in `scahi/dota/sahi_scahi.py` (D-H5 caches); shared code and all frozen rows unchanged.
- **Rule (user):** the planned calib checks (DOTA: YOLOv10s 2048 RG and box fusion, mAP50; VisDrone: YOLOv10s 2048 tiles + SCAHI refined grid and tiles + box fusion, AP@[.5:.95], AP50, objects < 16 px) align **all three details (letterbox gain, size rule, per-axis back-mapping) to Ultralytics at once**; the 0.05 rule of the 10:15 entry applies to the **combined** effect. **Only if a row reaches 0.05:** additionally report the contribution of each detail separately (one at a time) for the bug-fix report.

## 2026-11-02 10:36 – D-H5 parity gate passed on the final SAHI-layout caches (exact)

- Final caches `s640f` / `w640f` (`cache_state.json`): fused model from SAHI's load path (o2m kept), TF32 as the predictor, batch 1, Ultralytics LetterBox (size rule and resize gain r = min(640/h, 640/w)) and `scale_boxes` back-mapping, SAHI's slice grid and standard-pass rule. Superseded builds kept as a record, not used: `s640` / `w640` (unfused, single gain, batch 16), `w640f_padshape_superseded`, `*_lbgain_superseded`.
- **Gate (criteria unchanged; `runs/dota/fusion/<m>_1024/sahi_scahi_gate.json`): passed for both models.** (1) decoding at both geometries passed; (2) calib mAP50 of the frozen SAHI configuration from the cache passes: YOLOv10s 61.29125 vs. 61.29125, YOLOv10m 63.31461 vs. 63.31461 (|Δ| 0.0).
- Descriptive: raw pass detections identical in number on every image (calib 141/141, DOTA-val 458/458; 765 030 / 2 817 382 and 808 905 / 2 797 901 detections), boxes ≤ 0.002 px (float32 rounding of the decoding order). The nonzero "score max" of the nearest-neighbour comparison comes from coincident boxes of different detections of the same class (matching artefact of the descriptive check, not a score difference).
- Next per addendum 4: SAHI-F2 / SAHI-RG calib tuning only after the FCOS tuning, under the same memory rules.

## 2026-11-02 11:52 – FCOS tuning: refined-grid stage OOM-killed (scope cap worked); memory measured

- The FCOS calib tuning scope (MemoryMax 95 GB, oom_score_adj 1000) was ended by the OOM killer at 11:42, about 2 min into stage `RGC_o2m` at 1024 (4 config-split workers). Nothing else was affected. Stages finished and stored in `runs/dota/fusion/fcos_1024/tuning.json`: `NMS_o2m` (31 736 s), `F2_o2m` (3 824 s), `B-F2_o2m` (84 s); the stage is resumable, nothing of RGC was stored.
- Cause: the planning estimate (≈ 6.6 GB data + ≤ 15 GB peak per worker) came from the F2 grid in task-split processes. Measured now (single process, one core, own 40 / 80 GB scope): config-split worker after loading all calib data 5.1 GB; reference F2 config (τ_splat 0.02, τ_peak 0.01, cell 5, 4 bins) peaks at 32.6 GB on P1139; heaviest refined-grid variants on P1139 alone: τ_splat 0.0003125, τ_peak 1e-5, 8 bins, cell 7 → 47.6 GB (119 s), cell 13 → 67.7 GB (147 s). All four workers run the reference first on the same image order, so their peaks coincide.
- Consequence: with config split (every worker holds all images) only one worker fits under the cap with ≥ 20 GB free. The tuning is stopped until the user decides how to parallelise the refined-grid stages (time rule affected, see next entry). No threshold, limit, grid or other setting changed.

## 2026-11-02 13:12 – FCOS refined-grid stages: task split with the reference timed in the same process (user, option 3; before any FCOS RG result)

- No refined-grid (RG) result of FCOS exists; the choice is made before any outcome. Options considered (config split with 1 worker ≈ 4–5 days; task split with summed times; this option; hybrids; memory reduction in the fusion code; higher cap) are in the session brief; the user chose option 3.
- **Implementation (parallelisation only; grid, thresholds, limits and selection rule unchanged):** one task per (image, config), densest images first; per-task memory estimate that depends on the config, fitted conservatively on measured points (cache size, τ_splat, cell factor, bins); scope cap 95 GB, MemorySwapMax 0, oom_score_adj 1000, ≥ 20 GB free, cores 2–23. Each task also times the reference config on the same image in the same process, in alternating order (reference first on even tasks, variant first on odd tasks); `time_ratio_vs_ref` = sum of the variant's times / sum of the reference times over that variant's tasks. Detections are reassembled in image order; mAP50 must equal the config-split path.
- **Cross-check before FCOS:** the RG stage (row C) of YOLOv10s at 1024 on DOTA calib is re-run with option 3. Required: the same selected config as the frozen selection and identical mAP50 for it. Reported: per variant, the time ratio under option 3 vs the stored ratio, and every variant whose admissibility (ratio ≤ 2) differs. If the selected config differs: stop and report.
- **2048:** before the 2048 stages, the densest calib image is profiled with the heaviest variant. If a single task does not fit under 95 GB, it runs alone with the largest cap that keeps ≥ 20 GB free; if it still does not fit: stop and report. No variant is dropped because of memory.
- Then: FCOS RG stages at 1024 and 2048, limit check, freeze proposal with replay, stop before DOTA-val; afterwards the two mapping checks and the D-H5 tuning.

## 2026-11-02 14:09 – Option 3 cross-check (YOLOv10s, 1024, row C, DOTA calib): passed

- `scripts/dota/rg_option3_crosscheck.py` → `runs/dota/fusion/yolov10s_1024/rg_option3_crosscheck.json` (task parts in `rg_tasks/crosscheck_RGC_o2m/`, memory profile `rg_mem_C.json`); 15 369 tasks, 21 workers, own scope (40 GB cap, oom_score_adj 1000), ≈ 52 min. No stored row or selection changed.
- **Selected config identical** to the frozen selection (τ_splat 0.0003125, τ_peak 0.001, cell 7, 8 bins/oct), **mAP50 identical** (70.10812777799428); mAP50 identical for all 109 variants (detections reassembled in image order = config-split path).
- Time ratio option 3 / stored per variant: median 1.004, range 0.923–1.118. Admissibility (ratio ≤ 2) differs for 4 variants, all with stored ratio 1.94–2.01 (τ_splat 0.0003125, τ_peak 0.001):

  | cell | bins | mAP50 | ratio stored | ratio option 3 | admissible stored → option 3 |
  |---|---|---|---|---|---|
  | 9 | 8 | 69.969 | 1.943 | 2.120 | yes → no |
  | 11 | 7 | 70.011 | 1.956 | 2.002 | yes → no |
  | 13 | 6 | 68.875 | 1.983 | 2.084 | yes → no |
  | 13 | 7 | 69.822 | 2.005 | 1.899 | no → yes |

  None of them has a higher mAP50 than the selected config, so the selection does not depend on the timing protocol here. The flips show the timing noise at the cap (about ±10 %), present in both protocols.
- Next: profile the densest 2048 calib image with the heaviest variant (rule of the previous entry), then resume the FCOS tuning (RGC / RGB at 1024, then 2048) with option 3.

## 2026-11-02 14:16 – FCOS 2048 memory check passed; FCOS tuning resumed with option 3

- Heaviest refined-grid variant (τ_splat 0.0003125, τ_peak 1e-5, cell 13, 8 bins) as a single task (fused twice in one fresh process) on the densest 2048 calib image (P1139, row C cache 338 MB), own scope capped at 105 GB: **peak 65.7 GB**, 144 s per fusion (`runs/dota/fusion/fcos_2048/heavy_single_task_mem.json`). Fits under the 95 GB cap of the tuning scope; no task needs a separate run.
- FCOS tuning resumed with the unchanged launcher (`scripts/dota/run_fcos_tuning.sh 4 21 95G 2-23`): `fuse_job` skips the stored stages (NMS, F2, B-F2 at 1024); RGC / RGB use option 3 with 21 task workers within the 85 GB budget, each stage first profiling its memory (`rg_mem_<row>.json`). No threshold, limit, grid or selection rule changed.

## 2026-11-02 16:23 – Option 3 cross-check accepted; sensitivity note on the 2× time rule (user; reporting only)

- The user accepted the option-3 cross-check (entry above).
- **Sensitivity note (reporting only, selection rule unchanged):** the FCOS freeze proposal and later the D-H5 freeze proposal report, per refined-grid stage, the time ratio of the selected config and every variant with a higher calib mAP50 than the selected one whose time ratio lies in [1.8, 2.2]. Implemented in `scahi/dota/time_rule.py` (new module, so the running FCOS workers, which re-import `pipeline.py`, are not touched) and used in `scripts/dota/d_freeze_proposal_fcos.py` (also written to `runs/dota/freeze/time_rule_sensitivity_fcos.json`). Smoke test on the stored YOLOv10s 1024 stages: RG selected ratio 1.799, B-RG 1.888; no variant with higher mAP50 in the band.

## 2026-11-02 17:15 – Tighter memory model for the refined-grid tasks (user; scheduling only)

- **Approved by the user (parallelisation only; no change to results, grids, thresholds, limits or the selection rule).** The FCOS tuning was paused at 17:12 (manual stop of the scope during `RGC_o2m` 1024; 9 026 of 15 369 tasks stored, detections and timings complete for each) and is resumed from the stored tasks.
- **Why:** the conservative estimate scales the task peak on the densest image linearly with cache size; measured peaks of the finished tasks are 21 % of it in the median (max 80 %). Simulated remaining wall time of `RGC_o2m` 1024: ≈ 41 h.
- **Model (`scahi/dota/pipeline.py`, `_rg_tight_fit`, `_rg_est_tight`):** y = (peak − 1 GB) / (dense peak of the config − 1 GB) against x = cache size / cache size of the densest image, y ≈ Q·x^α. α by least squares on log y for x ≥ 0.2 (where memory decides the scheduling; on small images the peak is mostly base overhead and a fit over all x gave α 3.5, Q 30, useless). Q = 99.9th percentile of y / x^α in that range, raised if needed so that 1.25·Q·x^α covers every measured peak (all x, profile points included). Estimate = 1 GB + 1.25·Q·(dense peak − 1 GB)·x^α, and never below a measured peak of the same config on an image that is not larger. Refitted every 500 finished tasks; used once the stage has ≥ 1000 finished tasks, ≥ 20 of them with x ≥ 0.2. Estimates of running tasks are not changed.
- **Offline check on the 9 026 finished tasks:** α 1.87, Q 1.00 (47 points with x ≥ 0.2); max measured peak / estimate 0.84 over all finished tasks and 0.80 over the profile points; heaviest P1139 task 84.6 GB (as before; measured 67.7), P1869 heaviest 48.5 GB (conservative 63). Simulated remaining wall time of `RGC_o2m` 1024: ≈ 22 h (conservative ≈ 41 h); the densest images stay the bottleneck.
- **Fallback:** on first use in a stage the scheduler writes `tight_model.json` into the stage's task dir. If the scope is OOM-killed, `scripts/dota/fcos_tuning_loop.sh` (now started by `run_fcos_tuning.sh`) restarts it (at most 3 runs, only after an OOM kill found in the journal); the stage finds the marker and uses the conservative estimate for the rest of the stage. A manual pause of a stage that already uses the tight model requires renaming the marker (recorded here if it happens).
- Unchanged: scope cap 95 GB, MemorySwapMax 0, oom_score_adj 1000, budget 85 GB, cores 2–23, ≥ 20 GB free.

## 2026-11-02 17:20 – Mapping checks: Ultralytics-aligned caches built now (user; evaluation only after the FCOS tuning)

- `scahi/mapcheck.py`: all three details aligned at once (size rule: round, then pad each side to a multiple of 32; letterbox gain r = min(L/h, L/w); per-axis back-mapping as `scale_boxes`); everything else as the frozen caches (`load_unfused`, batch 16, TF32 default). Separate root `cache_ualigned/` with `cache_state.json` per level; no stored cache, row or value changes.
- Rows checked: DOTA YOLOv10s 2048, levels w2048 + t1.0n1024 (RG and its box-fusion counterpart); VisDrone YOLOv10s 2048, levels w2048 + t2.0n1024 (`XG_yolov10s_2048_o2m_C`, refined grid; `X1`, box fusion F0-NMS o2o). Calib only.
- Smoke test (3 calib images per dataset and level, against the frozen caches): probabilities, pass geometry and gains identical; boxes differ only by the back-mapping (DOTA w2048 ≤ 0.32 px, VisDrone w2048 ≤ 6·10⁻⁵ px, one upscaled DOTA edge tile ≤ 0.05 px; all other tiles bit-identical).
- Runs on cores 0–1, nice 19, own scope (MemoryMax 10 GB, no swap, oom_score_adj 1000). With the FCOS scope at its 95 GB cap the worst case leaves ≈ 7 GB free instead of 20 GB; the actual FCOS use is far below the cap most of the time, and an OOM of either scope ends only that scope.

## 2026-11-02 17:29 – D-H5 calib tuning started in parallel to the FCOS tuning (user; before any D-H5 tuning result)

- **Decision (user):** the D-H5 calib tuning (addendum 4, item 5) starts now on cores 0–1, in parallel to the FCOS tuning, instead of after it (addendum 4, item 8). Conditions: own systemd user scope, MemoryMax 10 GB (125 GB − 20 GB free − 95 GB FCOS cap; the two caps together never exceed 105 GB), MemorySwapMax 0, oom_score_adj 1000; the scheduler starts only tasks whose estimate fits the budget, tasks that do not fit wait until the FCOS tuning has finished and then run with the larger budget (no task dropped, results identical); RG timing with option 3 as for FCOS; FCOS keeps priority: the D-H5 scope is paused if the FCOS scope approaches its cap or shows memory warnings. Then: freeze proposal with replay, stop.
- **Implementation** (`scahi/dota/sahi_scahi.py`, `dh5_stage` / `dh5_tune`; `scripts/dota/run_dh5_tuning.sh`; the shared `pipeline.py` is not edited while FCOS runs): per model SAHI-F2 (VisDrone F2 grid, 24 configs; selection `pipeline.select`) then SAHI-RG (refined grid, 108 variants + the selected SAHI-F2 config as reference; `select(..., capped=True)`), o2m, calib, SAHI-layout caches `s640f` / `w640f` (`load`). One task per (image, config) in a fresh process, densest images first; RG tasks time the reference on the same image in the same process (alternating order); detections reassembled in image order for scoring. Memory estimate per task: measured peaks on the densest calib image (F2: every config; RG: grid corners, reference, two interior points, dominance rule as for FCOS), linear in cache size × 1.25, checked on the 10th densest image; task budget 8.5 GB inside the 10 GB scope; tasks above it are deferred.
- **Pipeline limit 200 000** for tuning and evaluation, as addendum 4 item 5 states. Note for the freeze proposal: the frozen YOLO SCAHI rows were selected at 20 000 and evaluated on DOTA-val at 200 000 (decision 2026-10-30); for the D-H5 rows selection and evaluation both use 200 000.
- **Watchdog** (`scripts/dota/dh5_watchdog.sh`, log `runs/dota/fcos_logs/dh5_watchdog.log`): `systemctl --user freeze` of the D-H5 scope while the FCOS scope's anonymous memory is ≥ 85 GB (its task budget), thaw below 75 GB; after a new memory warning in the FCOS log the D-H5 scope stays frozen until resumed by hand. Freeze/thaw tested on a dummy scope.
- Workers: 2 (cores 0–1, nice 19). Log `runs/dota/fcos_logs/dh5_tuning.log`; results `runs/dota/fusion/<m>_1024/dh5/tuning.json`.
- *Addendum (17:32):* first D-H5 run stopped after 19 tasks: for YOLO the task peaks on the densest image (≈ 0.9 GB) lie below the 1 GB base, so the linear formula base + 1.25 (peak − base) × ratio gave estimates just below the measured peaks (warnings, no risk: 2 × 1 GB against 8.5 GB). Formula changed to 1.25 × (base + max(peak − base, 0) × ratio) (≥ 1.25 × the measured peak on the densest image); restarted from the stored tasks.

## 2026-11-02 17:37 – FCOS limit rule: meaning if it triggers; limit check 1024 first (user; before any limit-check number)

- No number of the FCOS limit check exists (`runs/dota/freeze/limit_calib_fcos*.json` absent).
- **1. Meaning of the rule (user):** if |Δ mAP50| ≥ 0.05 in any FCOS row on calib, the selected configs stay as tuned on calib with 200 000; only the calib replay and the DOTA-val evaluation of **all** FCOS rows use 10 000 000. No re-tuning. This mirrors the YOLO rows (selected at 20 000, evaluated at 200 000). Replaces "caches and calib tuning of all FCOS rows anew" in the 2026-11-02 rule. The per-level candidate limit of addendum 3 item 6 (200 000) stays. `scripts/dota/d_freeze_proposal_fcos.py` states this; the calib replay at 10 000 000 is the column "mAP50 without limit" of the limit check.
- **2. Order (user):** the limit check of the FCOS 1024 rows runs as soon as all 1024 stages are finished, while the 2048 stages still run; the result is reported right away; the 2048 part follows after the 2048 stages. Code: `scripts/dota/fcos_limit_check.py calib 1024` / `calib 2048` (`final.proposal_rows(..., sizes)`; parts `limit_calib_fcos_<s>.json`, merged by the freeze proposal). Rows and rule unchanged.
- **Resource constraint (to be resolved with the user before it runs):** the check fuses the affected images of every row without the limit; for the SCAHI rows on the densest calib images this needs up to the task peaks of the tuning (P1139: 28.5 GB for the F2 reference, up to 67.7 GB for refined-grid variants at 200 000, possibly more without it). The free memory next to the FCOS scope (95 GB cap) is 10 GB (the D-H5 slot, free after the D-H5 tuning); the GPU part and the light rows fit there, the SCAHI rows on dense images do not.
- Aim (user): FCOS freeze proposal by Friday afternoon at the latest; any slip is reported early.

## 2026-11-02 17:38 – Option A for the 1024 limit check (user): FCOS paused for it

- **Decision (user):** when the FCOS 1024 stages are finished, the FCOS scope is stopped, the complete 1024 limit check (incl. the SCAHI rows on the dense images) runs alone under the 95 GB cap, the result is reported, then FCOS 2048 resumes from the stored stages / tasks. The pause (start, end, work lost and rerun) is recorded here when it has happened.
- Automated by `scripts/dota/fcos_limit_1024_pause.sh` (tmux `scahi_lim1024`, log `runs/dota/fcos_logs/limit_1024_pause.log`): waits until `runs/dota/fusion/fcos_1024/tuning.json` holds the 1024 selection (written right after the last 1024 stage), stops the FCOS scope, runs `scripts/dota/fcos_limit_check.py calib 1024` in its own scope (MemoryMax 95 GB, no swap, oom_score_adj 1000, cores 2–23, GPU for the passes without limit), then restarts the FCOS tuning with the unchanged launcher. The D-H5 scope (10 GB) may still run in parallel; the caps together stay at 105 GB.
- Work lost: the 2048 stage that has started when FCOS is stopped (box fusion, stored per stage only) restarts from its beginning; expected loss seconds to minutes, since the stop follows the 1024 selection within 20 s.
- *Addendum (18:31):* in the YOLOv10s SAHI-RG stage 23 tasks on P1869 (second-densest image) peaked up to 3 % above their estimate (≈ 2.0 vs 1.97 GB; no risk: 2 tasks against 8.5 GB): peak memory is not proportional to cache size there. The scheduler now scales the estimates of all open tasks by k = max(1, 1.25 × max(measured peak / raw estimate)) over the stage's finished tasks, updated after every task (restores the 1.25 margin; tasks that then exceed the budget are deferred, none so far). Restarted from the stored tasks (178 done; ≤ 2 in-flight tasks rerun); k = 1.288, max estimate 2.7 GB. Scheduling only.

## 2026-11-03 05:33 – D-H5 calib tuning finished; freeze proposal with replay; stop for approval

- Tuning finished 2026-11-03 05:20 (both models, F2 and RG; 0 tasks deferred; 31 memory warnings, all at the start of a stage before the factor k had measured peaks, highest task peak 2.97 GB under the 10 GB cap; no OOM; the watchdog never froze the D-H5 scope). Watchdog session ended.
- Replay path `scahi.dota.sahi_scahi.dh5_row_dets` (all images of the split in one process, `P.fuse`, limit 200 000) is also the path of the single DOTA-val evaluation. Replay on calib: 4/4 rows exact in mAP50 and mAP50 cap 4000 (`runs/dota/freeze/replay_dh5.json`; cores 0–1, 10 GB scope).
- Proposal: `docs/FREEZE_PROPOSAL_DOTA_DH5.md`, `configs/dota/frozen_rows_dh5.json` (`approved: false`), incl. the 2× time-rule sensitivity note and the limit note. **Stop:** DOTA-val is not touched before the user's approval.

## 2026-11-03 07:59 – D-H5 freeze approved (user); single DOTA-val evaluation of the 4 rows

- **Approval (user):** the 4 rows of `docs/FREEZE_PROPOSAL_DOTA_DH5.md` (`bf64c9a`) are frozen (`configs/dota/frozen_rows_dh5.json`, `approved: true`). Reporting rules as for the original DOTA freeze: all 4 rows, no re-tuning; primary mAP50 without cap; secondaries cap 4000, "all objects", AP@[.5:.95]; size bins. D-H5 = SAHI-RG − SAHI (frozen), paired bootstrap CI (1000, seed 0), Holm over the two models, holds if it holds for both; SAHI-F2 − SAHI reported as secondary. The frozen SAHI rows are not re-evaluated; their stored DOTA-val evaluation (`runs/dota/val/eval/SAHI_<m>*`, same bootstrap weights) is the reference.
- DOTA-val once, now, in the D-H5 slot (scope MemoryMax 10 GB, no swap, oom_score_adj 1000, cores 0–1, nice 19; FCOS keeps priority): `python -m scahi.dota.sahi_scahi val` (detections via `dh5_row_dets`, the replayed path; written once to `runs/dota/val/dets/<id>.parquet`; evaluation `final.evaluate_row`; result `runs/dota/val/results_dh5.json`).
- **Latency (user):** SAHI (frozen config) and SAHI passes + SCAHI (CPU and GPU fusion), both models, in ONE exclusive run on host-a after the FCOS tuning; the user closes VS Code for it and is told in advance; before Friday evening.

## 2026-11-03 08:07 – D-H5 latency: started automatically after the FCOS tuning (user)

- **User:** VS Code hardware acceleration is disabled permanently (`argv.json`), so VS Code has no process on the GPU; verified with nvidia-smi before the run. The exclusive D-H5 latency run may be started without the user as soon as the FCOS tuning has finished (also at night), from a detached tmux session; exclusivity rules unchanged (abort on any foreign compute process; graphics-only desktop processes recorded in setup).
- `scripts/dota/after_fcos_tuning.sh` (tmux `scahi_after_fcos`, log `runs/dota/fcos_logs/after_fcos.log`): waits for the end of the option-A pause and then the end of the FCOS tuning (2048 selection written, "CPU chain done", FCOS session gone; otherwise stop), then for the D-H5 DOTA-val / GPU-parity jobs; starts `scripts/latency_host_a_dh5.py measure` + `report` only if no compute process is on the GPU (retry every 5 min, at most 2 h, else no run); snapshots of all GPU processes with start times at start and end (`runs/latency_host_a/gpu_procs_{start,end}_dh5.txt`, for `setup_dh5.json`). Afterwards the FCOS limit check of the 2048 rows (95 GB cap, cores 2–23, as the 1024 part).
- D-H5 latency protocol (`scripts/latency_host_a_dh5.py`): as `scripts/latency_host_a_dota.py`; per model SAHI (frozen, same function as the host-a DOTA table) and the SAHI passes computed live (fused model of SAHI's predictor path, batch 1) + SCAHI of SAHI-F2 and SAHI-RG on the CPU and the GPU (blocked GPU fusion; GPU times only with a passed parity gate `runs/gpu_fusion/parity_dh5.json`, every DOTA-val image against the stored CPU detections); smoke (3 images) requires the CPU fusion of the live passes to equal the cached path on 3 calib images.

## 2026-11-03 09:10 – D-H5 DOTA-val evaluated once (4 rows)

- Detections written once (`runs/dota/val/dets/SAHI-{F2,RG}_<m>.parquet`, 0 images at the limit 200 000), evaluated with `final.evaluate_row`; SAHI rows from their stored evaluation (values identical to `results.json`). Result `runs/dota/val/results_dh5.json`. Scope 10 GB, cores 0–1, no OOM.
- **D-H5 holds** (both models): SAHI-RG − SAHI, mAP50 without cap: YOLOv10s +2.875 (CI [2.13, 4.33]), YOLOv10m +2.259 (CI [1.08, 3.25]); Holm-p 0.002 each, Δ > 0.30.
- Secondary (no verdict): SAHI-F2 − SAHI +2.758 (v10s), +2.018 (v10m); cap 4000, "all objects" and AP@[.5:.95] in the same direction.

## 2026-11-03 11:18 – D-H5 latency decoupled from the end of the FCOS tuning; FCOS runs to completion (user)

- User instruction: once the user confirms that no VS Code process is on the GPU, stop the FCOS scope, run the exclusive D-H5 latency measurement (SAHI frozen; SAHI passes + SCAHI, CPU and GPU fusion; both models; report carries the GPU parity gate result `runs/gpu_fusion/parity_dh5.json`), then resume FCOS from the stored tasks with the unchanged launcher (`run_fcos_tuning.sh 4 21 95G 2-23 85`). Script `scripts/dota/dh5_latency_pause.sh` (tmux `scahi_dh5_latency`, log `runs/dota/fcos_logs/dh5_latency_pause.log`: start, end, task files stored before / after the stop, processes in the scope; unfinished tasks are rerun). The option-A waiter `scahi_lim1024` would abort when the FCOS session disappears, so the script stops it before and restarts it after the pause (unchanged script). Exclusivity rules unchanged.
- Latency step removed from `scripts/dota/after_fcos_tuning.sh` (restarted 11:18); the 2048 limit check stays there.
- FCOS: no change to grid, stages or rules; the tuning runs to completion. If the freeze proposal cannot reach the user by Friday afternoon 2026-11-06, it is kept ready and work stops until the user approves it.
- Hourly progress estimate (reporting only): `scripts/dota/fcos_progress.py` (tmux `scahi_fcos_progress`, cores 0–1, nice 19) → `runs/dota/fcos_logs/progress_hourly.log` (stored task files of the running refined-grid stage, rate last 1 h / 6 h, ETA of that stage). At 11:18: 1024 RGC 13 285/15 369, about 100–120 tasks/h, ETA of RGC 17–21 h; 1024 RGB, the option-A pause and the 2048 stages follow.

## 2026-11-03 11:35 – Parallel slot (former D-H5 slot); mapping checks evaluated: below 0.05 for every row (user)

- **Slot (user):** parallel work next to FCOS without touching it or any held-out evaluation: `scripts/slot_run.sh` (one job at a time via flock; own scope "scahi-slot <tag>", MemoryMax 10G, MemorySwapMax 0, oom_score_adj 1000, cores 0–1, nice 19, no GPU; logs `runs/slot/`). FCOS cap 95 GB and priority unchanged (95 + 10 ≤ 105 GB). `scripts/slot_watchdog.sh` (tmux `scahi_slot_wd`) freezes slot scopes while the FCOS scope's anonymous memory is ≥ 85 GB (thaw < 75), after a new FCOS memory warning (until thawed by hand), and while `runs/slot/hold` exists — set by `dh5_latency_pause.sh` for the exclusive D-H5 latency run.
- **Mapping checks** (`python -m scahi.mapcheck evaluate dota|visdrone`, rule of 2026-11-02 10:15 / 10:29, calib only): each row fused twice through the same code path, from the frozen caches and from `cache_ualigned/` (all three details aligned to Ultralytics at once); the frozen-cache values reproduce the stored calib values exactly for every row. Δ = aligned − frozen:

  | Dataset | Row | Metric | Frozen | Aligned | Δ |
  |---|---|---|---|---|---|
  | DOTA | `RG_yolov10s_2048` (tiles + SCAHI, refined grid) | mAP50 | 76.01729 | 76.01583 | −0.00146 |
  | DOTA | `NMS-o2m_yolov10s_2048` (box fusion) | mAP50 | 73.57092 | 73.57208 | +0.00116 |
  | DOTA | `NMS-o2o_yolov10s_2048` (box fusion, o2o; extra) | mAP50 | 73.38836 | 73.38710 | −0.00126 |
  | VisDrone | `XG_yolov10s_2048_o2m_C` (tiles + SCAHI, refined grid) | AP@[.5:.95] | 43.45884 | 43.45741 | −0.00143 |
  | VisDrone | `X1` (tiles + box fusion F0-NMS o2o) | AP@[.5:.95] | 41.17029 | 41.15715 | −0.01315 |

  VisDrone secondaries (Δ): XG AP50 −0.0025, AP < 16 px +0.0168, AP50 < 16 px −0.0118; X1 AP50 −0.0015, AP < 16 px +0.0049, AP50 < 16 px −0.0118. Results `runs/mapcheck/{dota,visdrone}.json`.
- **Decision by the fixed rule:** |Δ| < 0.05 for every checked row of both datasets → no change, no held-out recomputation; the quantified deviation goes into the supplement as a known small difference.

## 2026-11-03 11:37 – Exchange archive D-H5 and release update with D-H5 / FCOS code; clean-room passed (user; slot)

- **Exchange archive** (user: D-H5 results, new fig_method, supplement tables; commit only the guide): `exports/scahi_dh5_exchange_20261103_4185a71b180f.zip` (24.7 MB, 169 files; PNGs at 150 dpi by the existing size fallback, PDFs full). Guide `reports/FOR_CLAUDE_DH5.md` (`281e4b9`, FOR_CLAUDE.md in the archive). D-H5 tables `reports/dota/tab_dota_dh5*` from `results_dh5.json` (reading stored results only; `scripts/dota/d_dh5_tables.py`, builder `scripts/make_exchange_zip_dh5.py`) – left uncommitted as instructed. No D-H5 latency and no FCOS numbers in the archive; the guide still lists the mapping checks as pending (built before their result, entry above).
- **Release** (`4185a71`, `dd72ac8`): manifest + D-H5 code (`sahi_scahi`, frozen rows DH5, freeze proposal, parity, latency script) and FCOS code (`fcos`, `fcos_train`, run / limit / size checks, time rule, `mapcheck`); ops launchers excluded as before; README table fixed. Build from the clean tree 606b8d1: 336 files, 0 forbidden terms, 0 missing imports. **Clean-room test (no GPU, slot) passed:** pytest 69 passed / 7 skipped, 51 modules import, regeneration 14/14 commands, 62/62 outputs byte-identical (`runs/release/cleanroom.json`).

## 2026-11-03 11:39 – Mapping checks closed; D-H5 tables committed; FCOS 2048 stages with cap 105G / budget 95 GiB (user)

- **Mapping checks (user):** below the 0.05 rule for every row → no correction, no held-out recomputation; the supplement states the quantified deviation as a known small difference. Next archive guide updated: `scripts/make_exchange_zip_dh5.py` reads `runs/mapcheck/{dota,visdrone}.json` and ships them; rebuilt `exports/scahi_dh5_exchange_20261103_328e49de1eb9.zip` (23.5 MB, 171 files, verify passed; replaces the earlier D-H5 zip as the current one; zips stay uncommitted). `reports/FOR_CLAUDE_DH5.md` updated.
- **Tree clean (user):** D-H5 tables `reports/dota/tab_dota_dh5{.md,_rows,_contrasts,_sizebins,_parameters.csv}` and their generators `scripts/dota/d_dh5_tables.py`, `scripts/make_exchange_zip_dh5.py` committed.
- **FCOS 2048 stages (user; parallelisation only, before the restart):** at the restart after the option-A pause the FCOS scope runs with MemoryMax 105G and task budget 95 GiB (`fcos_limit_1024_pause.sh` → `run_fcos_tuning.sh 4 21 105G 2-23 95`; waiter `scahi_lim1024` restarted 11:38 with the new script). Safety factor 1.25 and the tight memory model unchanged; ≥ 20 GB of 125 GB stay free (105 + 0). The 10G slot is no longer used then: `scripts/slot_run.sh` refuses to start while the FCOS scope's cap is above 95G. `dh5_latency_pause.sh` now resumes FCOS with the arguments of the run it stopped (95G / 85 during 1024, 105G / 95 during 2048). Background (analysis 11:28, calib task files only): in the 1024 RGC stage 216 tasks (P1869, P2641; estimate 42.5–51 GiB, measured ≤ 37.4 GiB, 8.6 h fusion time) ran without a second heavy task under the 85 GiB budget; a per-image rescaling of the estimates (first 10 tasks of an image) would have lowered them only to 46–49 GiB, so no change of the memory model.

## 2026-11-03 11:41 – Release regeneration covers the D-H5 tables; clean-room passed again (user)

- User: the regeneration script covers every paper table so that the clean-room byte comparison checks them; rerun the clean-room test after each extension. Now: `scripts/regenerate_paper.py` + `scripts/dota/d_dh5_tables.py` (5 D-H5 outputs), release ships it and `runs/dota/val/results_dh5.json` (`5509230`, `7855184`). Rebuild 343 files, 0 forbidden terms, 0 missing imports. **Clean-room (no GPU, slot) passed:** pytest 69 / 7 skipped, regeneration 15/15 commands, 67/67 outputs byte-identical.
- Still outside: the FCOS tables and the D-H5 latency table (not produced yet; added when they exist, then clean-room again – after the FCOS tuning or in a pause, since the slot is locked while FCOS runs at 105G); excluded as on 11-01: qualitative / volume / method figure data (need images or raw detections).

## 2026-11-03 11:47 – D-H5 latency: VS Code GPU process accepted as graphics-only; exclusive run started (user)

- Check 11:43: `nvidia-smi` lists no compute process and no "code" process; graphics-only (type G): gnome-shell, Xwayland, nautilus, firefox-esr, gnome-system-monitor. VS Code (started 08:10:30, `disable-hardware-acceleration: true` active) still runs a Chromium `--type=gpu-process` (PID 2735951) with `--use-gl=disabled` that holds the NVIDIA render node `/dev/dri/renderD129` open but no GPU context (not in `nvidia-smi`). Snapshot `runs/latency_host_a/gpu_procs_check_dh5_1143.txt`.
- **User decision (option 2):** this process counts as allowed (like the graphics-only desktop processes) and is recorded in setup_dh5.json; exclusivity rules otherwise unchanged (abort on any foreign compute process).
- Pause started 11:47:01 (`scripts/dota/dh5_latency_pause.sh`, FCOS scope stopped with launcher args `4 21 95G 2-23 85`, stage 1024 RGC, 13 344 task files stored before and after the stop; slot frozen by hold). Latency run started 11:47:21; end, rerun tasks and result in the next entry.

## 2026-11-03 12:02 – D-H5 latency measured (exclusive, all rows final); FCOS resumed

- **Pause:** FCOS stopped 11:47:01, resumed 11:59:44 with the unchanged launcher (`4 21 95G 2-23 85`; waiter `scahi_lim1024` restarted). Stored refined-grid tasks: 13 343 before and after the stop (the earlier count 13 344 included the `tight_model.json` marker); the resumed stage starts at "13 343 done" → **no stored task lost; only the tasks running at the stop (≤ 21 workers, about 3 busy at the time) are rerun.** Slot frozen 11:47:03 – 11:59:51.
- **Run** 11:47:32 – 11:59:43, 458 DOTA-val images, batch 1, default power limit (575 W); no compute process on the GPU at start and end; graphics-only processes and the VS Code gpu-process in `runs/latency_host_a/setup_dh5.json`; **all rows final**. Live-pass parity (CPU fusion of the live passes = cached path, 3 calib images × 2 rows): passed for both models. GPU fusion parity gate (`runs/gpu_fusion/parity_dh5.json`, 4 rows identical on every DOTA-val image) → GPU rows reported.
- **ms per image (mean; passes + merge step = total):**

  | Model | SAHI (frozen) | SAHI-F2 CPU | SAHI-F2 GPU | SAHI-RG CPU | SAHI-RG GPU |
  |---|---|---|---|---|---|
  | YOLOv10s | 180.1 + 17.0 = 197.1 | 145.1 + 100.5 = 245.6 | 145.1 + 12.6 = 157.8 | 145.1 + 189.8 = 334.9 | 145.1 + 12.7 = 157.8 |
  | YOLOv10m | 232.7 + 16.2 = 248.8 | 199.5 + 155.7 = 355.2 | 199.5 + 13.7 = 213.3 | 199.5 + 282.8 = 482.3 | 199.5 + 13.6 = 213.1 |

  Peak GPU memory of the fusion 1.48–1.80 GB. Files `runs/latency_host_a/dh5_latency.json`, `reports/dota/tab_dota_latency_host_a_dh5.{md,csv}`.

## 2026-11-03 12:03 – Release regeneration covers the D-H5 latency table; clean-room passed

- `scripts/regenerate_paper.py` also runs `scripts/latency_host_a_dh5.py report` (no GPU in that mode); inputs `runs/latency_host_a/dh5_latency.json`, `runs/gpu_fusion/parity_dh5.json` shipped (`66e842d`, `9c6d54d`). Rebuild 347 files, 0 forbidden terms, 0 missing imports. **Clean-room (no GPU, slot) passed:** pytest 69 / 7 skipped, 16/16 commands, 69/69 outputs byte-identical. Only the FCOS tables remain outside the comparison (besides the figures excluded since 11-01); added once they exist.

## 2026-11-03 14:30 – FCOS scheduler: back to the tight memory model after the planned pause (user)

- The restart after the planned D-H5 latency pause (11:47–11:59) found the `tight_model.json` marker of RGC 1024 and fell back to the conservative estimate, although there was no OOM. User decision: that fallback is not justified by the rule; the approved tight model (safety factor 1.25 unchanged) applies to the rest of RGC 1024 and the following stages. The fallback stays reserved for an actual OOM kill of the FCOS scope.
- Implementation (scheduling only, results unchanged): `scripts/dota/fcos_tuning_loop.sh` records each detected OOM kill of the scope in `runs/dota/fcos_logs/oom_kills.log` and passes it as `SCAHI_OOM_LOG`; `scahi/dota/pipeline.py` falls back only if the marker exists and an OOM kill is recorded since the marker's start. Without `SCAHI_OOM_LOG` the old behaviour holds (every marker counts as a kill). Planned stops (pauses, option A) write no record.
- Switched at a safe point: 14:28:53 planned stop of scope run-re598d318…, restart with the unchanged launcher args (4 21 95G 2-23 85) at once; 13528 RG task files stored before and after the stop, only in-flight tasks are rerun (recent RGC tasks 56 s median, 95 s max). Option-A waiter `scahi_lim1024` restarted. Log: `runs/dota/fcos_logs/tight_restart.log`.
- Resume line: "tight_model.json exists, no OOM kill recorded since (planned stop) -> tight model"; refit alpha 2.007, q999 1.103 (1986 points), open tasks max est 18.2 GiB (conservative 37.2).
- The release code changed (`pipeline.py`, launcher): the clean-room test is rerun in the slot when it is free again.

## 2026-11-03 14:45 – D-H5 latency: pass-time difference SAHI vs. SCAHI path explained in the report; release clean-room after the scheduler fix (user)

- User accepted the D-H5 latency (all rows final, GPU parity 0 differences) and asked why SAHI's passes take 180.1 / 232.7 ms (v10s / v10m) and the SCAHI path's 145.1 / 199.5 ms on the same passes.
- Profiling run, not reported as latency: `scripts/latency_dh5_profile.py` → `runs/latency_host_a/dh5_profile.json`. It ran on all 458 DOTA-val images on cores 0-1 next to the FCOS CPU tuning, with the GPU otherwise idle, as a slot scope (10G, oom_score_adj 1000) with the GPU visible, 14:35–14:42. The latency-run functions were instrumented with CUDA-synchronised wrappers.
- Result: the GPU-side work costs about the same in both paths (preprocessing, forward, head output; the SCAHI path's forward is even a bit slower). SAHI's surplus is CPU-side overhead of the library: the PIL image decode, one Python `ObjectPrediction` per detection (about 6100 per image), shifting and filtering in the slice loop, and conversion back to arrays. Under load the absolute values are larger than in the exclusive run, so the shares are what counts.
- Recorded in `reports/dota/tab_dota_latency_host_a_dh5.md`, generated by `latency_host_a_dh5.py report`; the CSV is unchanged. The section also carries the paper's wording rule: SCAHI's merge on the GPU vs. SAHI's merge on the CPU is no speed comparison. The valid statement: replacing SAHI's merge by SCAHI adds no substantial latency, and the passes dominate the cost.
- Release: `dh5_profile.json` is now an input (results.txt) and the profiling script is in the manifest. 349 files. Clean-room passed twice: after the scheduler fix (14:33, 16/16, 69/69) and after this extension (16/16 commands, 69/69 outputs byte-identical).

## 2026-11-03 15:11 – D-H5 latency into the exchange guide with the supplement sentence; FCOS without further pauses (user)

- No exclusive re-profiling: the shares of the non-exclusive profiling run are sufficient.
- Supplement sentence: "the extra pass time in SAHI is library overhead on the CPU, mainly image loading and per-detection Python objects". It is in `reports/FOR_CLAUDE_DH5.md` next to the paper's wording rule. That guide is generated by `scripts/make_exchange_zip_dh5.py` and now also lists the D-H5 latency values exactly as printed in the report. The placeholder "D-H5 latency is not measured yet" is removed.
- Exchange archive rebuilt in the slot: `exports/scahi_dh5_exchange_20261103_416713a7b8c4.zip`, 23.54 MB, 174 files. It adds the latency md/csv and `dh5_profile.json`. FOR_CLAUDE.md has 67 lines, nothing missing. The zip stays uncommitted.
- FCOS runs on without further pauses; the only pause left is the planned option-A pause (limit check 1024).

## 2026-11-03 15:29 – Plan change: FCOS moves host-a → host-b on Friday 2026-11-06; host-b is the only coding agent from Sunday (user)

- host-a is back around 2026-11-22. From Sunday 2026-11-08, host-b is the only coding agent. FCOS therefore moves to host-b on Friday, before the user leaves; I report by Thursday evening which time works.
- **Friday on host-a:**
  1. Stop the FCOS scope at a safe point, after the current tasks are stored, together with every other job on host-a (tmux `scahi_fcos_par`, `scahi_lim1024`, `scahi_after_fcos`, `scahi_fcos_progress`, `scahi_slot_wd`; slot jobs). Record in DECISIONS: stage, stored tasks, tasks in progress (lost).
  2. Build the complete transfer set:
     - git bundle;
     - all stored FCOS tuning tasks and partial stage results;
     - the FCOS caches (calib and DOTA-val, all levels);
     - FCOS weights and runs;
     - everything produced since the last migration (D-H5 caches and results, mapping-check caches, latency results);
     - a manifest with SHA-256;
     - HANDOFF section "host-a → host-b, Nov 6" with the exact resume procedure and every open task: FCOS stages, limit checks 1024 / 2048, freeze proposal (stop for approval), the single DOTA-val evaluation, final archive, release update incl. the FCOS tables in the regeneration and the clean-room test.
  3. Report the size of the transfer set early (Thursday), so the copy fits into Friday.
  4. After the copy, host-a stays stopped (no unattended jobs).
- **On host-b (from Sunday, for the host-b agent; goes into HANDOFF):**
  - environment check, then manifest verification;
  - replay gate on a few frozen rows;
  - equality test of a few FCOS tasks (host-b vs. the stored host-a values);
  - then resume FCOS from the stored tasks under the same rules (memory scope, tight model, option-3 timing).
  - A stage split across the two machines is recorded as such; each task times reference and variant on the same machine.
  - No FCOS latency in the host-a DOTA latency table; if measured on host-b, it is reported as a separate table.
- **Safe point:** a refined-grid stage is resumable per task (in flight about 1–3 min); a box-fusion stage (NMS / F2 / B-F2) is stored only as a whole and restarts after a stop. The stop is placed outside a running box-fusion stage, or its loss is recorded.
- **First size forecast (13:00 data, 2026-11-03):**
  - Repository tree now 193 GB.
  - New since the last migration (2026-10-31 evening): 129 GB. That is FCOS caches 64 GB (1024 and 2048, calib + val), FCOS RG tasks 46 GB, D-H5 fusion 4.6 GB, D-H5 caches ~5 GB, mapping-check caches 0.5 GB, FCOS weights / single passes ~3 GB.
  - By Friday about +50–70 GB more (RGB 1024, the start of the 2048 stages): delta ≈ 180–200 GB, whole tree ≈ 250–265 GB. The exact size follows on Thursday.

## 2026-11-03 15:51 – Addendum 5 (D-H6: Q4 for YOLO11s / YOLO12s on SAHI's passes) committed; SAHI raw calib done; chain started (user)

- **Addendum 5 committed:** `631f69e21c937a4c14bcf054e1487df9df880fc5` at 2026-11-03T15:45:51+01:00, before any result of these rows (`docs/PREREGISTRATION_DOTA.md`). Hypothesis D-H6: SAHI-RG − SAHI > 0.30 on DOTA-val, primary mAP50 without cap, Holm over YOLO11s / YOLO12s, holds if met for both. The D-H5 verdict is unchanged.
- **Implementation** in `scahi/dota/sahi_scahi.py`; YOLOv10 paths unchanged:
  - `A5_MODELS`;
  - `sahi_raw_single`: `pipeline.sahi_raw` without YOLOv10's o2o switch, so the predictor keeps Ultralytics' NMS (IoU 0.7, conf 0.001, max_det 200 000);
  - `a5_sahi_tune`: `pipeline.sahi_tune` unchanged, only the worker initializer keeps the processes on the slot cores 0–1 instead of pinning them to 16–23, which FCOS uses;
  - `nms_pass_dets`: Ultralytics' per-pass NMS from the cache, class-wise, descending score, clipped to the pass;
  - `frozen_sahi` / `overlap` read the calib selection of the SAHI tuning for the addendum-5 models;
  - tuning directory `<m>_1024/a5`.
  - `pipeline.py` is not edited while FCOS runs (its spawn workers import it).
  - GPU slot jobs: `scripts/slot_run_gpu.sh` (like `slot_run.sh`, GPU visible). Chain: `scripts/dota/run_a5.sh` (tmux `scahi_a5`, log `runs/slot/a5_chain.log`): SAHI tuning → caches calib + DOTA-val → gate → only if both gates passed, SCAHI tuning (budget 8.5 GiB).
- **SAHI raw calib** (GPU, 15:47–15:50): 3 overlaps × 2 models, 33–51 s each; at most 34 113–56 057 raw predictions per image.
- **Disclosed:** functional test of gate rule (1) for both models on 1 calib image per geometry, with the slice overlap set to 0.2 before the selection is known. Decoding and NMS were identical within tolerance (max 0.61 of the tolerance); no mAP was computed.
- **FCOS cap:** stays at 95G until addendum 5 is finished. `fcos_limit_1024_pause.sh` restarts after the option-A pause with 95G / 85 unless `runs/slot/a5_done` exists, then with 105G / 95. Waiter `scahi_lim1024` restarted 15:47.
- **Time estimate (grid known: 288 SAHI configs per model, as for YOLOv10):**
  - SAHI tuning ~0.5–1 h per model on 2 cores;
  - caches + gate ~1 h;
  - SCAHI F2 + RG tuning ~12–16 h for both models (D-H5: 12 h);
  - the freeze proposal is expected Wednesday 2026-11-04 midday to evening, so the Thursday deadline is feasible.

## 2026-11-03 19:21 – Addendum 5: Gate-Lauf abgebrochen (fehlende DOTA-val-Rohvorhersagen), deskriptiver val-Vergleich verschoben, Neustart ab Gate

- Kette `scahi_a5`: SAHI-Tuning fertig 16:14 (je Modell 288 Konfigurationen; gewählt für YOLO11s und YOLO12s jeweils
  Überlappung 0,2, NMS/IOU 0,4, conf 0,01; calib-mAP50 58,67 bzw. 60,32), Caches calib + DOTA-val fertig 16:19.
- Der Gate-Lauf brach um 16:19 nach dem deskriptiven calib-Vergleich ab (`FileNotFoundError`
  `yolo11s_1024/sahi_raw_val_o20.pkl`): `gate` las wie bei D-H5 auch SAHIs eigene DOTA-val-Rohvorhersagen, die es für die
  A5-Modelle nicht gibt; Addendum 5, Punkt 10, sieht sie erst nach der Freeze-Freigabe vor. Es wurde kein Gate-Urteil
  geschrieben; die Kette stoppte vor dem SCAHI-Tuning (wie vorgesehen). Die Stopp-Meldung „gate yolo11s not passed“ im
  Ketten-Log bedeutet hier: keine Gate-Datei, kein inhaltliches Durchfallen.
- Entscheidung: Das Gate-Urteil hängt nur an (1) Dekodierung und (2) calib-mAP50 (Addendum 5, Punkt 5) und bleibt
  unverändert. Der deskriptive DOTA-val-Vergleich (ohne Metrik, nicht Teil des Urteils) wird für die A5-Modelle mit den
  SAHI-DOTA-val-Rohvorhersagen im DOTA-val-Lauf nachgeholt (gate.json: `raw_vs_frozen_val` = „deferred …“).
  `run_a5.sh gate` setzt die Kette am Gate fort (SAHI-Tuning und Caches unverändert).

## 2026-11-03 19:23 – Addendum 5: Paritäts-Gate bestanden (beide Modelle), SCAHI-Tuning gestartet

- YOLO11s: Dekodierung ok (beide Geometrien, alle Prüfbilder); calib-mAP50 aus den Cache-Pässen 58,6717 gegen 58,6716 aus
  dem SAHI-Tuning, |Δ| = 0,00007 ≤ 0,01 → bestanden. Deskriptiv calib: 704 663 gegen 704 653 Rohdetektionen, 95/141
  Bilder gleiche Anzahl, Median-p99 der Boxabweichung 0,0002 px.
- YOLO12s: Dekodierung ok; calib-mAP50 60,3150 gegen 60,3150, |Δ| = 0 → bestanden. Deskriptiv calib: 623 566 gegen
  623 569, 103/141 Bilder gleiche Anzahl.
- SAHI-F2- und SAHI-RG-Tuning (CPU, 10G-Slot, Kerne 0–1) gestartet 19:22. Verlust durch den Abbruch ≈ 3 h; Freeze-Vorschlag
  weiter vor der Frist (Donnerstagabend) erwartet.

## 2026-11-03 22:04 – FCOS 1024: RGC fertig; Stufe RGB brach an der Speichermodell-Prüfung ab, Schätzung auf Grundlast begrenzt, Neustart

- RGC 1024 ist abgeschlossen (alle 15 369 Tasks, Stufe in tuning.json). Beim Start von RGB 1024 brach `tune_rg` um 22:01 an
  der Plausibilitätsprüfung des Speichermodells ab (`cache-size scaling underestimates: 0.8 > 0.7 GiB`), rc 1, kein
  OOM-Abbruch; der Launcher startete wie vorgesehen nicht neu, der Option-A-Waiter `scahi_lim1024` brach ab (Session
  weg). Kein RGB-Task war gestartet (nur das Speicherprofil `rg_mem_B.json`, 12 Punkte, alle 0,78 GiB).
- Ursache: Zeile B (ganzes Bild) braucht für die Fusion praktisch keinen Speicher; alle gemessenen Peaks (0,78 GiB) liegen
  unter der angenommenen Prozess-Grundlast `RG_BASE_GIB` = 1,0 GiB, (Peak − Grundlast) wird negativ, die Schätzung fällt
  unter den gemessenen Peak.
- Änderung (nur Planung, keine Tuning-Ergebnisse betroffen): `_rg_est` und `_rg_est_tight` werden nach unten auf
  `RG_BASE_GIB` begrenzt (konservativ: 1,0 ≥ 0,78 GiB); `_rg_tight_fit` liefert None (konservatives Modell), solange
  weniger als 20 Punkte im Fit-Bereich liegen. Prüfung reproduziert: Schätzung 1,0 gegen Peak 0,78 GiB → besteht.
  `pipeline.py` atomar ersetzt (die A5-Slot-Worker importieren es; sie nutzen diese Funktionen nicht).
- FCOS neu gestartet mit unveränderten Argumenten (4 21 95G 2-23 85; Deckel 95G bis Addendum 5 abgeschlossen),
  Option-A-Waiter `scahi_lim1024` neu gestartet. Dieselbe Begrenzung gilt später für RGB 2048.

## 2026-11-04 01:05 – FCOS 1024: RGB fertig, Option-A-Pause gestartet; Limit-Check 1024 nach Zeilen parallelisiert (Nutzer)

- RGB 1024 nach dem Neustart fertig 23:17 (15 369 Tasks, 4371 s); 1024-Auswahl geschrieben. Option-A-Pause automatisch
  23:17:32: FCOS-Scope gestoppt, keine 2048-Arbeit gespeichert oder verloren (2048 noch nicht begonnen).
- Der serielle Limit-Check (`fcos_limit_check.py calib 1024`, ein Prozess, ein Kern) rechnet zuerst die begrenzten
  Detektionen aller sechs Zeilen, dann die GPU-Pässe ohne Limit, dann die Neuberechnung je Zeile, und schreibt erst am
  Ende. Nach 1 h 46 min war keine Zeile fertig; Schätzung aus den Tuning-Zeiten (≈ 18 s je Bild und Konfiguration auf C,
  141 Bilder → ≈ 45 min je C-Zeile, vier C-Zeilen, dazu die Neuberechnung ohne Limit auf den dichtesten Bildern):
  deutlich mehr als 2 h Rest.
- Entscheidung (Nutzer, 2026-11-04): parallel nach Zeilen, nur Parallelisierung. Serieller Lauf 01:04 gestoppt (es war
  nichts gespeichert, daher nichts verloren außer der Rechenzeit). `fcos_limit_check.py calib 1024 --row ID` rechnet eine
  Zeile mit unveränderter Logik in ein Teilergebnis (`runs/dota/freeze/limit_calib_fcos_1024_parts/ID.json`), `--merge`
  setzt die Teile zur Datei des seriellen Laufs zusammen (gleiche Schlüssel, Zeilen in derselben Reihenfolge).
  `scripts/dota/fcos_limit_1024_par.sh` (tmux `scahi_lim1024`): alle sechs Zeilen gleichzeitig in einem Scope (95G, kein
  Swap, oom_score_adj 1000, Kerne 2–23, GPU); fehlende Zeilen werden einzeln nachgerechnet, fertige bleiben; danach Merge
  und FCOS 2048 wie bisher (Deckel 95G, solange Addendum 5 läuft).

## 2026-11-04 08:48 – Limit-Check 1024: fünf Zeilen fertig (alle Δ 0), NMS-Zeile nach Bildern parallelisiert (Nutzer); Addendum-5-Tuning fertig

- Parallellauf nach Zeilen (ab 01:04): A (8 Bilder betroffen), F2 (2), RG (2), B-F2 (0), B-RG (0), jeweils Δ mAP50 = 0;
  Spitzen ≈ 28 GB je C-Zeile, kein OOM.
- NMS-Zeile (IOS 0,8, conf 0,01): laut Tuning ≈ 1200 s Fusion je Bild, seriell ≈ 47 h nur für die begrenzten Detektionen,
  danach Pässe ohne Limit und Neuberechnung (im Tuning 59 Bilder am Limit). Nach 7 h 42 min (≈ ein Viertel) gestoppt.
- Entscheidung (Nutzer, 08:45): NMS-Zeile nach Bildern parallel, nur Parallelisierung. `fcos_limit_check.py --row`
  fusioniert Cache-Zeilen jetzt je Bild in eigenen Prozessen (`SCAHI_LIMIT_IMG_WORKERS`, Standard 20; dichteste Bilder
  zuerst; Ergebnis je Bild als parquet unter `runs/dota/freeze/limit_calib_fcos_1024_parts/<Zeile>/{limited,unlimited}/`,
  fertige Bilder bleiben bei einem Neustart erhalten), zusammengesetzt in der Bildreihenfolge des seriellen Aufrufs.
  `fuse` bearbeitet jedes Bild unabhängig; geprüft: B-F2 und F2 auf je 3 calib-Bildern, je Bild fusioniert ≡ seriell
  (DataFrame.equals). Die Runde 2 des Runners startete die NMS-Zeile 08:46 mit dem neuen Code; danach Merge und FCOS 2048
  wie bisher.
- Addendum 5: SAHI-F2- und SAHI-RG-Tuning beider Modelle fertig 07:11 (rc 0). Nächster Schritt: gemeinsamer
  Freeze-Vorschlag mit Replay auf calib, dann Stopp für die Freigabe.

## 2026-11-04 13:31 – Addendum 5: Freeze-Vorschlag (sechs Zeilen), Replay 6/6 exakt – Stopp für die Freigabe

- `docs/FREEZE_PROPOSAL_DOTA_A5.md`, `configs/dota/frozen_rows_a5.json` (`approved: false`); Replay auf calib über den
  DOTA-val-Pfad `sahi_scahi.a5_row_dets` (`runs/dota/freeze/replay_a5.json`): alle sechs Zeilen exakt (mAP50 und Cap 4000).
- calib-mAP50: YOLO11s SAHI 58,672 / SAHI-F2 62,089 / SAHI-RG 62,361 (Zeit 1,22× Ref.); YOLO12s SAHI 60,315 / SAHI-F2
  62,655 / SAHI-RG 62,831 (1,74×). Kein calib-Bild an der Grenze.
- Code: `a5_rows`, `a5_row_dets` (SAHI: Merge der Rohvorhersagen der gewählten Überlappung; DOTA-val-Rohvorhersagen erst
  nach der Freigabe), `a5_replay`; `dh5_rows`/`dh5_replay` nur parametrisiert (D-H5-Verhalten unverändert).
- Keine Latenz für diese Zeilen (kein exklusives Fenster vor Freitag). DOTA-val erst nach der Freigabe.

## 2026-11-04 14:01 – Addendum 5: Freeze freigegeben (Nutzer), vor jeder Berührung von DOTA-val

- Freigabe (Nutzer): sechs Zeilen wie in `docs/FREEZE_PROPOSAL_DOTA_A5.md` (d8c3bc1); `configs/dota/frozen_rows_a5.json`
  `approved: true`. Berichtsregeln wie D-H5: alle sechs Zeilen, kein Nachtuning; primär mAP50 ohne Cap; sekundär Cap 4000,
  „alle Objekte“, AP@[.5:.95], Größenbins, Recall dicht/isoliert; D-H6 = SAHI-RG − SAHI, gepaarter Bootstrap, Holm über
  YOLO11s und YOLO12s, gilt, wenn für beide erfüllt; SAHI-F2 − SAHI sekundär. Sensitivitätstabelle (Varianten 2,16 / 2,17)
  bleibt im Bericht. D-H5-Urteil (YOLOv10) unverändert. Keine Latenz für diese Zeilen.
- Danach: SAHI-Rohvorhersagen DOTA-val (gewählte Überlappung 0,2), der aus dem Gate verschobene deskriptive Vergleich,
  die einmalige DOTA-val-Auswertung, Bericht und Tabellen.

## 2026-11-04 14:42 – Limit-Check FCOS 1024 abgeschlossen: Δ mAP50 = 0 in allen sechs Zeilen, Replay exakt; FCOS 2048 läuft

- Ergebnis (`runs/dota/freeze/limit_calib_fcos_1024.json`, aus den Teilen zusammengesetzt; calib, Grenze 200 000 gegen
  10 000 000):

  | Zeile | Bilder betroffen (Bild / Pass) | mAP50 begrenzt | ohne Grenze | Δ | Replay |
  |---|---|---|---|---|---|
  | A_fcos_1024 | 8 (8 / 0) | 42,9003 | 42,9003 | 0 | exakt |
  | NMS-o2m_fcos_1024 | 59 (59 / 10) | 53,3926 | 53,3926 | 0 | exakt |
  | F2_fcos_1024 | 2 (2 / 0) | 54,5504 | 54,5504 | 0 | exakt |
  | RG_fcos_1024 | 2 (2 / 0) | 54,5504 | 54,5504 | 0 | exakt |
  | B-F2_fcos_1024 | 0 | 42,2254 | 42,2254 | 0 | exakt |
  | B-RG_fcos_1024 | 0 | 42,6856 | 42,6856 | 0 | exakt |

  Pässe ohne Limit (GPU, 59 Bilder × 2 Ebenen = 118): dichte Ausgaben identisch mit den gespeicherten Caches (118/118).
  Die Grenze 200 000 ändert für die 1024-Zeilen kein mAP50.
- Laufzeit der Pause 23:17 (03.11.) – 14:40 (04.11.): serieller Lauf 1 h 46 min (verworfen), Zeilen parallel ab 01:04,
  NMS-Zeile seriell bis 08:46 (verworfen), danach nach Bildern parallel (begrenzt 2,9 h, ohne Grenze 2,9 h).
- FCOS 2048 um 14:40 vom Runner neu gestartet (Deckel 95G, Budget 85, solange Addendum 5 läuft). Der 2048-Teil des
  Limit-Checks folgt nach den 2048-Stufen (auf host-b, HANDOFF); dort von Beginn an nach Zeilen und Bildern parallel.

## 2026-11-04 16:44 – Addendum 5 DOTA-val done: D-H6 holds for YOLO11s and YOLO12s

- Chain `scahi_a5val` ended 16:43:51 rc 0 (SAHI raw val GPU 14:02–14:06, dets + single evaluation of the 6 frozen rows, scope 10 GB, cores 0–1, no OOM). Result `runs/dota/val/results_a5.json`; tables `reports/dota/tab_dota_a5.md` + CSVs (`scripts/dota/d_a5_tables.py`).
- Primary mAP50 (no cap), SAHI / SAHI-F2 / SAHI-RG: YOLO11s 57.90 / 61.36 / 61.87; YOLO12s 60.21 / 63.13 / 63.30.
- D-H6 (SAHI-RG − SAHI, mAP50, ε 0.30, paired bootstrap, Holm over both models): YOLO11s +3.97 [+2.93, +5.06], Holm p 0.002; YOLO12s +3.10 [+2.23, +4.08], Holm p 0.002 → met for both → **D-H6 holds**. Secondaries (cap 4000, all objects, AP@[.5:.95], vehicles, crowd/isolated recall) all positive with CI above 0 for SAHI-RG − SAHI.
- Secondary SAHI-F2 − SAHI: +3.46 (YOLO11s), +2.92 (YOLO12s) mAP50; crowd recall R50 −17.4 / −19.3 (as with D-H5); no verdict.
- Deferred descriptive raw comparison (DOTA-val): cache path vs. SAHI raw 2,498,302 vs 2,498,323 (YOLO11s), 2,202,498 vs 2,202,499 (YOLO12s); box p99 0.0002 px.
- "Images at the limit" in the log for the SAHI rows (2 / 1: P1384, P2689) counts images with ≥ PIPE_MAX_DET detections; SAHI's merged output is not truncated there (max 28,184 / 21,917 per image). SCAHI rows 0 at 200 000.
- Sensitivity table (2.156 / 2.172 variants) kept in the report. No latency for these rows. YOLOv10 D-H5 verdict unchanged.

## 2026-11-04 18:04 – Counter "images at the limit" relabelled for the A5 SAHI rows (user); FCOS 105G at the next planned restart

- User: the per-image counter of the A5 SAHI rows counted images with >= 20,000 detections (PIPE_MAX_DET, default of `pipeline.at_limit`), while the pipeline limit is 200,000; relabel everywhere so nobody reads it as truncation.
- Done: `sahi_scahi.a5_row_dets` counts SAHI rows against 200,000 (`images_at_limit`) and keeps the 20,000 count as `images_ge_20000_dets` (informational); the val log line (`final._write_dets`) names it as such. Stored `runs/dota/val/results_a5.json` and `runs/dota/val/dets/SAHI_yolo1{1,2}s.json` relabelled from the parquet files (images_at_limit 2 / 1 → 0 at 200,000; images_ge_20000_dets 2 / 1; max 28,184 / 21,917 per image; note field); no metric changed (copy before: `results_a5.before_relabel.json`; CSV tables byte-identical). `reports/dota/tab_dota_a5.md` has a line with both counts. Note appended to `runs/slot/a5_val.log`; the earlier DECISIONS entry already explains it.
- Check of earlier tables: YOLOv10 val rows (`results.json`, `results_dh5.json`, `d_results*.py`) are counted against the frozen pipeline limit 200,000 (`final.row_dets` gets the limit) → correct label; only `RG_yolov10m_2048` has 1 (P1384, genuine, covered by `d_limit_check.py`). Calib "Grenze" columns of the freeze proposals (YOLOv10, D-H5, A5) count at the calib selection limit 20,000, all SAHI rows 0 (max ≤ 9,722 per image) → nothing misreadable, approved documents unchanged.
- User: no FCOS restart only for the 105G cap; it applies at the next planned restart. A5 release inputs + clean-room test added to the HANDOFF list (section 12 draft).

## 2026-11-04 18:30 – Exchange archive for the paper agent after addendum 5 (user)

- `scripts/make_exchange_zip_a5.py` (D-H5 archive + addendum 5: results, freeze proposal, frozen rows, replay, gates, per-row evaluations, `tab_dota_a5*`), guide `reports/FOR_CLAUDE_A5.md` (61 lines, generated from the result files; states: D-H6 holds, no A5 latency, no FCOS result, counter label of the SAHI rows).
- Built in the slot: `exports/scahi_a5_exchange_20261104_b7b3c3c08f0d.zip`, 23.60 MB, 193 files, nothing missing (PNGs at 150 dpi by the size fallback, PDFs full). Zip uncommitted; goes into the host-b transfer set.

## 2026-11-04 18:51 – Release with addendum 5, clean-room passed; supplementary code zip (user; slot only, FCOS untouched)

- User: until the handover on Friday host-a stays dedicated to FCOS (no interruption, no new experiments); only the 10G slot (cores 0–1) for release, supplementary zip and handover.
- Release (`3254608`): manifest + `configs/dota/frozen_rows_a5.json`, `scripts/dota/d_freeze_proposal_a5.py`, `scripts/dota/d_a5_tables.py`; results `runs/dota/val/results_a5.json`, `runs/dota/fusion/yolo1{1,2}s_1024/{a5/tuning.json,sahi_scahi_gate.json}`; `scripts/regenerate_paper.py` runs `d_a5_tables.py` (6 more outputs). Build: 363 files, 0 forbidden terms, 0 missing imports. **Clean-room (fresh env, no GPU, slot) passed:** pytest 69 passed / 7 skipped, 17/17 commands, 75/75 outputs byte-identical, no research-path references; env vs. lock only pip/setuptools/wheel/pyqt5-qt5 extra (as before).
- Supplementary code zip (the conference: code may be uploaded as supplementary material; no external links that expand the content; author guidelines state no MB limit): `scripts/make_supplementary_zip.py` packs the generated release as `exports/supplementary/code.zip` (top folder `code/`), deterministic, no metadata (date 1980-01-01, mode 0644, no extra fields / comments / directory entries). Scan on the zip contents itself: 2.06 MB (10.3 MB unpacked), 364 files, 0 forbidden terms (names + contents), 0 binary members (all UTF-8 text, no PDFs / images / weights), 0 metadata issues. URLs: 6, all third-party (DOTA devkit source page and raw file, PyTorch wheel index and CUDA allocator doc, FSF / GNU licence text); none to own code. Record `runs/release/supplementary.json` (private); zip uncommitted.

## 2026-11-04 21:11 – Handover preparation: limit check 2048 parallel on host-a after the tuning; Friday copy plan (user)

- User (question answered): when the FCOS 2048 tuning ends before the handover, the 2048 limit check runs on host-a **row- and image-parallel**, never serial. `scripts/dota/fcos_limit_2048_par.sh` (one row at a time, per-image fusion with 20 workers, retried with 10 / 5 on failure; scope 95G, no swap, oom_score_adj 1000, cores 2–23; per-image parts stored, so a stop at the handover loses only images in flight; then `--merge`). `scripts/dota/after_fcos_tuning.sh` calls it instead of the serial check; waiter `scahi_after_fcos` restarted 21:10 (it is only a waiter; FCOS untouched). Unfinished rows / images continue on host-b (HANDOFF 12.4).
- User: departure Friday ~20:00, Saturday morning possible if problems. Plan: FCOS / limit-check stop Friday ~17:30 at a safe point (refined-grid stage or limit check: per task / image; box-fusion stage: stop only outside it or record the loss), final manifest + git bundle (~10–15 min, manifest pre-hashed, only new files hashed), copy to the external drive until ~19:00–19:15, buffer to 20:00.
- Transfer tool `scripts/migration/transfer_host_a_host_b.py` (delta since 2026-10-31 21:00 + FCOS COCO checkpoint + bundle; manifest `runs/migration/manifest_host_a_host_b.tsv` with SHA-256 / link targets; `verify` on host-b). State 2026-11-04 19:00: 166.0 GB, 258 102 files, 132 777 symlinks; pre-hashed in the slot in 3.5 min. Expected by Friday: + 2048 RG tasks (1024: 64 GB), + limit-check 2048 parts / no-limit caches (1024: 2.6 + 6.1 GB) → about 240–290 GB; exact size Thursday evening.
- HANDOFF section 12 drafted (`6a9bc53`): state at handover (filled in at the stop), transfer set, host-b procedure (copy, env check, verify, replay gate, FCOS equality test, resume), open tasks.
- Observation FCOS 2048 RGC (no action, no interruption): heavy tasks (est. up to 81.9 GiB of the 85 GiB budget, P1139 cell 13) run alone by the memory rule, so phases with load ≈ 1 are expected; the "[rg] WARNING peak 13.5 > est" lines (1 506, all exactly 13.5 GiB) are a measurement artefact: `ru_maxrss` of a freshly spawned worker starts at the parent's high-water mark (parent `fcos_run.py` VmHWM 14 183 168 kB = 13.53 GiB from the in-process NMS / F2 stages; the workers' own VmHWM is far lower, used memory 25 GB). It only makes the refitted model more conservative (fewer parallel tasks), never less safe; it disappears with a fresh process (e.g. the resume on host-b). Checked 21:20 via /proc/<pid>/status.

## 2026-11-04 21:59 – Slot watchdog ignores the artefact peak warnings; handover drive check postponed to Friday (user)

- The slot watchdog had frozen the slot at 19:39 on the first FCOS "[rg] WARNING peak 13.5 > est" (artefact, entry 21:20). `scripts/slot_watchdog.sh` now counts a warning only if its peak exceeds the FCOS parent's VmHWM (+0.05 GiB); the 85 / 75 GB anon rule is unchanged. Restarted 21:52 (tmux `scahi_slot_wd`); FCOS untouched.
- Target drive `<external drive>/Datasets` (USB HDD WD 5 TB, NTFS via ntfs-3g / fuseblk, 308 GB free = less than 290 GB + 20 %): tar write of 20.6 GB real cache files at 76 MB/s (test file removed). NTFS → method would be tar archives (symlinks kept). User: stop here, focus on the experiments; drive, space and method are decided on Friday.

## 2026-11-05 01:11 – FCOS 2048 RGC: scheduler fix (min model + own peak measurement), restart with 105G / budget 95 (user)
- Finding (00:50): RGC 2048 ran effectively serially (about 21 tasks/h, one core busy, load 1.5). Cause: every task
  peak was measured as ru_maxrss, which a spawned worker inherits from the parent's high-water mark (13.53 GiB); 1 507
  small-image tasks read 13.5 GiB. The tight memory model refitted at 1 500 tasks on these values (alpha 0.157 instead
  of about 1) and estimated all 13 759 open tasks at 63–83 GiB -> every task alone under the 85 GiB budget.
- Simulation of the open tasks (greedy in task order as tune_rg, 21 workers; task times from the measured reference
  tasks, variant / reference 1.394 on P1139): serial sum 78.2 h. Budget 85: tight (current) 78.2 h, conservative 31.1 h,
  min 31.1 h. Budget 95: tight 77.6 h, conservative 26.0 h, min 26.0 h.
- User decision (approved pending the simulation; gain substantial -> implemented), commit 9aa4e9f, scheduling only (task
  outputs and results unchanged):
  1. per-task estimate = min(tight, conservative); OOM fallback rule unchanged (conservative only after an actual OOM
     kill of the FCOS scope);
  2. task peak = the worker's own VmHWM (/proc/self/status, reset at exec), recorded with `peak_src: VmHWM`; checked:
     parent with 6.15 GiB peak -> spawned worker VmHWM 0.15, ru_maxrss 6.15. Refit leaves out legacy records (no
     `peak_src`) at the inherited floor (smallest legacy value + 0.05 GiB, incl. the profile points; the scaling-check
     point P1791 13.53 is such a value and is an upper bound only, so the conservative assert still holds). Currently 105
     of 1 616 records usable -> no tight fit (needs 1 000) -> conservative estimate until enough clean records exist.
  3. restart at a safe point: scope stopped 01:08:55 directly after task 104/P1139 was stored (nothing in flight lost
     except the tasks running alongside; RG is resumable per task), restarted 01:09 with
     `scripts/dota/run_fcos_tuning.sh 4 21 105G 2-23 95` (the A5 slot is no longer needed; while the cap is above 95G
     scripts/slot_run.sh refuses slot jobs). Waiter `scahi_after_fcos` restarted 01:09. Start line: "15369 tasks,
     1617 done, budget 95.0 GiB, tight model marker, no OOM since".
- Expected (simulation, budget 95): RGC about 26 h -> end about Fri 2026-11-06 03:00; RGB (row B, small peaks, 1024 took
  4 371 s) about 1–2 h after it -> FCOS 2048 tuning end about Fri 04:00–05:00; then the 2048 limit check starts
  automatically. Throughput is reported after about 1 h.

## 2026-11-05 01:20 – Friday plan revised (user); freeze proposal chained after the limit check 2048; disk test stopped
- Plan (user): (1) disk test; (2) after the limit check 2048 the FCOS freeze proposal with the calib replay on host-a,
  reported immediately; if it reaches the user by about 11:00 Friday and is approved, the single DOTA-val evaluation on
  host-a, stopped cleanly at 16:00 if not finished (finished rows kept, rest on host-b); later -> no DOTA-val on host-a,
  proposal + state to host-b; (3) stop for the handover at 16:30 (not 17:30) unless the disk check shows hashing + copy +
  verification fit comfortably into a later start; then final manifest (new files only), git bundle, copy, verification
  on the disk before departure. Recorded in HANDOFF 12.0.
- `scripts/dota/after_fcos_tuning.sh`: after the limit check 2048 merge (runs/dota/freeze/limit_calib_fcos_2048.json
  present) it runs `scripts/dota/d_freeze_proposal_fcos.py` (approved: false; reads no DOTA-val). Waiter restarted 01:17.
- Disk test: started outside the slot in an own scope (4G, no swap, oom_score_adj 1000, cores 0-1, nice 19, ionice 2/7),
  then stopped at the user's request ("Speicherbedarf zum Kopieren checken wir Freitag ... Konzentration auf die
  Experimente"); test folder removed, target folder empty. Measured before the stop (facts for Friday): 344 GiB free;
  subset 6.44 GB real files + 9 869 symlinks (15 841 entries): rsync -a 135 s (about 48 MB/s), rc 0, copy identical in
  content, size, mtime and link targets, but the read-back came from the page cache (not a disk read); tar 109 s (about
  59 MB/s); tar verification not run. Transfer set now 171.3 GB (261 372 files, 132 777 symlinks), expected about 205 GB
  by Friday.
- Expected timing: limit check 2048 from about Fri 04–05 (1024 took about 20 min for the five small rows, the NMS row
  about 6 h image-parallel) -> freeze proposal about Fri 06:00–12:00; whether it reaches the user by 11:00 is open.

## 2026-11-05 09:42 – FCOS DOTA-val: conditional advance approval by the user, automatic start after the freeze proposal
- User: "ich gebe DOTA frei" (09:3x); asked whether it can run now: no – there is no freeze yet (the 2048 rows are selected
  only after the tuning, then the limit check 2048 with the calib replay; evaluating 1024 now and 2048 later would break
  "DOTA-val once, frozen"), and host-a stays with FCOS until then. User: "starte automatisch".
- Advance approval with conditions (proposed by me, accepted by the user): the single DOTA-val evaluation of the FCOS rows
  starts automatically once the freeze proposal exists, if (a) every row of `configs/dota/frozen_rows_fcos.json` has a
  calib replay with |Δ| = 0, (b) every row meets the limit rule |Δ mAP50| < 0.05, (c) both sizes are present. Otherwise
  nothing is approved and the user decides. The approval is written into the freeze file (`approved_at`, `approval`)
  and committed before any DOTA-val detection is computed.
- Implementation (no change to any row, selection, metric or rule):
  - `scripts/dota/fcos_val.py` – `approve` (checks the conditions; tested on a scratch freeze file: fails without 2048
    rows and with a row lacking the replay, passes when all are met), `gpu` (= `scahi.dota.final.val_gpu` on the FCOS
    freeze: rows A, X), `cpu ROW` (cache row fused by image in parallel – the per-image path of
    `scripts/dota/fcos_limit_check.py`, whose calib replay through this path was exact; parts kept in
    runs/dota/val/parts/<id>, concatenated in image order, written once as runs/dota/val/dets/<id>.parquet), `eval`.
  - `scahi.dota.final.val_eval_fcos` – addendum 3 item 9: D-H1 A_2048 − NMS-o2m_1024, D-H2a/b RG_s − NMS-o2m_s, D-H3
    RG_{S_max} − A_{S_max} (S_max 2048); one model, no Holm: met = criterion (Δ > 0.30; D-H3 also lower CI bound > 0) and
    unadjusted one-sided p ≤ 0.05; secondaries as in SECONDARY without o2o / SAHI; descriptively the count of the five
    DOTA detectors meeting each hypothesis (earlier verdicts read unchanged). -> runs/dota/val/results_fcos.json.
  - `scahi.dota.pipeline._cache_bytes` takes the split (default calib, unchanged); used only to submit the densest val
    images first.
  - `scripts/dota/fcos_val_auto.sh`, called by `scripts/dota/after_fcos_tuning.sh` after the proposal: approve + commit
    -> GPU rows -> cache rows in the order RG_2048 (D-H3 first), F2 / B rows 2048, then 1024, NMS-o2m_2048,
    NMS-o2m_1024 (box fusion last), each with 20 image workers, retried with 10 / 5 / 2 / 1 after a failure -> evaluation
    only when every row exists. Every step in its own scope (95G, no swap, oom_score_adj 1000, cores 2–23) with
    RuntimeMaxSec up to **Fri 2026-11-06 16:00**; then a clean stop, finished rows and images kept, rest on host-b.
    Waiter `scahi_after_fcos` restarted.
- Expected duration (from the calib timings; DOTA-val has 458 images, the C caches are 3.2× the calib volume, the densest
  image is as dense as calib's): the SCAHI rows are cheap (calib RG 1024 7.2 s per image); the box-fusion rows are not
  (calib NMS-o2m_1024 1 200 s per image, image-parallel limit check on calib about 6 h). So the GPU rows and the eight
  SCAHI rows should finish on Friday; NMS-o2m_2048 maybe; NMS-o2m_1024 most likely not before 16:00. Then the evaluation
  (all hypotheses) runs on host-b after the rest; no partial verdicts are reported.

## 2026-11-05 14:29 – FCOS DOTA-val: automatic approval removed; the approval is the authors' decision

- User: remove the automatic approval. The approval of a freeze before a held-out evaluation is a human decision by the
  authors, as for every earlier freeze; the paper's LLM Usage Statement states that the single authorized evaluation on
  held-out data was decided by the authors. Replaces the conditional advance approval of this morning (entry
  "FCOS DOTA-val: conditional advance approval …", commit `d97b8f7`). **That automatic approval was never used:** no
  freeze proposal exists yet (the FCOS 2048 tuning is still running), `configs/dota/frozen_rows_fcos.json` does not
  exist, and no DOTA-val detection of FCOS has been produced.
- The checks stay, as an automatic **pre-check reported in the freeze proposal**; they never set `approved`:
  exact calib replay of every row (|Δ| = 0), limit rule |Δ mAP50| < 0.05 for every row, both sizes (1024, 2048) present.
  `scripts/dota/d_freeze_proposal_fcos.py` writes the result into the proposal (section "Automatische Vorprüfung") and
  into the freeze file (`precheck: {all_met, failed}`, `approved: false`). `scripts/dota/fcos_val.py precheck` repeats
  it (report only). Tested on synthetic rows: all met -> empty; replay ≠ 0, limit rule, missing row, missing size -> listed.
- Approval: the user sets it on Friday morning, or tells me to; then `scripts/dota/fcos_val.py approve NOTE` (the user's
  words as the note; refuses an empty note or a second approval) and a commit of the freeze file. No automatic commit.
- The evaluation chain stays prepared: `scripts/dota/after_fcos_tuning.sh` (waiter `scahi_after_fcos`, restarted)
  writes the proposal, then waits until `approved: true` is committed (working file identical to HEAD); only then
  `scripts/dota/fcos_val_auto.sh`, which itself refuses without the committed approval. No approval before Fri 16:00 ->
  no DOTA-val on host-a (proposal + state to host-b). Unchanged: row order (GPU rows, RG_2048 first for D-H3, SCAHI rows,
  box fusion last), scopes, hard end **Fri 2026-11-06 16:00**, finished rows / images kept, unfinished rows on host-b.

## 2026-11-05 18:47 – Disk test for the handover copy (measured; stop time open for the user)

- User: disk test today, outside the slot, not waiting for FCOS; then confirm the Friday stop time from the measured times.
  Run in its own scope (4G, no swap, oom_score_adj 1000, cores 0-1, nice 19, ionice 2/7); FCOS unaffected.
- Disk: `/dev/sdb1`, WD WD50NDZW 5 TB USB HDD (rotational), NTFS through ntfs-3g (fuseblk); free 368.9 GB (343.5 GiB).
  Transfer set now 229.9 GB (288 475 files, 132 777 symlinks); expected 240–290 GB, + 20 % = 288–348 GB < 368.9 GB
  (fits; at the upper end about 21 GB to spare).
- Real subset (6.44 GB, 15 841 entries incl. 9 869 symlinks), written to `python_lab/scahi_2026_11_06/_disktest`:
  `rsync -a` 140 s incl. sync = 45.9 MB/s, rc 0, no errors; `tar -cf` 105 s = 61.6 MB/s, rc 0. Verification read from
  the disk (cache dropped by unmount + mount): rsync copy SHA-256 / size / mtime / link targets all identical (294 s =
  21.9 MB/s); `tar -d` against the source rc 0, 0 differences (272 s = 23.8 MB/s). Test folder removed.
- **Incident (mine):** the remount through udisks first chose the kernel driver ntfs3, which refused the volume; mounted
  again with ntfs-3g read-write (`udisksctl mount -t ntfs`), contents intact, but now at **<external drive>/Datasets1**
  (the original mount at <external drive>/Datasets was made by root; that directory remains, owned by root). Back to the
  old path only with sudo: `sudo umount <external drive>/Datasets1 && sudo mount -t ntfs-3g /dev/sdb1 <external drive>/Datasets`.
- Consequence: reading back from the disk (~23 MB/s) is the bottleneck. Copy + verification of the whole set after the
  stop: rsync 4.4 h (240 GB) – 5.3 h (290 GB); tar 4.0–4.8 h; plus manifest + bundle ≈ 0.3 h. With a stop at 16:30 that
  ends ≈ 21:00–22:00, after the planned departure ≈ 20:00. Options proposed to the user (decision open): (a) pre-copy
  now with `rsync -a` everything already final, verified tonight, so Friday copies and verifies only the delta (stop
  16:30 holds); (b) without pre-copy, stop ≈ 14:30.

## 2026-11-05 21:36 – Friday schedule: experiments first, copy only after the stop (no pre-copy)

- User: no copying for now; experiments first. The pre-copy (option a of the disk-test entry) is not started; the
  external disk stays untouched until the stop. Option b (stop ≈ 14:30) is not taken either.
- Priority until Friday: finish the FCOS 2048 tuning, the 2048 limit check and the freeze proposal (`approved: false`,
  with the pre-check); after the user's approval on Friday morning, the DOTA-val evaluation until the hard end (16:00,
  or the time the user gives); finished rows kept, the rest on host-b.
- After the experiments have stopped (unattended, may run into the night): final manifest, git bundle, **one**
  `rsync -a` run to `<external drive>/Datasets1/python_lab/scahi_2026_11_06`, then the full verification from the disk
  after unmount / mount with ntfs-3g (`udisksctl mount -b /dev/sdb1 -t ntfs`), then `sync`, unmount, power-off of the
  disk and a "safe to unplug" report. The user unplugs the disk only after that report. Expected ≈ 4.7–5.6 h after the
  stop (copy 1.5–1.8 h, verification 2.9–3.5 h for 240–290 GB, manifest + bundle ≈ 0.3 h).
- The stop at 16:30 of the earlier Friday plan is replaced by this; HANDOFF 12.0 items 3–5 updated.

## 2026-11-05 21:42 – Final Friday timing: stop 18:30, "safe to unplug" by 21:00; host-b from Saturday

- User: departure Friday between 21:00 and 22:00, plan for 21:00. Stop of all experiments **Fri 2026-11-06 18:30**
  (clean, finished rows / images kept); then final manifest, git bundle, one `rsync -a` run to
  `<external drive>/Datasets1/python_lab/scahi_2026_11_06`, verification of the **irreplaceable** files from the disk
  (unmount / mount with ntfs-3g), sync, unmount, power-off, "safe to unplug" by 21:00 at the latest. If the estimate for
  copy + verification exceeds 2.5 h, the stop moves earlier and the user is told in the morning. host-b takes over from
  **Saturday 2026-11-07** (not Sunday). Replaces the "copy after the stop, may run into the night" schedule (previous
  entry) and its full verification on the disk; the full verification (caches included) moves to host-b.
- Hard end of the DOTA-val chain moved from 16:00 to 18:30 (`scripts/dota/fcos_val_auto.sh`, approval wait in
  `scripts/dota/after_fcos_tuning.sh`); waiter `scahi_after_fcos` restarted (still waiting for the end of the tuning).
  No change to approval, row order or rules.
- Classification for the disk verification (mine, stated so the user can correct it): caches = `cache/`,
  `cache_ualigned/`, `runs/dota/fcos_nolimit_cache/` and the RG task results `runs/dota/fusion/*/rg_tasks/` (recomputable
  from caches + code); everything else in the manifest is irreplaceable (weights, training, freeze / limit-check
  artefacts, DOTA-val detections and parts, results, logs, exports, the external checkpoint, the bundle; all symlinks
  by target). Now (21:40): 256.2 GB in total, irreplaceable 16.2 GB (151 104 files + 132 777 symlinks), caches 240.0 GB.
- Estimate (measured 05.11.: write 45.9 MB/s, read-back 22–24 MB/s), expected ≈ 285 GB at the stop (RGB 2048 rest
  ≈ +22 GB, limit check 2048 and DOTA-val ≈ +5–8 GB): rest of the manifest + bundle ≈ 0.15 h (manifest refreshed in my
  own scope during Friday so only the rest is hashed at the stop), copy ≈ 1.75 h, remount + verification of ≈ 20 GB ≈
  0.3 h, sync / unmount / power-off ≈ 0.05 h → ≈ 2.25 h, "safe to unplug" ≈ 20:45. Within 2.5 h, but with little
  margin; re-estimated Friday morning from the measured size and reported (stop earlier if > 2.5 h).
- HANDOFF 12.0 (items 2–4), 12.1 (stop state to fill at 18:30, host-b from Saturday) and 12.4 (open tasks on host-b: full
  verification of the caches, rest of the FCOS DOTA-val evaluation if unfinished, release / clean-room with FCOS,
  final archive) updated.

## 2026-11-05 22:24 – fig_method (Fig. 1): small fixes, regenerated

- User: small fixes in the TikZ source, regenerate PDF + PNG, commit; few seconds of CPU outside the slot with nice
  (`scripts/make_figures.py method`); FCOS not touched.
  1. Cell factor in the paper's notation: "cell $g_j = \max(1, \sigma_j / f)$" (was c; the repository docs call it κ /
     `cell_factor`, unchanged there).
  2. "(pyramid)" removed.
  3. Spacing around "=": the source had no extra spaces; the wide gaps came from TikZ nodes with a fixed text width,
     which stretch the stretchable math glue (`\thickmuskip` plus 5mu). Fixed for the whole figure by math spacing
     without stretch (`\thickmuskip=5mu`, `\medmuskip=4mu`); this also normalises the same stretched gaps in
     $S = \sum w p / (C+\epsilon)$, $p \ge 10^{-3}$ and $z \in \{\dots\}$. No symbol or formula changed by it.
  4. Notation check against the paper's terms: further mismatches reported to the user, **not changed** (user decides).
- Next exchange archive: must contain the new `reports/figures/paper/fig_method.png`, `tex/fig_method.pdf`,
  `tex/fig_method.tex` (`scripts/make_exchange_zip_paper.py` lists them; the A5 template used for the final archive does
  not – add them there). Noted in HANDOFF 12.4.

## 2026-11-05 22:28 – Qualitative DOTA crops for the main paper: rule fixed before rendering (candidates; authors decide)

- User: 3–4 candidate crops of crowded scenes from the existing qualitative DOTA material, each a side-by-side pair
  box fusion | SCAHI, only from images selected by the recorded rule (entry 2026-10-30 ~18:30; no new image search).
  CPU only, nice, outside the slot; FCOS not touched. Exploratory, post hoc (as the source figure).
- **Fixed before rendering:**
  - Images: only the five of `reports/figures/dota/fig_dota_qualitative.csv` – P0199 (quartile 1), P0454 (quartile 2),
    P0749 (quartile 3), P0683 (quartile 4), P1478 (most vehicles). Methods, detections and operating points unchanged:
    YOLOv10s 2048, frozen DOTA-val rows NMS-o2m_yolov10s_2048 ("box fusion") and RG_yolov10s_2048 ("SCAHI", refined
    grid), thresholds from `fig_dota_qualitative_operating_points.json` (calib precision 0.80: box fusion s ≥ 0.499,
    SCAHI s ≥ 0.305). Per-image AP50 of both methods from `fig_dota_qualitative_per_image_dAP.csv` (devkit port, mAP50
    of the image).
  - Crop per image: square window of 384 × 384 px (original pixels; readable at column width) with the most
    non-difficult small-vehicle + large-vehicle GT centres, stride 16 px, ties: smaller y, then smaller x; same window
    for both methods.
  - Candidates: the 4 images with the most such vehicle centres in their window (ties: smaller image name); the fifth
    is reported, not rendered. Order of the files = that rank.
  - Drawing: matching on the whole image (devkit rule, IoU 0.5, greedy per class in score order) as in
    `scripts/dota/d_qualitative.py`; TP green, FP red (solid), FN yellow dashed; difficult GT and matches on it not
    drawn; line width 0.5 pt. Titles only "box fusion" and "SCAHI"; no counts or internal labels in the image.
  - One file per candidate: `reports/figures/dota/fig_dota_crop_<rank>_<image>.{pdf,png}` (column width 3.35 in);
    counts (TP / FP / FN in the crop), window and per-image AP50 in `reports/figures/dota/fig_dota_crops.csv`.
  - Script `scripts/dota/d_qualitative_crops.py`. Candidates go into the next exchange archive.

## 2026-11-05 22:29 – Qualitative DOTA crops rendered (rule above, unchanged)

- `scripts/dota/d_qualitative_crops.py` (own scope, nice 19, CPU only; FCOS not touched) → `reports/figures/dota/
  fig_dota_crop_<rank>_<image>.{pdf,png}` + `fig_dota_crops.csv`. Vehicle centres in the 384-px window: P1478 184,
  P0683 42, P0454 32, P0199 19, P0749 8 → candidates 1 P1478 (most vehicles), 2 P0683 (quartile 4), 3 P0454
  (quartile 2), 4 P0199 (quartile 1); P0749 (quartile 3) not rendered.
- In the crop (TP / FP / FN; per-image AP50 box fusion → SCAHI): P1478 188/1/0 vs 188/0/0 (73.27 → 73.25);
  P0683 45/7/2 vs 46/6/1 (68.68 → 76.99); P0454 31/0/4 vs 32/0/3 (98.49 → 98.55); P0199 23/5/0 vs 23/7/0
  (97.16 → 96.46). Description reported to the user (differences small at this operating point; P0199 SCAHI worse,
  P0683 SCAHI one large FP box). Authors decide; nothing selected by me. In the next exchange archive
  (`make_exchange_zip_paper.py` takes `reports/figures/dota/*`; the A5 template for the final archive must add them).

## 2026-11-06 00:47 – Qualitative DOTA crops: none in the main paper (authors)

- User: none of the four crops goes into the main paper (differences at the operating point are small; two candidates
  look worse for SCAHI). No re-selection with other criteria. Crops, script and rule stay in the repo and in DECISIONS
  as documentation (`scripts/dota/d_qualitative_crops.py`, `reports/figures/dota/fig_dota_crop_*`, entries of
  2026-11-05). The supplement keeps the existing quartile figure (`reports/figures/dota/fig_dota_qualitative.*`)
  unchanged.

## 2026-11-06 08:28 – FCOS freeze approved by the authors; DOTA-val runtime layout

- User: "freigeben". Approval note (verbatim, in `configs/dota/frozen_rows_fcos.json`, field `approval`):
  "FCOS freeze approved by the authors after review of FREEZE_PROPOSAL_DOTA_FCOS.md (18542c2): pre-check passed (13/13 exact calib replay, limit rule met, both sizes, dense-output identity); refined-grid selection equals the F2 reference at 1024 and 2048 (reported as a finding); reporting rules as in addendum 3 incl. the secondary comparison against the inference crossing X_2048 (2560 px)."
- Set by me on the user's instruction (`scripts/dota/fcos_val.py approve NOTE`), rows unchanged, committed before any
  DOTA-val detection of FCOS (none existed: `runs/dota/val/dets/*_fcos_*` empty).
- Runtime (user: all cores free; parallelisation only, no change to rows, settings or limits): every cache row
  image-parallel on all 24 cores within the 105G cap (CPU-row scope 88G with 22 image workers, box fusion < 3 GB per
  image; GPU scope 17G; no swap, oom_score_adj 1000), longest rows first (NMS-o2m_1024, NMS-o2m_2048, then the SCAHI
  rows RG / F2 / B-RG / B-F2 at 2048, then 1024), GPU rows (A, X) in parallel on the GPU. Retry with 11 / 6 / 2 / 1
  workers after a failure, finished images kept. Hard stop 18:30 unchanged; evaluation only when every row exists.
  `scripts/dota/fcos_val.py` ORDER and `scripts/dota/fcos_val_auto.sh` changed accordingly; the waiter
  `scahi_after_fcos` was stopped before the approval commit and the chain started directly (tmux `scahi_fcos_val`).

## 2026-11-06 08:29 – Correction: calib_selection_limit of the FCOS freeze

- My `fcos_val.py approve` (copied from the YOLO approvals) also set the metadata field `calib_selection_limit` to
  20 000 (the YOLO selection limit). For FCOS the selection on calib used 200 000 (proposal, field as generated).
  Restored to 200 000 before the chain started; no code reads this field (only the proposal scripts write it), so no row,
  setting, limit or result was affected. `approve` no longer touches it. Approval, note and rows unchanged.

## 2026-11-06 09:05 – FCOS DOTA-val: new row priority + cross-row filling

- User: order change approved – NMS-o2m_1024 (running, untouched) → the 8 SCAHI rows → NMS-o2m_2048; plus cross-row
  filling: idle workers take (image × row) tasks of the next rows in priority order within the 105G cap and the memory
  model; parallelisation only; finished images kept; if all rows finish before 18:30, evaluation of all FCOS hypotheses
  on host-a, otherwise the finished rows stay and the rest goes to host-b.
- Status before the change (08:55): GPU rows A_1024, A_2048, X_2048 done 08:37 (rc 0; images at the limit 24 / 76 /
  82); NMS-o2m_1024 running since 08:29 with 22 image workers, 0 images finished (densest first), ≈ 58 GB.
- Implementation: `scripts/dota/fcos_val_fill.py` (tmux `scahi_fcos_fill`, own scope 97G, no swap, oom_score_adj 1000,
  RuntimeMaxSec to 18:30): tasks = one image of one row through `fcos_limit_check._fuse_image` (the per-image path of
  `fcos_val.py cpu`, same part files runs/dota/val/parts/<id>/limited/), one fresh process each; slots = 23 cores −
  images NMS-o2m_1024 still has in flight; admission: actual memory of the NMS-o2m_1024 scope + estimates of the running
  tasks + the new one ≤ 97 GiB and MemAvailable ≥ 20 GiB after it; estimates box fusion 3 GiB per image, SCAHI tasks
  by the conservative model of the tuning (`pipeline._rg_est`, profiles rg_mem_<B|C>.json); failed task retried once
  alone; a row with all images is concatenated in image order and written once (`final._write_dets`); evaluation
  (`fcos_val.py eval`) only when all 13 rows exist before the deadline.
- The old chain (`fcos_val_auto.sh`) keeps running only for NMS-o2m_1024: its controller could not be paused (tmux
  continues a stopped pane process at once), so `fcos_val.py cpu` now skips rows listed in
  runs/dota/val/parts/FILL_OWNER (written by the filler; the 9 remaining rows). After NMS-o2m_1024 the old loop therefore
  passes the other rows without work and ends without evaluation; the filler evaluates.

## 2026-11-06 09:38 – Cross-row filler: two fixes, restarted

- My bug: the filler did not create the part directory of a row, so its first task (RG_fcos_2048 P1854) failed twice
  (no output, nothing written). Second issue: sorting the images by cache size took ~3.5 min per row
  (`pipeline._cache_bytes` resolves the weights / cache directory per image), so no task started for ~28 min. Fixed:
  part directory created per task; cache sizes = the same sum with the directories resolved once per row type (checked
  equal to `_cache_bytes`). Filler scope stopped (`systemctl --user stop`) and restarted 09:37; NMS-o2m_1024 untouched
  (3 images done at 09:37). No row, setting or limit changed.

## 2026-11-06 16:46 – FCOS DOTA-val: NMS-o2m_fcos_2048 not started on host-a; stop 18:30 sharp; HANDOFF row list

- User 2026-11-06 ~16:45: NMS-o2m_fcos_2048 is no longer started on host-a (its densest images, ≈ 5 h each by
  NMS-o2m_fcos_1024 P1854, would not finish by 18:30; the row is computed on host-b). The cross-row filler
  (`scripts/dota/fcos_val_fill.py`, scope `scahi-fcosval fill`) is stopped by me (`systemctl --user stop`) right after
  B-F2_fcos_1024 is written; parts are written atomically (.tmp + replace), so any image it finished is complete and kept.
  `fcos_val_auto.sh` keeps skipping NMS-o2m_fcos_2048 (listed in runs/dota/val/parts/FILL_OWNER).
- User 2026-11-06 ~16:45: stop at 18:30 sharp, even if NMS-o2m_fcos_1024 is a few images short (it has 457/458 images,
  the last one in flight in its own scope with RuntimeMaxSec to 18:30); the copy and its verification before the
  departure have priority.
- HANDOFF 12.1 lists every FCOS row as complete / partial (with finished images) / not started; host-b evaluates the FCOS
  hypotheses only after all 13 rows are complete (no partial verdicts).
- Forecast correction: my 14:05 / 15:51 forecasts extrapolated linearly in cache bytes; the cost per image is strongly
  superlinear (P1854, 365 MB, ≈ 5 h; >300 small images in ≈ 10 min), so the rows finished far earlier than forecast.

## 2026-11-06 16:49 – Handover after the stop: scripts/migration/after_stop.sh (waiting in tmux scahi_handover)

- Unattended sequence (user 2026-11-05/06, HANDOFF 12.0 point 4): 18:30 sharp stop of every experiment (scopes
  `scahi-fcosval*`, tmux `scahi_fcos_val` / `scahi_slot_wd`; finished rows / parts kept) → FCOS DOTA-val row status into
  HANDOFF 12.1 + DECISIONS, committed → git bundle `--all` (verified) → final manifest → **one** `rsync -a
  --files-from` run (manifest entries + the manifest itself, absolute paths mirrored under
  <external drive>/Datasets1/python_lab/scahi_2026_11_06/) → sync, unmount, mount with `-t ntfs` → verification read from
  the disk (`scripts/migration/verify_disk.py`: irreplaceable files size + SHA-256; cache files present + size, their
  full check on host-b; every link with its target; manifest on the disk = local) → log onto the disk, unmount,
  power-off, "SAFE TO UNPLUG". Any failure: stop, disk stays mounted, "NOT SAFE". Heavy steps in own scopes (cap, no
  swap, oom_score_adj 1000). Runs in tmux (not as a systemd service) so udisksctl has the same session as my mounts.
- Fix in `transfer_host_a_host_b.py`: the manifest no longer lists itself (it held a stale hash of itself, which `verify`
  on host-b would have reported); it is copied alongside and compared on the disk. Manifest refreshed 16:4x (294.1 GB,
  65 889 files hashed), so only the last two hours are hashed at the stop.
- Dry run (16:49, no disk, no stop, no commit; subset 355 entries / 8 GB into the scratch dir): bundle verified, rsync
  rc 0, verification found exactly the 2 files changed after the manifest (HANDOFF, DECISIONS) and passed without them
  (0 deviations, link targets kept).

## 2026-11-06 18:00 – Handover copy started now instead of 18:30 (user)

- User 2026-11-06 ~17:58: start the copy now, since no experiment is running any more (NMS-o2m_fcos_1024 complete
  16:56; the chain ended 16:57; NMS-o2m_fcos_2048 not started on host-a, user 16:45). `scripts/migration/after_stop.sh`
  takes `STOP_AT` (default 2026-11-06 18:30); started with `STOP_AT=now` in tmux `scahi_handover`, target
  <external drive>/Datasets1/python_lab/scahi_2026_11_06. Sequence otherwise unchanged.

## 2026-11-06 18:00 – Stop of all experiments (18:30, user); FCOS DOTA-val status

- Stopped by `scripts/migration/after_stop.sh`; finished rows / image parts kept. Complete 12/13: A_fcos_1024, NMS-o2m_fcos_1024, F2_fcos_1024, RG_fcos_1024, B-F2_fcos_1024, B-RG_fcos_1024, A_fcos_2048, X_fcos_2048, F2_fcos_2048, RG_fcos_2048, B-F2_fcos_2048, B-RG_fcos_2048; partial: none; not started: NMS-o2m_fcos_2048. No evaluation on host-a; host-b evaluates the FCOS hypotheses only once all 13 rows are complete (no partial verdicts). Next: bundle, manifest, one rsync -a run to the external disk, verification from the disk.

## 2026-11-07 15:30 – host-b: working repository completed from the backup (user instruction 1.0)

- Situation (user): `~/scahi_2026_10_30` = complete host-b repository as of the 31.10. handover
  (BACKUP, never modified); `~/scahi` = the delta copied from host-a (no `.git`), working repository.
- Space before copying (15:06): 2.8 TB free on `/`; backup 75 GB (`du -sh`, nothing excluded); delta 281 GB. Plain copy,
  no hard links needed.
- 15:17–15:20: `rsync -a --ignore-existing scahi_2026_10_30/ scahi/` (nice 19, ionice idle), rc 0: 548 873 files and
  1 827 directories added (incl. `.git`), no symlink added, no delta file overwritten; rsync set mtime/permissions of 77
  already existing directories from the backup (directories only). Delta path list taken before the copy (458 983 paths).
- Git: bundle `exports/scahi_20261106_host_a_host_b_994d163054d6.bundle` verified (complete history; master 994d163,
  tag paper-draft-v1 35e4e72 = local tag), fetched to `refs/remotes/host-a/master`; 2da67ad is an ancestor of 994d163
  (fast-forward). A `git pull` would refuse (the working tree already holds the host-a state), so the working tree was
  compared with 994d163 in a temporary index first: exactly one difference, `env_export/host-b/configs_local_host_b.yaml`
  missing (committed on host-a 01.11. in 7912698 as a copy of `configs/local/host-b.yaml`, same SHA-256; old mtime, so
  outside the delta rule "mtime after 31.10. 21:00"); no untracked files. Then `git reset --mixed 994d163` (index/HEAD
  only, working tree untouched) and that file restored from 994d163. HEAD = 994d163, `git status` clean.
  `core.fileMode` was already false.
- Permissions (NTFS transfer, all delta entries 777; the backup already had 118 107 files with 777 from the October
  transfer): in the working tree without `.git`, files 777 → 664, except `*.sh`, files with a shebang and the two
  git-100755 scripts (`scripts/chain_queue.sh`, `scripts/start_queue.sh`) → 775 (33 files); directories 777 → 775
  (1 699). Symlinks, contents and mtimes unchanged.
- FCOS COCO checkpoint `fcos_resnet50_fpn_coco-99b0c9b7.pth` was not yet in `~/.cache/torch/hub/checkpoints/`; copied
  (`cp -p`) from the handover disk copy (`<external drive>/Datasets/python_lab/scahi_2026_11_06~/.cache/...`);
  SHA-256 99b0c9b7… and size = manifest entry.
- The backup may be deleted only after the full manifest verification (`scripts/migration/transfer_host_a_host_b.py
  verify`) has passed on this repository.

## 2026-11-07 15:40 – host-b: manifest verification passed (HANDOFF 12.3 step 3; user section 1.4)

- `scripts/migration/transfer_host_a_host_b.py verify` (15:29–15:30, cores 16–23, nice 19, ionice idle; log
  `runs/migration/verify_host_b_20261107.log`): 457 396 entries (files incl. caches with size + SHA-256, 132 777 links
  with target), 1 deviation: `docs/DECISIONS.md` = my entry of 15:30 (its state at 994d163 matches the manifest exactly,
  size 274 575, SHA-256 84d55efe…). → PASSED. Spot check: 3 random files > 500 MB re-hashed separately, all equal.
- Cross-check of the part filled from the backup: `scripts/migration/manifest.py --verify` (31.10. host-b → host-a
  manifest; log `runs/migration/verify_host_b_20261107_oldmanifest.log`): 2 mismatches, `configs/release/forbidden_terms.txt`
  and `configs/release/manifest.yaml`, both changed on host-a and in the host-a → host-b manifest (verified there).
- User 2026-11-07: takeover steps accepted; HANDOFF 12.3 step 5 (FCOS equality test of stored RG tasks) is obsolete, no
  tuning is resumed. Next: replay gate on a few frozen rows incl. one FCOS row on calib, then NMS-o2m_fcos_2048 and the
  FCOS evaluation. The backup `scahi_2026_10_30` stays until the user decides.

## 2026-11-07 15:36 – host-b: core rule for the FCOS DOTA-val row (user)

- User 2026-11-07: `NMS-o2m_fcos_2048` runs on **cores 2–23 with 22 image workers**; the earlier host-b rule "CPU jobs
  ≤ 8 cores (16–23)" dates from parallel training and no longer applies, since nothing else runs on host-b. **Cores 0–1
  stay free for light parallel work** (release / clean-room preparation, supplementary zip). Memory: own scope,
  MemorySwapMax=0, oom_score_adj 1000, ≥ 20 GB RAM left free (box fusion ≈ 2.6 GB per image). host-b: 125.3 GiB RAM
  (MemTotal 131 337 068 kB) → scope cap **MemoryMax=100G** (≈ 25 GiB outside the scope; 22 × 2.6 GB ≈ 57 GB inside).
  Launcher `scripts/dota/fcos_val_host_b.sh 2-23 22 100G` (as `fcos_val_auto.sh` on host-a without the deadline: releases
  the row from `runs/dota/val/parts/FILL_OWNER`, retries with fewer workers, keeps finished images; evaluation only once
  all 13 rows exist). Started only if the replay gate passes.
- `configs/local/host-a.yaml` (from the delta) moved unchanged to `env_export/host-a/configs_local_host_a.yaml`
  (SHA-256 = manifest entry), since `scahi.machine` allows one file in `configs/local/` (mirror of host-a, 2026-11-01).
- Not before the FCOS evaluation: the border-weighted box-fusion analysis (user).

## 2026-11-07 15:46 – host-b: replay gate passed (HANDOFF 12.3 step 4)

- `scripts/migration/host_b_replay.py` (the rows of `host_a_replay.py` + FCOS; writes only to
  `runs/migration/host_b_replay/`; tmux `scahi_replay`, cores 16–23, 15:33–15:45). |Δ| = 0 in every row:
  VisDrone val `E_yolov10m_full2048_o2o` (GPU), `XG_yolov10s_2048_o2m_C` (CPU); DOTA calib at limit 200 000
  `A_yolov10s_2048`, `RG_yolov10s_2048`, `A_yolo11s_2048`, `RG_yolo11s_2048` (against `runs/dota/freeze/replay_*_limit200000*.json`);
  FCOS calib against `configs/dota/frozen_rows_fcos.json`: `A_fcos_2048` (GPU, row_dets) 60.29881720791235, 23 images at
  the limit, and `F2_fcos_1024` (CPU, by image = the DOTA-val path of `fcos_val.py cpu`, 8 workers) 54.55044618770758,
  2 images at the limit – both as stored. host-b reproduces FCOS exactly. First start (15:33) failed at once: two machine
  configs in `configs/local/` (fixed, see 15:36). → `NMS-o2m_fcos_2048` started.

## 2026-11-07 15:50 – Release / clean-room update with FCOS prepared (cores 0–1, while NMS-o2m_fcos_2048 runs)

- Started 15:46 (`scripts/dota/fcos_val_host_b.sh 2-23 22 100G`, tmux `scahi_fcos_val_host_b`); at 15:49 22 workers, scope
  56 GB of 100G.
- New `scripts/dota/d_results_fcos.py` → `reports/D_RESULTS_FCOS.md` (hypotheses with the unadjusted p and the count
  of five, secondaries, secondary contrasts incl. RG_2048 − X_2048 with the crossing's calib values, all rows, size bins,
  training records; every number from `results_fcos.json`, the freeze and the training records). `d_rows_table.py`,
  `d_h3_table.py`, `d_perclass_table.py` add FCOS **only once `runs/dota/val/results_fcos.json` exists**; until then their
  outputs are byte-identical (checked). Dry run against a synthetic results file (YOLO11s values renamed, scratch only,
  never in the repo): all four scripts run.
- Release manifest: + `configs/dota/frozen_rows_fcos.json`, `scripts/dota/fcos_val.py`, `scripts/dota/d_results_fcos.py`.
  Test build into `runs/release/prep_fcos/scahi`: 366 files, forbidden hits 0, missing imports 0 → passed (official
  build record unchanged).
- After the evaluation (in this order): `results.txt` + `runs/dota/val/results_fcos.json` and
  `runs/dota/train/fcos_{1024,2048}_s0/training.json`; `regenerate_paper.py` + command `scripts/dota/d_results_fcos.py`
  and output `reports/D_RESULTS_FCOS.md`; regenerate the DOTA tables (five detectors); `make_release.py` →
  `release_cleanroom.py` (fresh env from the lock file) → `make_supplementary_zip.py`; RELEASE.md. The border-weighted
  box-fusion analysis only after the FCOS evaluation (user).

## 2026-11-07 15:54 – Safeguard before the FCOS evaluation: freeze SHA-256 check (user)

- User 2026-11-07: `fcos_val.py eval` (and the chain that calls it) must verify the SHA-256 of
  `configs/dota/frozen_rows_fcos.json` against the approved, committed version and abort on any mismatch. The file in use
  is the 60d7850 version (SHA-256 7349a658…), not 32c81fc (a4602383…): the two differ only in the metadata field
  `calib_selection_limit` (correction 2026-11-06 08:29, one minute after the approval and before the chain started; all
  12 finished rows used 60d7850). User's choice (2026-11-07): **exact SHA-256 = 60d7850, plus a check that the content
  differs from the approved 32c81fc only in `calib_selection_limit` and that 32c81fc is approved.**
- Implemented as `verify_freeze()` in `scripts/dota/fcos_val.py`, called first in `eval` (the chain
  `scripts/dota/fcos_val_host_b.sh` calls `fcos_val.py eval`, so it is covered; the running launcher script itself was
  not edited, since bash reads it while running). Pinned SHA-256 in the code, re-checked against `git show 60d7850:…`;
  any failure → SystemExit, no evaluation. `fcos_val.py` replaced atomically (`os.replace`), so a retry of the running
  row never reads a partial file. Tests (cores 0–1, read-only on the real file; failure cases on scratch copies via a
  redirected path): real file passes; changed row conf, `approved: false`, an added trailing newline → abort.
- Rule (user 2026-11-07): **no test may move or modify files that a running job uses** (after my 15:5x test that briefly
  moved `frozen_rows_fcos.json` aside; restored unchanged, SHA-256 still 7349a658…).

## 2026-11-07 15:55 – Check before the FCOS evaluation: 60d7850 changes documentary metadata only (user request)

- 60d7850 (2026-11-06 08:29:05, 19 s after the approval 32c81fc) changes exactly: `configs/dota/frozen_rows_fcos.json`
  `calib_selection_limit` 20000 → 200000 (one line); `scripts/dota/fcos_val.py` `approve()` no longer sets that field;
  DECISIONS entry. Cause: `approve()` (copied from the YOLO approvals, `scahi/dota/final.py` `approve` / `approve_new`
  set it to `P.PIPE` = 20 000, the YOLO calib selection limit) overwrote the proposal's value 200000 (written by
  `d_freeze_proposal_fcos.py`, = the FCOS calib selection limit of addendum 3 item 6; proposal 18542c2 has 200000).
  60d7850 restored the proposal value.
- Reads of the field (`git grep` over code, configs, scripts, tests at HEAD): none in any path that produces or evaluates
  DOTA-val detections. Only writers (`final.approve`, `final.approve_new`, the three freeze-proposal scripts) and the new
  safeguard `fcos_val.verify_freeze` (compares it, uses nothing). Every use of the FCOS freeze dict reads only `approved`,
  `approved_at`, `approval`, `pipeline_limit`, `rows`, `code_commit` (`final.frozen`, `val_gpu`, `val_cpu`,
  `val_eval_fcos`, `fcos_val.cpu`, `fcos_val_fill`, `fcos_limit_check`); no spread or key iteration over the dict; the
  field is not copied into `results_fcos.json`. Rows, configs, thresholds, `pipeline_limit` 200000 and the approval note
  are identical in 32c81fc and 60d7850. → documentary only; no approval needed.

## 2026-11-07 16:04 – fig_method (Fig. 1): main flow on one horizontal axis; box-fusion label (user)

- Main path on one axis y = 0 (centre of the input image): zoom stack symmetric around it (z0 yshift 8.905 mm = half of
  thumbnail height 1.181 cm + gap 6 mm); the two zoom levels merge orthogonally (like the split after the image) into one
  horizontal arrow into the detector (before: two slanted arrows); dense per-position block shifted so its anchor sits on
  the axis (before 0.05 above the detector centre → slightly slanted arrow); border-taper plot centred on the axis
  (before 0.15 above → slanted arrow into it). Scale space, fusion and detections were already on that height. No slanted
  connector on the main path.
- Label: "box fusion across passes (NMS/IOS, GREEDYNMM, WBF)" → "(NMS, greedy merging, WBF)".
- Remaining differences to the paper terms, reported, not changed: "z_0: whole image at train size" (the paper writes
  relative zooms z1/3, z1, z2; the whole-image pass at train size is z1 in "z1 + z2 tiles"); training note "SAHI chips"
  (paper: T0 "ignore regions dropped", T1 "ignore regions painted grey" on the same chips).
- Rebuilt with `scripts/make_figures.py method` (cores 0–1, nice 19; no file of the running FCOS row touched).

## 2026-11-07 16:08 – fig_method (Fig. 1): no z value for the whole-image pass; training-variants box removed (user)

- User 2026-11-07: in the paper z is the **absolute** zoom (input pixels per image pixel); the whole-image pass at train
  size has no fixed z (e.g. ≈ 0.94 for a 1360 px VisDrone image at 1280 px). "$z_0$: whole image at train size" →
  "whole image at train size"; heading "Zoom stack $\{z_k\}$" → "Zoom stack"; "$z{=}2$: tiles" kept.
- "Training variants (main study)" box removed (training variants are not part of SCAHI; described in the study and the
  supplement). The standalone page crops to the content; main flow still on one horizontal axis.
- Rebuilt with `scripts/make_figures.py method` (cores 0–1, nice 19; no file of the running FCOS row touched).

## 2026-11-07 16:13 – Displayed model names YOLOv10 / YOLOv11 / YOLOv12 in all generated paper figures and tables (paper decision)

- User 2026-11-07: displayed names "YOLOv11s", "YOLOv12s", "YOLOv11m", "YOLOv12m" (and "YOLOv11", "YOLOv12" in legends)
  in all paper figures and tables generated by the repo; Ultralytics' names YOLO11 / YOLO12 stay unchanged in code,
  configs, ids and file names. No recomputation; CPU on cores 0–1 only; no file of the running FCOS row touched (the row
  imports `scahi.*`, `fcos_val.py`, `d_freeze_proposal_fcos.py`, `d_freeze_proposal.py`, `fcos_limit_check.py`: none edited).
- Changed: string literals that reach outputs (not comments, not docstrings; token-based, atomic writes) in
  `paper_figures.py`, `tiles_payoff.py`, `ablation_report.py`, `scahi_algorithm_doc.py`, `latency_breakdown.py`,
  `latency_host_a_dota.py`, `make_exchange_zip_paper.py`, `ext_e4_report.py`, `ext_testdev_analysis.py`, and in
  `scripts/dota/` `d_figures.py`, `d_rows_table.py`, `d_h3_table.py`, `d_perclass_table.py`, `d_results_new.py`,
  `d_a5_tables.py`, `d_results_fcos.py`; "YOLO11/12" → "YOLOv11/v12". Names stored in measurement / analysis data
  (latency JSONs, per-image tiles CSVs, the ε rule text) are mapped at report time (`display()` in `latency_breakdown.py`,
  `tiles_payoff.py`, `ext_e4_report.py`); the data files are unchanged.
- Regenerated: `scripts/regenerate_paper.py --reference git` (17/17 commands ok; 29 of 75 outputs changed, and after
  mapping the new names back every changed output equals the committed version except the intended "YOLOv11/v12");
  `ext_e4_report.py` and `ext_testdev_analysis.py --doc-only` (names only, plus the generation time in E4_EXT.md). PDFs
  whose PNG did not change (creation date only) were restored to the committed version. Figures with names: DOTA
  hypotheses / rows / calib curves, extension calib curves / forest / recall / waterfall / X-grid cost / scaling,
  ablation VisDrone.
- Not changed (not generated paper figures / tables): the stored input data (`reports/tiles_payoff/per_image_*.csv`,
  run JSONs), the `extra.eps.rule` metadata field in `fig_ext_forest.data.json` (not displayed), freeze proposals,
  earlier exchange documents (`FOR_CLAUDE_*`, `EXCHANGE_*`) and design notes. The release (`expected/`) follows with the
  FCOS release update; the final exchange archive takes the regenerated files.

## 2026-11-07 16:28 – Chain to the end (user) and framing of the border-weighted box-fusion analysis (before any number)

- User 2026-11-07, one chain without waiting in between: (1) NMS-o2m_fcos_2048 (running), then the FCOS evaluation with the
  freeze check; (2) report the FCOS verdicts at once, no stop; (3) release with the FCOS tables (regenerate_paper.py incl.
  FCOS, expected/), clean-room test, rebuilt supplementary code zip with the forbidden-term scan; (4) final exchange
  archive (≤ 25 MB, FOR_CLAUDE.md; FCOS results and tables, all figures and tables with YOLOv11 / YOLOv12, fig_method
  b97c036, RELEASE.md), commit only FOR_CLAUDE.md, report; (5) the exploratory border-weighted box-fusion analysis, then
  STOP and report. Any failure: stop there and report. Rules unchanged.
- **Framing of (5), recorded before any number exists** (spec: `docs/ANALYSIS_WEIGHTED_BOX_FUSION.md`): post hoc, like
  the component ablation; never toward any verdict; the authors decide afterwards whether and how it is reported; nothing
  from it in exchange archives or paper tables without the user's decision. Purpose: whether SCAHI's gain over box fusion
  comes from the coverage-aware weighting or from merging the dense outputs before boxes are decided. Variants: (a)
  weighted NMS (per-pass scores × SCAHI's weight at the box centre, same function as SCAHI, then the frozen IoS-NMS merge),
  (b) weighted WBF (main-study WBF implementation and grid). Only the existing box-fusion parameters (and the WBF grid)
  tuned on calib as for the frozen rows. VisDrone val (YOLOv10s, YOLOv11s) and DOTA-val (YOLOv10s, YOLOv11s), 2048, same
  passes and head as the frozen box-fusion rows; paired bootstrap CIs against box fusion and SCAHI (pre-specified, refined
  grid) on the same passes.

## 2026-11-07 16:33 – Border-weighted box-fusion analysis: head pairing (user; amends fd750d8, before any number)

- Only same-head comparisons. VisDrone YOLOv10s on both heads: o2o = X1 (frozen box fusion) | weighted (a) / (b) | SCAHI
  refined grid `XG_yolov10s_2048_o2o_C`; o2m = plain box fusion on o2m tuned on calib exactly like X1 (post hoc, new) |
  weighted (a) / (b) | SCAHI pre-specified X2 and refined grid `XG_yolov10s_2048_o2m_C`. YOLOv11s (single head) and DOTA
  (all rows o2m) unchanged. No cross-head CI in the report. Amendment appended to `docs/ANALYSIS_WEIGHTED_BOX_FUSION.md`.

## 2026-11-07 20:26 – FCOS evaluation: OOM kill, restart with fewer workers (user); post-evaluation limit check

- NMS-o2m_fcos_2048 finished 20:19 (rc 0; 458 images, 16 321 s, 110 254 959 detections, 159 images at the limit; scope
  peak 52.2 GB). Freeze check passed (60d7850, 7349a658…, differs from 32c81fc only in `calib_selection_limit`).
- `fcos_val.py eval` with SCAHI_VAL_EVAL_WORKERS=11 was killed by the OOM killer of its scope (MemoryMax 100G) after
  ~2 min (journal: `oom-kill`, rc 143); no results file written, detection caches only read.
- User 2026-11-07: restart only `fcos_val.py eval` with SCAHI_VAL_EVAL_WORKERS=3, fallback 1 after another failure; same
  scope (MemoryMax 100G, MemorySwapMax=0, oom_score_adj 1000, cores 2-23); parallelism only
  (`scripts/dota/fcos_eval_retry_host_b.sh`).
- After the evaluation and before step 3: the pre-specified post-evaluation limit check on DOTA-val (as for P1384,
  `fcos_limit_check.py val`): per FCOS row the images reaching 200 000, and exploratory mAP50 without the limit for every
  affected row. Verdicts stay as computed under the pre-registered limit; differences reported next to them. If the
  recomputation without limit is too heavy for a row, report the cost first.

## 2026-11-07 21:03 – FCOS evaluation follows the frozen, executed path; box-fusion limit per pass only (user, before any FCOS DOTA-val number)

- Third and fourth eval attempt (SCAHI_VAL_EVAL_WORKERS 3 and 1) were OOM-killed as well (20:31, 20:46): the largest row
  NMS-o2m_fcos_1024 (184.6 M detections) does not fit into 100G even with one worker. No FCOS eval file written.
- Finding: the two FCOS box-fusion rows hold up to 3 495 205 (1024) / 3 407 292 (2048) detections per image (mean 403 143 /
  240 731). `scahi.fusion.run.f0_image` applies the pipeline limit only per pass (`pass_dets`, ≤ 200 000 per pass); the
  merged IoS-NMS output is not capped. Same code for every YOLO box-fusion row (max 43 642 per image there, so it never
  mattered); the FCOS calib selection and the exact replay (13/13, |Δ| = 0) approved in 32c81fc ran the same way (up to
  2.99 M per calib image). Rows A, X (torchvision per image) and SCAHI (top_k) are capped at 200 000 per image.
- **Decision (user):** evaluate as frozen. The frozen object is the executed path used for the calib selection and the
  approved exact replay: for box fusion the 200 000 limit applies per pass, the merged output is not capped – for YOLO and
  FCOS alike. The wording "per image and per pass" in addendum 3 item 6 / the 2026-11-01 decision does not match the
  implementation for box fusion; the deviation from the wording is reported in the report and the paper's supplement.
- Docstring of `scripts/dota/fcos_limit_check.py` corrected (there is no final max_det for box fusion).
- Next (user): memory-efficient evaluation path with an equality proof before use (old vs. new path on rows that fit, exact
  equality of all metrics, per-image values and bootstrap vectors; any difference → stop), then `fcos_val.py eval`; then
  the limit check on DOTA-val: capped rows (A, X, SCAHI) images at the limit and exploratory mAP50 without limit; the two
  box-fusion rows the number of passes at the per-pass limit, the recomputation without the per-pass limit only after
  reporting its cost.

## 2026-11-08 09:24 – Label of the B rows (user)

- "B" rows are SCAHI on the single pass without tiles (decomposition step B), not "tiles only" (my mislabel in a chat
  report). Label in all FCOS reports and tables: "SCAHI, single pass (no tiles)" (B-RG: "SCAHI, single pass (no tiles),
  refined grid"). `scripts/dota/d_rows_table.py` (shared with the YOLO rows: wording only, same meaning) and
  `scripts/dota/d_results_fcos.py` (new description column; X label with the chosen inference size; limit wording per the
  2026-11-07 decision: box fusion per pass only, merged output not capped).

## 2026-11-08 12:37 – Step 5 split: VisDrone part of the border-weighted box fusion now on cores 0-1 (user)

- User 2026-11-08: start the VisDrone part (YOLOv10s on both heads, YOLOv11s; spec `docs/ANALYSIS_WEIGHTED_BOX_FUSION.md`,
  head pairing as decided) now on cores 0-1 in its own small scope; DOTA part unchanged, after the FCOS evaluation.
  Framing unchanged (fd750d8 + head-pairing amendment); nothing into archives or paper tables without the user's decision.
- Run: `scripts/weighted_box_fusion.py visdrone --variants a,b --workers 2`, SCAHI_WBF_CORES=0-1, scope MemoryMax 8G,
  MemorySwapMax=0, oom_score_adj 1000 (tmux `scahi_wbf_visdrone`, log runs/analysis_wbf/visdrone.log). It imports no
  file under scahi/dota/ (the FCOS evaluation path that is being changed) and writes only to runs/analysis_wbf/.
- Guard (`runs/analysis_wbf/mem_guard.sh`): SIGSTOP of its process tree while MemAvailable < 22 GB, resume above 30 GB.

## 2026-11-08 12:49 – Memory-lean evaluation and vectorised bootstrap proven; FCOS evaluation of the remaining ten rows

- **Lean build** (`scahi/dota/evaluate.py`, `LEAN`; old path: `lean=False` / SCAHI_EVAL_LEAN=0): equality proof
  (`scripts/dota/eval_lean_equality.py`, runs/dota/eval_lean_proof/equality.json): old vs. new path, every point metric,
  2 359 per-image arrays (TP / FP / npos per class × IoU threshold × size bin, crowded / isolated recall counts) and every
  bootstrap vector **exactly equal** on NMS-o2m_yolov10s_2048, RG_yolov10m_2048 (both also equal to their stored
  eval/ files), F2_fcos_1024, B-RG_fcos_2048, A_fcos_2048. Peak RSS old → new: 1.8 → 1.6, 2.7 → 1.9, 20.3 → 6.4,
  9.9 → 3.6, 53.2 → 16.1 GiB. Cause of the OOM: one int64 copy of the image index per (threshold, size bin) key.
- **Adopted (user):** the new-path outputs of F2_fcos_1024, B-RG_fcos_2048, A_fcos_2048 copied to runs/dota/val/eval/
  (same call `evaluate_row` on the same parquet).
- **Vectorised bootstrap** (user 2026-11-08; `scahi/dota/boot_vec.py`, wired in `final.evaluate_row`, `BOOT_VEC`; old
  path: SCAHI_BOOT_VEC=0): integer resample weights → cumulative TP / FP sums exact in float64 in any order; only the TP
  positions are evaluated (max precision at recall ≥ t is attained there); same rec / prec / 11-point / 101-point / mean
  operations. Same RNG stream (resample_weights, seed 0). Equality proof (`scripts/dota/boot_vec_equality.py`,
  runs/dota/boot_vec_proof/): all five matching-based bootstrap vectors exactly equal on NMS-o2m_yolov10s_2048,
  RG_yolov10s_2048 (vs. stored eval/) and B-RG_fcos_2048 (vs. the lean-proof vectors); CIs and p-values of D-H2b YOLOv10s
  (both directions) identical and equal to results.json; wired `evaluate_row` reproduces the stored json and boot.npz
  of NMS-o2m_yolov10s_2048 and B-RG_fcos_2048 exactly (102 s; 292 s instead of 9 015 s).
- Next: `fcos_val.py eval` (freeze check first) for the remaining ten rows with both paths, scope as before
  (`scripts/dota/fcos_eval_retry_host_b.sh`: MemoryMax 100G, no swap, oom_score_adj 1000, cores 2-23; 3 workers,
  fallback 1).

## 2026-11-08 14:10 – Exploratory FCOS diagnostics (user; framing before any number of them)

- User 2026-11-08: exploratory, post hoc; no verdict changes, no training, no new GPU passes (existing detections and
  caches only); CPU in an own memory scope after the running DOTA-val limit check; reported to the user only; nothing
  into archives or paper tables without the user's decision. Script `scripts/dota/fcos_diagnostics.py`, outputs under
  runs/dota/val/diag_fcos/.
- **1. Equal cap for both fusions.** D-H2a / D-H2b for FCOS under the pre-specified secondary "cap 4000" (contrast
  already in results_fcos.json, computed with the evaluation) and, as an exploratory addition, with both box-fusion rows
  (NMS-o2m_fcos_1024 / _2048) capped to the top 200 000 detections per image by score (`evaluate.apply_cap`: stable,
  ties keep the stored order; the same number as the SCAHI rows' top_k); evaluated with `final.evaluate_row`; paired
  bootstrap CIs and one-sided p against the stored RG vectors (`final._contrast`, same resamples). Plus max / mean
  detections per image of every FCOS row (from the val parquet files).
- **2. Prediction redundancy** (FCOS, YOLOv10s, YOLOv11s at 2048 px, DOTA-val, o2m caches of the frozen 2048 SCAHI
  rows, stack C = whole image + tiles). Prediction = cached anchor with max class probability >= tau_splat, as in
  `scalespace.prediction_table`; tau_splat of the row's frozen F2 setting and, where different, of its refined-grid
  setting. Per non-difficult GT object: number of predictions with IoU >= 0.5 to its (devkit, horizontal) box,
  class-agnostic (as the splat uses max_c p) and, as a second column, with argmax class = GT class; per pass (whole-image
  pass; each tile pass: mean over the tile passes whose view contains the object's centre) and summed over all passes.
  Mean, median, quantiles and share with zero; by size bin (`evaluate.SIZE_EDGES`) and crowded vs. isolated
  (`final.crowd_gt`). Descriptive only, no test.
- **3. Crowded vs. isolated recall** (R50, as in the evaluation) of NMS-o2m and RG at 1024 and 2048 from the stored
  FCOS eval files, with paired bootstrap CIs of RG − NMS-o2m from the stored vectors.
- **4. FCOS (2048) added to the DOTA part of the border-weighted box-fusion analysis** (step 5, after the release steps;
  spec docs/ANALYSIS_WEIGHTED_BOX_FUSION.md unchanged): passes, head (o2m), per-pass limit 200 000 and merge exactly as
  the frozen NMS-o2m_fcos_2048 row; NMS grid and selection rule of that row on calib; same-model comparisons against
  NMS-o2m_fcos_2048, F2_fcos_2048 and RG_fcos_2048 only. Variant (b) only if the wbf_probe on FCOS is not
  disproportionate (reported first).

## 2026-11-08 14:25 – Updated priorities (user): submission deadlines for the main paper and the supplement

- Everything below exploratory / post hoc unless stated; no verdict changes; nothing into archives or paper tables
  without the user's decision; each result reported as soon as it exists.
- **A (main paper, results to the user by Nov 13):** A.1 DOTA part of the border-weighted box fusion (YOLOv10s, YOLOv11s,
  FCOS at 2048 px; (a) first, (b) only after its cost is reported); deliverable in the form of one possible extra row of
  the main-paper ablation table: per dataset and detector AP of plain box fusion, weighted NMS, SCAHI pre-specified and
  refined grid (same passes and head), paired bootstrap 95 % CIs of weighted NMS vs. plain box fusion and vs. SCAHI
  refined grid, crowded / isolated recall of the same rows, one summary line per dataset (range of weighted NMS − plain
  box fusion and of SCAHI − weighted NMS); `reports/ANALYSIS_WEIGHTED_BOX_FUSION.md` with VisDrone and DOTA. A.2 the
  running DOTA-val limit check (differences next to the verdicts).
- **B (supplement / rebuttal, by Nov 20):** B.3 FCOS diagnostics (framing 14:10), B.4 release incl. FCOS and analysis
  tables, clean-room test, supplementary code zip with the forbidden-term scan, B.5 final exchange archive
  (<= 25 MB, FOR_CLAUDE.md; only FOR_CLAUDE.md committed).
- **C (at the supplement submission):** a git tag on the exact state used; recorded here; no cleanup
  of caches and results needed to regenerate every number.
- Order A.2 -> A.1 -> B.3 -> B.4 -> B.5. Light parts may use idle cores 0-1 while a long job runs on 2-23 (as for the
  VisDrone part): the YOLO models of A.1 (small caches) and the wbf_probe start on cores 0-1 during A.2 (own scope,
  MemoryMax 16G, no swap, oom_score_adj 1000; writes only runs/analysis_wbf/; imports nothing the limit check changes).
  `weighted_box_fusion.py dota --models` runs a subset; results of earlier models are kept.

## 2026-11-08 14:24 – Variant (b) on DOTA (user)

- Time plan accepted. (b) weighted WBF on DOTA: YOLOv10s / YOLOv11s only if its measured cost (wbf_probe) is small,
  < 2 h in total; FCOS: skipped (no probe). (a) answers the core question. If (b) is skipped, the report states why: cost,
  and (b) was below plain box fusion on VisDrone.

## 2026-11-08 17:33 – fig_method (Fig. 1): example in the detector block (user)

- Detector block: "dense detector (e.g. YOLOv10s)" – generic block, example in parentheses (as before 648e375, now with
  parentheses); the single "Dense per-position outputs" block stays without head-specific labels; nothing else changed.
  PDF + PNG regenerated (`scripts/make_figures.py method`, cores 0-1). PNG vs. b97c036: same size and layout; outside
  the detector block only isolated anti-aliasing pixels of glyph edges differ (709 px). Goes into the next exchange archive.

## 2026-11-08 21:16 – Table 2 (VisDrone compute control, YOLOv10s): new row X1 and one new latency run (user)

- Table 2 gets the row "2048 + tiles, box fusion" = frozen main-study row X1 (2048 px single pass + z=2 tiles, NMS with
  intersection over the smaller box, IOS 0.8, conf 0.01, o2o). Frozen values: val AP 38.85 / AP50 64.73, test-dev
  AP 33.25 / AP50 56.65; GFLOPs per image (unified method, `runs/gflops_unified.json`) 561.7 (same passes as X2).
- **Table 2 latencies are replaced by one new exclusive run on host-b** (latencies in one table come from one run): single
  pass 2048 (R2), single pass 2560 (R3), 2560 weights at 3072 px (crossing row), 2048 + tiles box fusion (X1), 2048 + tiles
  SCAHI (X2), 2048 + tiles SCAHI refined grid (XG_yolov10s_2048_o2m_C). Per row: passes, merge step (box fusion on the
  CPU; SCAHI on the CPU and on the GPU), total. Same procedure and exclusivity rules as all earlier latency runs (default
  power limit; no foreign compute process on the GPU; foreign CPU load < 1 core; smoke first; every row "final",
  otherwise the whole table is measured again; nvidia-smi process lists saved at start and end). Script
  `scripts/latency_table2.py`, output `runs/latency_table2/`, table `reports/tab_compute_control.{md,csv}`.
- Timing: after A.1 FCOS and B.3, before B.4/B.5; overlaps with no other job. If a VS Code GPU process prevents
  exclusivity, stop and report. The table goes into the final exchange archive. Earlier Table 2 latencies (E2,
  latency_breakdown) stay stored, but Table 2 uses the new run only.

## 2026-11-09 09:13 – Table 2: SCAHI rows in the optimised path, whole table re-measured (user)

- Option 2: the whole Table 2 is measured again in ONE new exclusive run; SCAHI rows (X2, XG) in the optimised path of the
  E2 run (`pipeline_gpu.measure`: unfused model - fusing removes the one-to-many branch -, one-to-many head only,
  decoding and SCAHI on the GPU); single-pass rows and X1 unchanged. Per row passes, merge (CPU/GPU where applicable),
  total; all rows "final"; nvidia-smi lists at start and end (`scripts/latency_table2.py measure_opt`).
- Footnote: code path per row; SCAHI needs the dense one-to-many outputs instead of final boxes (a real cost of the
  method). In the optimised path they stay on the GPU; only final detections go to the host. So the footnote states
  this cost as "dense outputs + unfused model", not as a GPU->host copy (reported to the user).
- The reference-path values of the run 2026-11-09 07:57 (`runs/latency_table2/latency.json`) stay in the report only,
  not in the paper table.
- Parity of the optimised path vs. reference path on every val image (`runs/latency_table2/parity_opt.json`): same
  counts, scores and classes; box differences up to 1.9 px for 4 of 217,492 (X2) and 2 of 249,999 (XG) detections.

## 2026-11-09 09:56 – B.4 release with FCOS and analysis tables

- Regeneration (`scripts/regenerate_paper.py`) now also runs `scripts/dota/d_results_fcos.py`, `scripts/weighted_box_fusion.py
  report` (exploratory, post hoc; its report was approved for the archives as exploratory) and `scripts/latency_table2.py
  report` (Table 2); 81/81 outputs byte-identical against git, and in the clean room against `expected/`.
- Not shipped in the release (exploratory, no decision by the user): FCOS diagnostics B.3 and the DOTA-val limit check.
- Release 383 files (source 057043501806), 0 forbidden terms, 0 missing imports; clean room passed (69 passed / 7 skipped);
  `exports/supplementary/code.zip` 2.14 MB, 384 files, term scan passed, SHA-256 a06509e7…d3dda. Details: reports/RELEASE.md.

## 2026-11-09 10:11 – B.3 diagnostics and DOTA-val limit check in the supplement (user)

- Both go into the final archive and the supplement, marked exploratory, post hoc. To make every supplement number
  regenerable they are in the release regeneration and byte comparison: `scripts/dota/fcos_diagnostics.py report` (from
  the stored JSON files; no data) -> `reports/dota/FCOS_DIAGNOSTICS.md`, `tab_fcos_diag_*.csv`. Supersedes the
  exclusion in the B.4 entry above. Clean room rerun: 21/21 commands, 87/87 outputs byte-identical; code zip rebuilt
  (395 files, checks passed, SHA-256 in reports/RELEASE.md).

## 2026-11-09 10:59 – Supplement code usage; public GitHub version built (user; nothing uploaded or published)

- Supplement step 1: `release/README.md` starts with a quickstart (install from the lock file; regeneration without data
  or GPU; SCAHI on an own image; pipeline per study). New `examples/scahi_yolov10s.py` (public COCO YOLOv10s weights by
  model name, whole-image pass 2048 + z = 2 tiles, SCAHI pre-specified or refined grid, CPU or GPU fusion, optional box
  fusion on the same passes). Tested in the clean-room environment on two VisDrone val images and one DOTA val image
  (`runs/release/example_test_cleanroom.log`). Release 395 files; clean room 87/87 byte-identical; code zip 396 files,
  term scan passed. Step 2 (supplementary.zip with the PDF) waits for the authors' PDF.
- GitHub version (`scripts/make_github_release.py` -> `exports/github/scahi`, sources `release_github/`): same code as the
  release, machine names and the decision log not hidden; README with authors / affiliation as in the arXiv PDF of
  2026-11-09 10:43, BibTeX with an arXiv-ID placeholder; CITATION.cff (no ORCIDs: not in the paper - placeholder
  comment); docs: the three preregistrations and this log, sanitised (personal paths, user name, host names, dataset
  paths, the restricted dataset copy). Inverted scan, secrets, internal URLs, binaries: passed; clean room from
  `exports/github/scahi`: 87/87 byte-identical.
- Weights for a later GitHub release asset: `exports/github/RELEASE_ASSETS_WEIGHTS.md` (33 paper models). **Finding:**
  Ultralytics' `best.pt` equals the frozen `best_calib.pt` model (EMA, fp16) for only 5 of 31 YOLO models; the
  decision of 2026-11-01 ("only best.pt") would not reproduce the paper. Recommendation reported to the user: stripped
  `best_calib.pt` (identical weights, no optimizer). Open: authors' decision; ORCIDs; whether the venue-specific mentions in the public decision log stay.

## 2026-11-09 11:20 – Supplement package, GitHub version finalised, weight assets (user decisions 0-5; nothing uploaded)

- paper/ (given by the authors: arXiv PDF and source, review PDF and source, supplementary.pdf) is not modified, moved or
  committed; committing anything from paper/ needs the authors' decision (review PDFs never in the public version; the
  arXiv source only after the arXiv upload).
- Supplementary package: `exports/supplementary/supplementary.zip` = README.md (one page, `release/supplement/README.md`) +
  supplementary.pdf (byte-identical to paper/supplementary.pdf) + code/ (`scripts/make_supplementary_package.py`); scan of
  member names, every text member and the PDF (Info, XMP, object dictionaries, all decompressed streams, extracted text):
  0 hits; only binary = the PDF. Byte scans of compressed PDF streams give random 3-letter matches and are not used.
- GitHub version: author order Becker, Bayer, Münch, Hübner (README, CITATION.cff, BibTeX key becker2026sliced); ORCIDs
  from the authors; public docs with venue-specific details neutralised and a note at the top of the public decision
  log (this internal log unchanged); figstyle comment without venue name; local git repository with one initial commit
  (author set locally to the paper name and institute address), no remote.
- Weights: stripped copies of the frozen models (`scripts/make_weight_assets.py` -> `exports/github/weights/`, 33 files,
  885 MB, list `exports/github/RELEASE_ASSETS_WEIGHTS.md`): tensor-identical 33/33, frozen SHA-256 = freeze record 33/33,
  metadata free of local paths; replays with the stripped weights identical for VisDrone main study (R2 val AP), extension
  (E_yolov10m_full2048_o2o val AP), DOTA original / addendum 2 / addendum 3 (A_*_2048 calib mAP50).
- Licensing check reported to the user (no change): SCAHI core (scalespace, scalespace_torch) free of ultralytics and
  (A)GPL code; third-party ports listed (DOTA_devkit MIT header without LICENSE file, VisDrone toolkit without licence,
  re-implementations of pycocotools / torchvision / Ultralytics behaviour).

## 2026-11-09 11:22 – Licensing (authors' decision)

- The SCAHI core (`scahi/fusion/scalespace.py`, `scahi/fusion/scalespace_torch.py`) is licensed under **Apache-2.0**
  (SPDX header, "Copyright (c) 2026 Fraunhofer IOSB", citation request with the arXiv entry); everything else
  **AGPL-3.0-or-later** (SPDX header in every own source file). Root LICENSE (AGPL-3.0), LICENSES/Apache-2.0.txt, NOTICE
  for the Apache part, THIRD_PARTY_NOTICES.md. Applied only in the public GitHub version (`exports/github/scahi`); the
  supplementary code zip stays as submitted.

## 2026-11-09 12:12 – Public repository on host-b filled (user; not pushed)

- `~/public_repos/scahi` (clone of the public GitHub repository, branch main): contents of
  `exports/github/scahi` copied (406 tracked files incl. .gitignore; tree identical) and committed as one commit on top
  of the existing README commit (author set locally to the paper name and institute address); 33 stripped weights copied
  to `weights/` (ignored by git, SHA-256 equal to the asset list). Nothing pushed.

## 2026-11-09 12:54 – DOTA latencies re-measured on host-b; the host-b run replaces the host-a tables (user)

- All DOTA latencies are measured again on host-b in ONE exclusive run (`scripts/latency_dota_host_b.py`), so that every
  latency of the paper comes from one machine; the host-b run **replaces** the host-a tables
  (`reports/dota/tab_dota_latency_host_a*.md` stay stored for comparison only). Rows: single pass, tile passes with box
  fusion (CPU) and SCAHI refined grid (CPU and GPU), SAHI (YOLOv10s/m), D-H5 rows (SAHI passes + SCAHI, pre-specified and
  refined, CPU and GPU) - same code and protocol as the host-a runs; the DOTA SCAHI rows additionally in the optimised
  path of Table 2, with a parity check against the reference path on every DOTA-val image before timing. D-H5 keeps the
  host-a protocol (its passes use SAHI's slice grid and Ultralytics' box mapping; no parity-tested GPU-resident path).

## 2026-11-09 14:56 – Parity of the optimised GPU path with one-to-one matching (user): no genuine deviation

- `scripts/parity_opt_matched.py`: per image and class, one-to-one matching on the box distance (mutual nearest
  neighbours, Hungarian on the rest), scores compared after matching; reference = SCAHI on the CPU from the stored caches.
  VisDrone Table 2 rows (X2, XG o2m C; 548 images): all detections matched, 0 count mismatches, 0 pairs beyond 1e-4
  relative, max |Δbox| 2e-12 px, max |Δscore| 2.2e-16. DOTA RG rows (YOLOv10s/m, YOLOv11s 2048, YOLOv12s 1024; 458 images):
  same, max |Δbox| 4.9e-4 px (= 2^-11, one float32 step at 4096-8192 px), max |Δscore| 3.3e-16.
- The earlier deviations (up to 70 px) were an artefact of the sorted comparison (pairs swapped at equal scores). The
  optimised-path GPU latencies are therefore valid SCAHI timings; the table texts now cite the matched parity.

## 2026-11-09 15:53 – All SCAHI latencies in the optimised GPU path, all from host-b (authors)

- The paper uses the optimised GPU path for all SCAHI latencies (Table 2 and DOTA), all measured on host-b
  (`reports/tab_compute_control.md`, `reports/dota/tab_dota_latency_host_b{,_dh5}.md`). The host-b DOTA and D-H5 tables
  and the matched parity enter the release regeneration; the Table 2 footnote cites the one-to-one matched parity.
- Release, clean-room test, supplementary.zip and the GitHub version are rebuilt only after the authors place the new
  supplementary.pdf in paper/.

## 2026-11-09 16:07 – Rebuild with the new supplementary.pdf (authors' file of 2026-11-09 16:01)

- paper/supplementary.pdf (26 pages, SHA-256 761046eea9a18a9971749eb19627c54a0abf7991676b2058251b48702de87aad) used unchanged. Release 404 files, 0 forbidden terms, 0 missing
  imports; clean room 69 passed / 7 skipped, regeneration 22/22 commands, 91/91 outputs byte-identical; code zip 405 files,
  scan passed; `exports/supplementary/supplementary.zip` 14.66 MB, 407 files, full scan incl. the PDF passed, SHA-256 1d5e6e94a196027b750f7c03811243b520c89cb6212bf31d482c08bfa4a0bd94.
- GitHub version 415 files, scans passed, clean room 91/91; local commit e1f0d32 in exports/github/scahi (no push).

## 2026-11-09 16:14 – fig_method (Fig. 1) notation as in Eqs. (1)-(4) (authors)

- Border-taper block: "weight $\omega=b_t$: border taper of tile $t$ only" (no hat; the optional "(no validity window,
  no uncertainty term)" dropped). Scale-space block: "trilinear splat of $\omega\cdot\max_c p$, $\omega=b_t$". Nothing
  else changed; PDF + PNG regenerated (`scripts/make_figures.py method`, cores 0-1), same size (2114x495).
- Not changed (reported to the authors): the axis label "border $\hat b_t(x)$" and the fusion formula
  "$S=\sum w\,p/(C+\epsilon)$" still use the old notation.
- 16:15 follow-up (authors): taper axis label "border $b_t(x)$" (no hat); fusion block
  "$F=\frac{1}{C}\sum \omega\,\max_c p\,\kappa$" (field F as in Eq. (1), no epsilon; S stays the class score of Eq. (3)).
  Full form fits, no overlap; nothing else changed.
- 16:19 follow-up (authors): scale-space heading "Scale space $F(x,y,\log\sigma)$" (F = field of Eq. (1); S = class
  score of Eq. (3)); nothing else changed (changed pixels only in the heading).

## 2026-11-09 16:24 – supplementary.zip with the replaced supplementary.pdf (authors' file of 2026-11-09 16:21)

- paper/supplementary.pdf (26 pages, SHA-256 8d77cc7e308a40e54e454e4e1b72a44d238d1f71b6fc3fbd400f5a74dcc21fae) used
  unchanged; code/ byte-identical to the clean-room-tested release of c30d0fa (RELEASE_FILES.sha256 unchanged).
  `exports/supplementary/supplementary.zip` 14.66 MB, 407 files, full scan incl. the PDF passed, SHA-256
  6d36dddcda5ce56e7afe6166c3e55d7c7f01a94d4e325556bb01474719047911.

## 2026-11-09 16:31 – Machine names are internal: hidden in every public or anonymous output (user)

- "host-b" / "host-a" (and host names) never appear in the public repository, its file names or commit messages; this
  replaces "machine names may stay" for the GitHub version. Release and GitHub build now also replace machine names
  inside identifiers (file, column, function names; `(?i)host-a` -> host_a, `(?i)host-b` -> host_b) and the term checks
  look for both names without word boundaries. Rebuilt: release (404 files, clean room 91/91), code.zip, supplementary.zip
  (SHA-256 403a8421fdeaaf02a7bf2572f5d6c7a614bdc06d29887df28df3b163726e05a3), GitHub version (clean room 91/91).
- Public repo: the cleanup is a new local commit on top of the pushed state (no history rewrite). Earlier pushed commits
  still contain the names; removing them needs a history rewrite by the authors.
