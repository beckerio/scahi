# SCAHI – Präregistrierung der Erweiterungsstudie E (weitere Detektoren, VisDrone)

Geschrieben am 2026-10-26 vor dem ersten Training der Erweiterung (Auftrag `docs/EXTENSION_BRIEF.md`, Abschnitt 4).
Nach dem Commit wird diese Datei **nicht mehr geändert, nur ergänzt** (Abschnitt „Ergänzungen“ am Ende, mit Datum). Der
Commit-Hash dieser Fassung wird in allen `EXT_`-Reports genannt. Die abgeschlossene Studie (Tag `paper-draft-v1`,
Präregistrierung d067729c) bleibt unverändert; sie wird hier weder neu ausgewertet noch ergänzt.

## 1. Modelle, Daten, Rezept

- **Bestätigend gezählt (5 neue Modelle):** YOLOv10m, YOLO11s, YOLO11m, YOLO12s, YOLO12m (ultralytics 8.4.183,
  COCO-Gewichte Release v8.4.0, SHA-256 in `runs/ext/heads.json`). **Anker (nur berichtet):** YOLOv10s mit den
  bestehenden Zeilen der abgeschlossenen Studie.
- **Zweige:** YOLOv10m hat o2o (NMS-frei) und o2m (+ NMS); YOLO11/12 haben einen dichten Kopf (+ NMS), im Code als „o2m“
  geführt (strukturell identisch zum o2m-Kopf von YOLOv10; `runs/ext/heads.json`).
- **Daten:** VisDrone2019-DET, Setting visdrone10, Splits calib / val / test-dev wie bisher, Protokoll `visdrone_ignore`
  (Headline = Toolkit-Port-AP@[.5:.95]), `plain` sekundär, Ausgabe pro Bild auf 500 gekappt, gleiche Auswertungssoftware.
- **Training (je Modell, s ∈ {1280, 2048}):** Rezept von B-FULL YOLOv10s unverändert: COCO-Init, 100 Epochen, imgsz s,
  Batch 8 (1280) bzw. 3 (2048), Seed 0, AMP, Ultralytics-Standardaugmentierung und -Loss, Trainingsliste `train_main`.
  Modellauswahl: Epoche mit der höchsten Toolkit-Port-AP auf calib (10 Auswertungen wie bisher); YOLOv10m über o2o,
  YOLO11/12 über die Standardausgabe (NMS IoU 0,7, conf 0,001, max_det 500). Bei OOM im festen Batch: einmalige Reduktion
  (8 → 6 bzw. 3 → 2), dokumentiert; die Zeile bleibt in der Bestätigungsmenge.
- **Zusatz-Seeds:** YOLO11s @ 2048, Seeds 1 und 2 (für ε, Abschnitt 3).

## 2. Zeilen (alle Parameter auf calib, berichtet auf val)

| ID | Definition | auf calib getunt |
|---|---|---|
| E-1280, E-2048 | Einzelpass bei der Trainingsgröße (YOLOv10m: o2o; sonst Standardausgabe) | conf ∈ {0,001; 0,01; 0,05; 0,1} |
| E-SAHI | SAHI 640 px mit Ganzbild-Pass auf den 1280-Gewichten, Raster v2 (Overlap × Merge × Metrik × Schwelle × conf) | beste Merge-Variante nach calib-AP |
| E-MZ-NMS-*s* | Pass bei *s* wie trainiert + z = 2-Kacheln (512 px, N = 1024), Cache-Overlap 0,3, Box-Fusion NMS mit Metrik IOS | Schwelle {0,3…0,8} × conf {0,01; 0,1} |
| E-MZ-DENSE-*s* | dieselben Pässe, dichte Fusion **F2** (fest, keine Variantenwahl), Fenster a = 8 (6–14), b = 192 (192–256) wie bisher | festes 24er-Raster (tau_splat, tau_peak, cell_factor, bins_per_oct) |

- **Zweige der MZ-Zeilen:** YOLO11/12: der eine Kopf. YOLOv10m: E-MZ-DENSE-*s* mit o2m (Hauptzeile) und zusätzlich o2o
  (für E-H4); E-MZ-NMS-*s* in **beiden** Zweigen; Hypothesenzeile siehe Abschnitt 3.
- Andere F-Varianten höchstens als Diagnostik für **ein** Modell; sie gehen in keine Hypothese ein.

## 3. Hypothesen (auf val definiert)

**ε** = max(0,30; 2·σ_seed) AP, σ_seed = Stichproben-SD (ddof 1) der val-AP von E-2048 YOLO11s über die Seeds 0, 1, 2.
Ist 2·σ_seed > 0,30, wird ε als datierte Ergänzung eingetragen, **bevor** die Hypothesenauswertung (Skript der Phase E4)
läuft. (Die val-Zeilen selbst dürfen vorher entstehen; ε folgt aus einer festen Formel ohne Wahlfreiheit.)

Je Modell m (Seed 0), Δ mit gepaartem Bootstrap über val-Bilder (1000 Resamples, Seed 0, Perzentil-KI 2,5/97,5 %):

- **E-H1 Auflösung schlägt Kachelung:** Δ = E-2048 − E-MZ-NMS-1280; erfüllt, wenn Δ > ε. (YOLOv10m: beide Zeilen o2o.)
- **E-H2 Dichte Fusion schlägt Box-Fusion:** Δ = E-MZ-DENSE-*s* − E-MZ-NMS-*s*, **getrennt für s = 1280 und s = 2048**;
  erfüllt, wenn Δ > ε. Gleiche Gewichte, gleiche Pässe, **gleicher Zweig** (YOLOv10m: o2m gegen o2m; die o2o-NMS-Zeile wird
  zusätzlich berichtet).
- **E-H3 Über den Einzelpass hinaus:** Δ = E-MZ-DENSE-2048 − E-2048; erfüllt, wenn Δ > ε **und** die untere KI-Grenze > 0.
  (YOLOv10m: DENSE o2m gegen Einzelpass o2o.)
- **E-H4 o2m schlägt o2o** (nur YOLOv10m): Δ = E-MZ-DENSE-2048(o2m) − E-MZ-DENSE-2048(o2o); berichtet werden Richtung und
  KI (wie H3 der abgeschlossenen Studie, keine Schwelle); s = 1280 zusätzlich deskriptiv.

**Verallgemeinerung:** Eine Hypothese gilt familienübergreifend, wenn sie bei **≥ 4 von 5** neuen Modellen erfüllt ist
(E-H2 je s getrennt). Zusätzlich der **gepoolte Effekt** = Mittelwert der Deltas über die 5 Modelle, KI per gepaartem
Bootstrap über Bilder mit denselben Resamples für alle Modelle (pro Resample Mittel der 5 Deltas).

**Fehlende Läufe** werden als fehlend berichtet; ein Modell ohne die nötigen Zeilen zählt für die Hypothese als
„unklar“ (nicht als erfüllt); das Kriterium „≥ 4 von 5“ wird mit dieser Zählung angewendet.

**Seeds:** Primär Seed 0. σ_seed nur für ε; die YOLO11s-Seeds 1, 2 werden sekundär berichtet und ändern kein Urteil.

## 4. Auswahl, Gates, test-dev

- Alle Hyperparameter auf calib (Raster oben); auf val wird nur berichtet; keine Auswahl anhand von val.
- Gates: G-E1 Parität je neuem Modell (eigene Dekodierung gegen Ultralytics, Toleranz max(1e-4 px, 1 float32-ulp) an allen
  verwendeten Eingabegrößen; Ein-Kopf-Modelle: eigene Dekodierung + NMS gegen den Predictor), G-E2 Plausibilität
  (E-1280 der s-Modelle innerhalb ±5 AP von YOLOv10s, sonst Prüfung und Bericht), G-E3 = G5b je Modell (Einzelpass,
  F2 gegen NMS auf calib, Abfall ≤ 1,5 AP).
- test-dev genau einmal nach Freeze (`docs/FREEZE_PROPOSAL_EXT.md`) und Freigabe; Berichtsregeln wie im letzten Freeze.

## Ergänzungen

- **2026-10-26 – G-E1-Toleranz (Nutzerentscheid, vor jedem trainierten Modell):** pro Koordinate max(1e-4 px, **2**
  float32-ulp des betragsgrößeren Werts) statt 1 ulp; Prüfung mit abgeschaltetem cuDNN-TF32. Grund (gemessen auf den
  COCO-Gewichten): Der Erwartungswert der DFL-Verteilung wird bei uns als softmax-gewichtete Summe, bei Ultralytics als
  conv1d berechnet; die unterschiedliche Summationsreihenfolge ergibt ≤ 5 ulp der Distanz und × Stride ≤ 2 ulp der
  Koordinate (≤ 1,22e-4 px). Der Box-Schritt ist bei gleichen Distanzen bitidentisch. Mit TF32 rechnet Ultralytics' DFL in
  reduzierter Genauigkeit. Keine Änderung an Hypothesen, Zeilen oder Auswahl.

- **2026-10-26 – Nutzerentscheide nach Diskussion der Designnotizen** (`reports/EXT_DESIGN_NOTES_20261026.md`; eingetragen
  bevor irgendein E-H3-Ergebnis existiert – bis dahin ist kein Modell der Erweiterung fertig trainiert):
  1. **Rechenkontrolle zu E-H3:** zusätzlich **R3 = B-FULL 2560** für YOLOv10s (gleiches Rezept: COCO-Init, 100 Epochen,
     Batch 3, Seed 0, Auswahl o2o auf calib); **B-FULL 3072** nur, wenn es im Smoke-Test ohne Rezeptänderung (Batch 3) in
     den Speicher passt. R3 wird neben E-H3 berichtet (AP, GFLOPs, Latenz), ändert aber kein E-H3-Urteil.
  2. **Budget:** YOLO12m @ 2048 entfällt (finanziert R3). Für YOLO12m sind E-H1, E-H2(2048) und E-H3 damit „nicht
     prüfbar“ (zählen als „unklar“ im Kriterium „≥ 4 von 5“). **Feste Kürzungsreihenfolge**, falls die Prognose 75 GPU-h
     übersteigt: YOLO12m 2048 (bereits gestrichen) → YOLO11s-Zusatz-Seeds (dann ε = 0,30, keine σ_seed-Prüfung) →
     YOLO11m 2048.
  3. **OOM-Batch-Reduktion:** Ultralytics akkumuliert über round(nbs/batch) Schritte (nbs = 64) und skaliert den
     Weight Decay mit batch·accumulate/nbs; effektiver Batch 8 → 64, 3 → 63, 6 → 66, 2 → 64. Je Lauf in
     `training.json` dokumentiert (batch, accumulate, effektiver Batch).
  4. **YOLOv10m:** deskriptive Zeilen „Einzelpass, o2m + NMS“ bei 1280 und 2048 (Trennung von Kopf- und Fusionseffekt in
     E-H1/E-H3); keine Hypothesenzeilen.
  5. **G-E1 (ersetzt die Ergänzung vom selben Tag):** bestanden = maximale Abweichung ≤ **4 ulp** (pro Koordinate,
     max(1e-4 px, 4 ulp des betragsgrößeren Werts), TF32 aus) **und** Toolkit-Port-AP der nachgelagerten Detektionen auf
     calib identisch auf 4 Dezimalen (eigene Dekodierung gegen Ultralytics' Dekodierung desselben Vorwärtslaufs, gleiche
     Nachbearbeitung: Ganzbild-Pass bei der Trainingsgröße, je Zweig). Die AP-Bedingung ist entscheidend. Der Vergleich mit
     dem Predictor (gefuste Gewichte) wird nur berichtet.
  6. **Latenz:** zwei getrennte Spalten „Referenzimplementierung“ und „optimierter Pfad“. In der optimierten
     Pareto-Ansicht nutzt **jede** Zeile (auch der Einzelpass) ihren optimierten Pfad: gefustes Modell, gleiche Präzision
     (float32). Die Genauigkeitswerte stammen weiter aus dem Referenzpfad.
  7. **Multiples Testen:** Hauptkriterium bleibt „≥ 4 von 5“. Zusätzlich im Anhang Holm-adjustierte Urteile je Modell:
     einseitiger Bootstrap-p-Wert p = (1 + #{b: Δ*_b ≤ ε}) / (B + 1) (B = 1000, für E-H3 zusätzlich die KI-Bedingung
     unverändert), Holm über die Modelle je Hypothese (E-H2 je s getrennt), α = 0,05. Der gepoolte Effekt wird nur
     **deskriptiv** berichtet (der Bild-Bootstrap erfasst keine Varianz zwischen Modellen).
  8. **Bekannte Einschränkung:** E-H3 auf VisDrone-val ist nur teilweise unabhängig von der Entdeckung (X2 wurde auf
     denselben Bildern gefunden). Die entscheidende Bestätigung von E-H3 wird für den zweiten Datensatz präregistriert.

- **2026-10-27 – Budget 75 → 77 GPU-h (Nutzerentscheid; eingetragen nach den Zwischenwerten von YOLOv10m und YOLO11s, vor
  jedem weiteren E-H-Ergebnis).** Anlass: Der Neustart von YOLO12s 2048 mit der präregistrierten Fallback-Batch 2
  (OOM bei Batch 3) dauert gemessen ≈ 15,9 h statt prognostiziert 6,6 h; die Prognose stieg damit auf ≈ 76 GPU-h. Die
  feste Kürzungsreihenfolge (Punkt 2 oben) hätte als Nächstes YOLO11m 2048 gestrichen. **Entscheid:** Budget auf
  77 GPU-h erhöht, YOLO11m 2048 bleibt im Plan; die Kürzungsreihenfolge gilt weiter für Überschreitungen von 77 GPU-h.
  Grund: YOLO12m 2048 ist bereits für R3 gestrichen, E-H1 und E-H3 sind für YOLO12m nicht prüfbar. Würde auch YOLO11m
  2048 gestrichen, wären diese Hypothesen nur bei 3 von 5 Modellen prüfbar und das Kriterium „≥ 4 von 5“ unerreichbar.
  **Nicht prüfbare Modelle:** bereits geregelt (Abschnitt 3 „Fehlende Läufe“ und Punkt 2 oben): ein Modell ohne die
  nötigen Zeilen zählt als „unklar“, nicht als erfüllt, und das Kriterium „≥ 4 von 5“ wird mit dieser Zählung
  angewendet. Diese Regel wird wie geschrieben angewendet; keine neue Regel. Hypothesen, Zeilen und Auswahl unverändert.

- **2026-10-27 (≈ 21:30) – Wiederherstellung von YOLO12m 2048, Budget nach Bedarf (Nutzerentscheid; ersetzt die
  Budget-Ergänzung vom selben Abend; eingetragen vor dem Start von YOLO12m 1280, also bevor irgendein YOLO12m-Ergebnis
  existiert).**
  1. **Budget:** wird nach Bedarf erhöht (einige GPU-h sind unerheblich); die feste Kürzungsreihenfolge aus Punkt 2 der
     Ergänzung 0f6fecbc entfällt damit. Aktuelle Prognose ≈ 105 GPU-h (Unsicherheit vor allem YOLO12m 2048).
  2. YOLO11m 2048 bleibt im Plan.
  3. **YOLO12m 2048 wird wiederhergestellt** (hebt die Streichung zugunsten von R3 auf). Damit gilt wieder das
     ursprünglich präregistrierte Design: alle fünf Modelle (YOLOv10m, YOLO11s/m, YOLO12s/m) werden auf VisDrone
     vollständig ausgewertet, „≥ 4 von 5“ über alle fünf wie geschrieben. **Transparenz:** Die Wiederherstellung wurde
     beschlossen, nachdem der Zwischenwert von E-H3 für YOLOv10m (und YOLO11s) bekannt war; sie hängt von keinem
     YOLO12m-Ergebnis ab und stellt nur den ursprünglichen Plan wieder her.
  4. **Speicherregel (jetzt fest):** YOLO12m 2048 läuft nur mit Batch ≥ 2 (effektiv 64 über Akkumulation, nbs 64);
     Batch 1 ist nicht zulässig (unzuverlässige BatchNorm-Statistik). Vorher ein kurzer Worst-Case-Speichertest
     (`e_smoke_yolo12m_2048`: dichteste Trainingsbilder, Batch-Wächter gegen Ultralytics' stille Halbierung; Batch 3 laut
     Rezept, sonst die präregistrierte Rückfall-Batch 2); das Ergebnis wird vor dem Trainingsstart berichtet. Passt
     Batch 2 nicht (im Test oder später im Lauf), entfällt der Lauf. **Auslegung, datiert vor jedem weiteren
     E-H-Ergebnis:** Ein dadurch nicht prüfbares Modell zählt im Kriterium „≥ 4 von 5“ als **„nicht erfüllt“**; das
     entspricht der präregistrierten Zählung („unklar“, nicht als erfüllt, Abschnitt 3 „Fehlende Läufe“).
  5. **Einschränkung (fürs Paper):** YOLO12s 2048 (und YOLO12m 2048, falls gelaufen) trainieren mit Batch 2 statt der
     Standard-Batch 3; die Akkumulation hält den effektiven Batch bei 64, die BatchNorm-Statistik beruht aber auf
     2 Bildern je Schritt.
  6. R3 (2560/3072) bleibt (in der Queue jetzt zuletzt).
  7. Was bis zur Deadline auf host-b (So 01.11. 12:00) nicht fertig wird, läuft nach dem Wechsel auf host-a weiter
     (gleiche Regeln; zuerst das Replay-Gate).
  8. Die Darstellung von YOLO12 im Paper (Haupttext oder Anhang) ist **nicht** festgelegt und wird nach E4 entschieden.
