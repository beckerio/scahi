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

- **2026-11-13 – Ergänzung: YOLO26m als sechster Detektor der Erweiterung (Nutzerentscheid; eingetragen bevor irgendein
  YOLO26m-Ergebnis auf VisDrone-val oder test-dev existiert).** Bis zu diesem Eintrag existieren für YOLO26m nur zwei
  abgebrochene Tutorial-Trainings mit anderem Rezept (1280 px, Epoche 7 bzw. 4; Ultralytics-Validierung auf calib bzw.
  keine), ohne jede Auswertung auf val oder test-dev; sie werden verworfen und gehen in nichts ein.
  1. **Modell:** YOLO26m, ultralytics 8.4.183 (gepinnt), COCO-Gewichte `yolo26m.pt` (Release v8.4.0, SHA-256
     401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7, in `runs/ext/heads.json` nachgetragen). Kopf wie
     YOLOv10: zwei Zweige, o2m und o2o; Standardausgabe = o2o (NMS-frei, Top-k). Box-Kodierung `reg_max = 1`: der Boxkopf
     gibt die vier Seitenabstände direkt aus (keine DFL-Verteilung). Die eigene Dekodierung erhält dafür einen direkten
     Pfad (nur für `reg_max = 1`; Standardabweichung 0, von F2 nicht genutzt); für alle anderen Modelle bleibt sie
     unverändert und bitidentisch (geprüft über Regeneration und Replay-Gate).
  2. **Einordnung:** YOLO26m wird nach den Regeln für Zwei-Zweig-Modelle behandelt, d. h. **genau wie YOLOv10m**:
     E-1280 / E-2048 und Modellauswahl über o2o (o2m-Kurve mitprotokolliert); E-SAHI auf den 1280-Gewichten (o2o);
     E-MZ-NMS-*s* in beiden Zweigen; E-MZ-DENSE-*s* mit o2m (Hauptzeile) und zusätzlich o2o; deskriptiv „Einzelpass o2m +
     NMS“ bei 1280 und 2048; die Refined-Grid-Zeilen (X-Grid B = Runde-1-Raster, C = 108-Varianten-Raster, Zeitgrenze
     ≤ 2× F2, Auswahl max. calib-AP) und die Ganzbild-Zeilen wie bei den fünf Modellen, mit derselben Einstufung wie dort.
  3. **Rezept unverändert** (Abschnitt 1 und Ergänzung 0f6fecbc): COCO-Init, 100 Epochen, Batch 8 (1280) bzw. 3 (2048),
     Seed 0, AMP, Ultralytics-Standardaugmentierung und -Loss, `optimizer=auto`, Trainingsliste `train_main`, gleicher
     Code (`scahi.ext.e_train` / `scahi.train.full.train_full`). Argumentvergleich mit YOLO11m 1280/2048 in
     `docs/DECISIONS.md` (2026-11-13): nur Modell und Ausgabepfade verschieden. Modellauswahl: Toolkit-Port-AP
     (`visdrone_ignore`) auf calib an denselben Epochen (alle 5 Epochen bis 80, dann jede der Epochen 81–100;
     36 Auswertungen), o2o; die Ultralytics-Fitness wird nicht verwendet. Zusätzlich wird der Checkpoint jeder
     ausgewerteten Epoche aufbewahrt (Kopie nach dem Speichern, keine Änderung am Training). OOM-Regel unverändert:
     einmalige Reduktion 8 → 6 bzw. 3 → 2, Batch ≥ 2.
  4. **Fragen und Urteile:** E-H1, E-H2 (je s) und E-H3 (= Q1–Q3) unverändert, ε = 0,30, gepaarter Bootstrap über
     val-Bilder (1000, Seed 0), **Urteil je Detektor** für YOLO26m; E-H4 deskriptiv wie bei YOLOv10m. Die Urteile der fünf
     Modelle, das Kriterium „≥ 4 von 5“ und der gepoolte Effekt bleiben unverändert; YOLO26m wird getrennt berichtet und
     zählt in keinem Familienkriterium. Holm-Anhang: für YOLO26m allein (Familie aus einem Modell, also unadjustiert),
     getrennt ausgewiesen.
  5. **Gates:** G-E1 (eigene Dekodierung gegen Ultralytics, beide Zweige, Koordinaten ≤ 4 ulp mit TF32 aus und
     |ΔAP| ≤ 0,01 auf calib), G-E3 je Größe; G-E2 gilt nur für s-Modelle und entfällt. Fällt G-E1, entstehen keine
     dichten Zeilen.
  6. **Nicht prüfbare Fälle:** Passt bei 2048 auch Batch 2 nicht, scheitert ein Lauf endgültig oder fällt ein Gate, fehlen
     die betroffenen Zeilen; jede Hypothese, die eine fehlende Zeile braucht, ist für YOLO26m „nicht prüfbar“ und zählt
     als **„nicht erfüllt“** (Regel der Ergänzung 2026-10-27, Punkt 4). Die übrigen Hypothesen werden normal entschieden.
  7. **Ablauf:** Caches, calib-Tuning (Einzelpass-conf, NMS-IOS-Raster, F2-24er-Raster, Refined Grid mit Zeitgrenze,
     SAHI-Raster v2) wie in der Erweiterung; Freeze mit Gewichts-Checksummen und Herkunft jeder Wahl; Replay-Gate; danach
     genau **eine** Auswertung auf val und genau **eine** auf test-dev, Berichtsregeln wie im Freeze der Erweiterung
     (val-Urteil maßgeblich, test-dev nicht bestätigend daneben). Alles in `docs/DECISIONS.md`.

- **2026-11-14 – Ergänzung: YOLO26s als siebter Detektor der Erweiterung (Nutzerentscheid; eingetragen bevor irgendein
  YOLO26s-Training oder -Ergebnis existiert).** Bis zu diesem Eintrag existiert für YOLO26s nur die heruntergeladene
  COCO-Gewichtsdatei; kein Training, keine Auswertung.
  1. **Modell:** YOLO26s, ultralytics 8.4.183 (gepinnt), COCO-Gewichte `yolo26s.pt` (Release v8.4.0, SHA-256
     646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b, in `runs/ext/heads.json` nachgetragen). Kopf wie
     YOLO26m: zwei Zweige (o2m / o2o), `reg_max = 1`; derselbe direkte Dekodierpfad wie für YOLO26m (Ergänzung 7c01b5d,
     Punkt 1), für alle anderen Modelle unverändert.
  2. **Einordnung:** genau wie YOLO26m (Ergänzung 7c01b5d, Punkt 2), d. h. nach den Regeln für Zwei-Zweig-Modelle wie
     YOLOv10m: dieselben 25 Zeilen (E-1280 / E-2048 beide Zweige, E-SAHI auf den 1280-Gewichten o2o, E-MZ-NMS-*s* beide
     Zweige, E-MZ-DENSE-*s* o2m und o2o, Refined Grid B / C, Ganzbild-Zeilen D) mit derselben Einstufung.
  3. **Rezept unverändert** (Abschnitt 1, Ergänzungen 0f6fecbc und 7c01b5d Punkt 3): COCO-Init, 100 Epochen, Batch 8 (1280)
     bzw. 3 (2048), Seed 0, AMP, Ultralytics-Standardaugmentierung und -Loss, `optimizer=auto`, `train_main`, gleicher Code.
     Vor dem Start wird ein Diff aller Trainingsargumente gegen YOLO26m 1280 / 2048 in `docs/DECISIONS.md` eingetragen
     (erwartet: nur Modell und Ausgabepfade verschieden). Modellauswahl wie YOLO26m: Toolkit-Port-AP auf calib, o2o, an
     denselben 36 Epochen; Checkpoints jeder ausgewerteten Epoche werden aufbewahrt. OOM-Regel unverändert (einmalige
     Reduktion 8 → 6 bzw. 3 → 2, Batch ≥ 2; Wiederaufnahme aus `last.pt` nach einem Abbruch wie bei YOLOv12m, als
     Abweichung mit Epoche protokolliert).
  4. **Rückrechnung der Boxen:** Die Cache-Zeilen verwenden von Beginn an die Ultralytics-Konventionen (Größenregel,
     Letterbox-Faktor, Rückrechnung je Achse; `scahi.mapcheck.ultralytics_prep`, Cache-Wurzel `cache_ualigned/`), wie für
     YOLO26m nach der Entscheidung vom 2026-11-14; calib-, val- und test-dev-Caches werden nur so gebaut.
  5. **Fragen und Urteile:** E-H1, E-H2 (je s) und E-H3 unverändert, ε = 0,30, gepaarter Bootstrap über val-Bilder (1000,
     Seed 0), **Urteil je Detektor**; E-H4 deskriptiv. YOLO26s wird wie YOLO26m getrennt berichtet und zählt in keinem
     Familienkriterium („≥ 4 von 5“, gepoolter Effekt unverändert); Holm-Anhang für YOLO26s allein (unadjustiert).
  6. **Gates:** G-E1 (eigene Dekodierung gegen Ultralytics, beide Zweige, |ΔAP| ≤ 0,01 auf calib), G-E3 je Größe, und –
     da YOLO26s ein s-Modell ist – **G-E2**: E-1280 (o2o) innerhalb ±5 AP von YOLOv10s E-1280 (B-FULL 1280, val 33,66);
     geprüft nach der einzigen val-Auswertung, sonst Prüfung und Bericht, ändert kein Urteil. Einzelpass-Gate auf calib
     wie bei YOLO26m (bitgleich in zwei Wiederholungen und |Δ| zum Tuning-Wert ≤ 0,05 AP) und vor test-dev das exakte
     val-Replay aller Zeilen.
  7. **Nicht prüfbare Fälle:** wie Ergänzung 7c01b5d Punkt 6 (fehlende Zeile → betroffene Hypothese „nicht prüfbar“ =
     „nicht erfüllt“).
  8. **Deskriptiver Vergleich bei gleichem Rechenaufwand** (wie im Paper YOLOv10s 2048 gegen YOLOv10m 1280): Einzelpass
     YOLO26s bei 2048 px gegen Einzelpass YOLO26m bei 1280 px, beide o2o (die Hypothesenzeilen E-2048 bzw. E-1280):
     GFLOPs je Bild (exakt je Eingabeform, `scahi.flops`, über die val- bzw. test-dev-Bilder) und AP auf val und test-dev,
     Δ mit gepaartem Bootstrap-KI (1000, Seed 0). Kein Urteil, keine Schwelle.
  9. **Ablauf:** wie Ergänzung 7c01b5d Punkt 7; der Freeze-Vorschlag geht vor jeder Auswertung auf val oder test-dev an
     die Autoren; erst nach ihrer Freigabe genau eine Auswertung auf val (mit exaktem Replay) und genau eine auf test-dev.
