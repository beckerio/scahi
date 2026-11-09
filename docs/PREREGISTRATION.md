# SCAHI – Präregistrierung (Phase 3–6, val)

Geschrieben am 2026-10-23 vor dem ersten Phase-3-Training (Auftrag `docs/AUTORUN_BRIEF.md`, Abschnitt 2). Nach dem Commit
wird diese Datei **nicht mehr geändert, nur ergänzt** (Abschnitt „Ergänzungen“ am Ende, mit Datum). Der Commit-Hash
dieser Fassung wird in allen späteren Reports genannt.

## 1. Referenz und Messbedingungen

- **Referenz R:** B-MZ-native, Zweig o2o (Hauptzeile): B-FULL-1280-Gewichte
  (`runs/p2/20261022_v10s_BFULL1280_train_s0/.../best_calib.pt`), Pass bei imgsz 1280 wie trainiert plus z = 2-Kacheln
  (T = 512 px, N = 1024), Box-Fusion NMS/IOS, conf/Schwelle/Overlap auf calib getunt (Job `bmz_native`,
  exp_id `20261023_v10s_BFULL1280_mznative…_nms-ios_s0`).
- **Messbedingungen für alle Vergleiche:** Split val (548 Bilder); Protokoll `visdrone_ignore`; Headline-AP =
  Toolkit-Port (AP@[.5:.95], Klassenmittel nach Bildzahl gewichtet); Ausgabe pro Bild nach Score sortiert und auf 500
  gekappt; dieselbe Auswertungssoftware (`scahi.eval`) für Referenz und Methode.
- **Rechenaufwand:** verarbeitete Eingabe-Megapixel pro Bild (Summe über alle Pässe; Netzeingang inklusive Padding),
  Mittel über die Bilder; GFLOPs pro Bild werden zusätzlich berichtet (gleiches Backbone YOLOv10s → proportional).
  Die Zulässigkeit („höchstens gleicher Aufwand“) wird **auf calib** bestimmt (Mittel über die calib-Bilder), berichtet
  wird zusätzlich der val-Wert. Fusions-/Nachbearbeitungszeit zählt nicht zum Rechenaufwand, sie erscheint in der Latenz.

## 2. Hauptkriterium

**Kandidaten** (alle mit Phase-3-Gewichten T4, Seed 0): Stapel ∈ {S4 {0,64, 2} N=1024@z2, S5 {1,0, 2} N=1024@z2,
S1 {⅓, 1, 2} N=640, S2 {⅓, 1, 2} N=1024@z2} × Fusion ∈ {F0-NMS/IOS, F0-GREEDYNMM, F0-WBF, F1, F2, F3, F3e, F4, F5} ×
Zweig ∈ {o2o, o2m}, jeweils mit den auf calib getunten Parametern (festes Raster, gleiches Budget je Fusionsvariante).
Cache-Overlap aller Stapel = der für R auf calib gewählte Overlap (gleiche Kachelgeometrie wie R auf der z = 2-Stufe).

**Auswahl (nur calib):**
- **M_ab** = Kandidat mit der höchsten calib-AP (Toolkit-Port) unter allen Kandidaten mit Rechenaufwand ≤ dem von R.
- **M_c** = Kandidat mit der höchsten calib-AP unter allen Kandidaten mit Rechenaufwand ≤ 0,7 · Aufwand(R).
  Gibt es keinen solchen Kandidaten, ist (c) „nicht prüfbar“ (nach Zoom-Planer bei Overlap 0,25: keiner der vier
  Stapel erreicht ≤ 70 %; das wird so berichtet, es wird kein Stapel nachträglich ergänzt).

**Ziele (auf val, mindestens eines muss erfüllt sein):**
- **(a)** ΔAP(M_ab − R) ≥ +1,0 (Toolkit-Port-AP) **und** die untere Grenze des 95-%-KI von ΔAP > 0.
- **(b)** ΔAP_σ<16(M_ab − R) ≥ +2,0, AP_σ<16 = pycocotools-AP@[.5:.95] für GT mit σ = √(w·h) < 16 px
  (Flächenbereich [0, 256)), auf den nach `visdrone_ignore` bereinigten Daten, maxDets 500, **und** die untere Grenze
  des 95-%-KI > 0.
- **(c)** Aufwand(M_c) ≤ 0,7 · Aufwand(R) **und** gleiche AP: das gesamte 95-%-KI von ΔAP(M_c − R) liegt in [−ε, +ε]
  (Äquivalenz; die Forderung „KI schließt 0 aus“ ist für eine Gleichheitsaussage nicht anwendbar und wird durch
  diese strengere Bedingung ersetzt).

**Statistik:** gepaarter Bootstrap über die val-Bilder, 1000 Resamples, Seed 0, Perzentil-KI (2,5 %/97,5 %); pro
Resample werden beide Methoden auf derselben Bildstichprobe neu ausgewertet (`scahi/eval/bootstrap.py`, gewichtete
Neuauswertung ohne erneutes Matching; exakt für Gewicht 1, Test `tests/test_bootstrap.py`).

**Seeds:** Primäranalyse = Seed 0 (vorab festgelegt, keine Auswahl des besten Seeds). Liegen weitere Seeds von T4 vor,
werden Mittelwert ± Standardabweichung der Deltas (gleiche Auswahl, gleiche Parameter) sekundär berichtet. Erfüllt
Seed 0 ein Ziel, der Seed-Mittelwert aber nicht die Schwelle, lautet das Urteil für dieses Ziel „unklar“.

## 3. Hypothesen

- **ε (Rauschschwelle)** = max(0,3; 2·σ_seed) AP, σ_seed = Stichproben-Standardabweichung (ddof = 1) der
  Toolkit-Port-AP von B-FULL 640 (o2o @ 640, val) über die Seeds 0, 1, 2. Zahlenwert: siehe Ergänzungen
  (wird nach den Seed-Läufen eingetragen, vor dem ersten Phase-3-Training).
- **H1:** T4 gegen T2 (jeweils Seed 0), gleicher Stapel und gleiche Fusion F0-NMS/IOS (je Modell auf calib getunt);
  Stapel = der für T4 mit F0-NMS/IOS auf calib beste Stapel; Zweig o2o. Erfüllt, wenn ΔAP > ε (val); KI berichtet.
- **H2:** beste F-Variante (F1–F5, F3e; auf calib gewählt) gegen F0-NMS/IOS, gleiche Gewichte (T4, Seed 0), gleiche
  Pässe (Stapel von M_ab), gleicher Zweig (der von M_ab). Erfüllt, wenn ΔAP > ε (val); KI berichtet.
- **H3:** o2m gegen o2o unter der besten F-Variante (gleiche Gewichte, Stapel, Parameter-Tuning je Zweig auf calib):
  nur Richtung und KI, keine Schwelle.

## 4. Auswahl- und Datenregeln

- Alle Hyperparameter (Fusionsparameter, conf, Schwellen, Overlap, Stapel, Variante, Zweig) werden auf calib bestimmt;
  auf val wird nur berichtet. Keine Auswahl anhand von val.
- test-dev ausschließlich nach Freeze von `configs/final/` und Freigabe durch den Nutzer.
- Fehlende Läufe werden als fehlend berichtet; ein Ziel, dessen Läufe fehlen, gilt als „unklar“.

## Ergänzungen

- 2026-10-23 – ε-Zahlenwert (automatisch vor dem ersten Phase-3-Training, Formel aus Abschnitt 3): AP der Seeds 0: 22.861, 1: 22.627, 2: 22.683; σ_seed = 0.122 AP; **ε = 0.300 AP**.
- 2026-10-23 – **Zweite (sekundäre) Referenz „stärkste Baseline“** (Nutzervorgabe 23.10. nachmittags). Anlass: Die Referenz R
  (B-MZ-native, val 35,00 AP) wurde festgelegt, bevor das B-FULL-1536-Ergebnis existierte; B-FULL 1536 (o2o @ 1536,
  val 35,75 AP, `runs/p2/20261022_v10s_BFULL1536_full1536_none_s0`) liegt um mehr als ε über R. Festlegung:
  - Das **Hauptkriterium gegen R bleibt unverändert** und wird wie oben geschrieben ausgewertet (Auswahl M_ab/M_c auf calib,
    Aufwandsvergleich gegen R).
  - **Zusätzliche Referenz R2 = stärkste Baseline:** die höchste val-AP (Toolkit-Port) unter allen Referenzzeilen zum Zeitpunkt
    des Freeze; Referenzzeilen = B-FULL 640/1280/1536 (und B-FULL 2048, falls gerechnet), B-SAHI-FT, B-SAHI (B-FULL-1280-
    Gewichte), B-MZ-native, B-MZ S4/S1, B-MZ-1536-native (falls gerechnet), jeweils o2o-Hauptzeile mit der auf calib
    getunten Nachbearbeitung. Derzeit R2 = B-FULL 1536.
  - Der Abschlussbericht wertet die Ziele **(a)–(c) gegen R und gegen R2** aus, jeweils mit gepaartem Bootstrap-KI (wie
    Abschnitt 2). Für R2 wird der Rechenaufwand von R2 als Vergleichsgröße angegeben; M_ab und M_c bleiben die auf calib
    gegen R ausgewählten Kandidaten (keine neue Auswahl).
  - Begründung: Der Nachtrag **verschärft nur die Messlatte** (eine Behauptung muss zusätzlich gegen die stärkste Baseline
    standhalten); er wird geschrieben, bevor ein val-Fusionsergebnis eines T-Modells existiert (bisher val-Fusionszeilen
    nur für die B-FULL-1280-Gewichte aus Phase 5, nicht eingesehen).
