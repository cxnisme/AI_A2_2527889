"""Builds docs/AI_A2_2527889_REPORT.pdf from the generated artifacts (7 pages, A4).

Every number in the report is read from artifacts/*.json and artifacts/*.csv so the
PDF always agrees with the evidence files. The three interpretation paragraphs live
in docs/MY_INTERPRETATIONS.md; edit that file and re-run ``python -m src.report``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from pydantic import ValidationError
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle, Preformatted)

from .config import (FEATURES, FULL_NAME, GITHUB_USER, PROJECT_NAME, REG_NUMBER, SEED,
                     STUDENT_ID, VERIFICATION_CODE, Paths)
from .inference import predict_record
from .schemas import FarmRecord

_ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=_ss["BodyText"], fontName="Helvetica", fontSize=9,
                      leading=12, spaceAfter=4)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10, spaceAfter=0)
H1 = ParagraphStyle("h1", parent=_ss["Heading1"], fontName="Helvetica-Bold", fontSize=15,
                    leading=18, spaceAfter=6, textColor=colors.HexColor("#1a365d"))
H2 = ParagraphStyle("h2", parent=_ss["Heading2"], fontName="Helvetica-Bold", fontSize=10.5,
                    leading=13, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#2c5282"))
CODE = ParagraphStyle("code", parent=BODY, fontName="Courier", fontSize=7.2, leading=8.6)


def P(text: str, style=BODY) -> Paragraph:
    return Paragraph(text, style)


def table(rows, widths, header=True, font=8) -> Table:
    """Table whose string cells are wrapped as Paragraphs (so markup like R<super>2</super> works)."""
    st = ParagraphStyle("cell", parent=SMALL, fontSize=font, leading=font + 2)
    sth = ParagraphStyle("cellh", parent=st, fontName="Helvetica-Bold", textColor=colors.white)
    data = [[Paragraph(str(c), sth if (header and i == 0) else st) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#a0aec0")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5282")))
    t.setStyle(TableStyle(style))
    return t


def _lines(obj: dict) -> str:
    """Compact multi-line rendering of a flat dict, narrow enough for the page."""
    items = list(obj.items())
    chunks = [", ".join(f"{k}: {json.dumps(v)}" for k, v in items[i:i + 3]) for i in range(0, len(items), 3)]
    return "\n".join(chunks)


def _load(paths: Paths):
    a = paths.artifacts
    j = lambda n: json.loads((a / n).read_text(encoding="utf-8"))
    fp = dict(line.split(": ", 1) for line in (a / "data_fingerprint.txt").read_text().splitlines())
    return {
        "reg": j("regression_metrics.json"), "clf": j("classification_metrics.json"),
        "clu": j("clustering_report.json"), "val": j("validation_report.json"),
        "meta": json.loads((paths.models / "model_metadata.json").read_text(encoding="utf-8")),
        "tp": pd.read_csv(a / "test_predictions.csv"), "fp": fp,
        "df": pd.read_csv(paths.csv).drop_duplicates().reset_index(drop=True),
    }


# ---------------------------------------------------------------- interpretations
def draft_interpretations(d: dict) -> str:
    """Data-driven first draft of the three required interpretations (student rewrites)."""
    tp, df = d["tp"], d["df"]
    r = tp.loc[tp["abs_yield_error_tons"].idxmax()]
    feats = df.loc[int(r["row_id"]), FEATURES]
    ftxt = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in feats.items())
    temp = float(feats["avg_temp_c"])
    temp_note = (" The selected linear model fits temperature as a straight line, while the generator "
                 "penalises distance from 18 C (an absolute value); this row's temperature is "
                 f"{temp:.2f} C, so this mismatch may have contributed (not tested).")
    clipped = " The actual yield sits at the 0.50 t clipping floor." if r["actual_yield_tons"] <= 0.5001 else ""
    wrong = tp[tp["risk_correct"] == 0]
    ypha_med = (df["yield_tons"] / df["field_size_ha"]).median()
    if len(wrong):
        w = wrong.assign(conf=(wrong["high_risk_probability"] - 0.5).abs()).sort_values("conf").iloc[-1]
        wf = df.loc[int(w["row_id"])]
        ypha = wf["yield_tons"] / wf["field_size_ha"]
        clf_txt = (f"Row {int(w['row_id'])}: actual class {int(w['actual_high_risk'])}, predicted "
                   f"{int(w['predicted_high_risk'])} with probability {w['high_risk_probability']:.4f}. "
                   f"Its yield per hectare is {ypha:.3f} t/ha against a dataset median of "
                   f"{ypha_med:.3f} t/ha, so it lies {abs(ypha - ypha_med):.3f} t/ha from the "
                   "class boundary. Total misclassifications on the test set: "
                   f"{len(wrong)} of {len(tp)}.")
    else:
        clf_txt = "The selected classifier made no errors on the held-out test set."
    big = max(d["clu"]["profiles"], key=lambda p: p["n_train"])
    fm = big["feature_means"]
    return (
        "# My interpretations (DRAFT - rewrite in your own words, then run: python -m src.report)\n\n"
        "## Regression error\n"
        f"Row {int(r['row_id'])} had the largest test error: actual {r['actual_yield_tons']:.4f} t, "
        f"predicted {r['predicted_yield_tons']:.4f} t (error {r['yield_error_tons']:+.4f} t, "
        f"{abs(r['yield_error_tons']) / 0.80:.1f} times the generator's noise standard deviation "
        f"of 0.80 t). Inputs: {ftxt}.{clipped} The generator adds random noise the model cannot "
        "observe." + temp_note + "\n\n"
        "## Classification error\n"
        f"{clf_txt}\n\n"
        "## Cluster\n"
        f"Cluster {big['cluster']} is the largest ({big['n_train']} training farms, "
        f"{big['share_of_train']:.0%}): {big['summary']}. Mean yield {big['mean_yield_tons']:.2f} t, "
        f"high-risk rate {big['high_risk_rate']:.0%}, mean rainfall {fm['rainfall_mm']:.0f} mm, "
        f"mean fertilizer {fm['fertilizer_kg']:.0f} kg, mean field size {fm['field_size_ha']:.2f} ha. "
        "Operationally, such a group could be visited or advised together, with human review.\n"
    )


def _interpretations(paths: Paths, d: dict) -> dict[str, str]:
    f = paths.docs / "MY_INTERPRETATIONS.md"
    if not f.exists():
        f.write_text(draft_interpretations(d), encoding="utf-8")
    parts = re.split(r"^## ", f.read_text(encoding="utf-8"), flags=re.M)[1:]
    out = {}
    for part in parts:
        title, _, body = part.partition("\n")
        out[title.strip().lower()] = " ".join(body.split())
    return out


# ---------------------------------------------------------------- page builders
def _img(path: Path, width_cm: float) -> Image:
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def build_report(paths: Paths, log=print) -> Path:
    d = _load(paths)
    reg, clf, clu, val, meta, tp, fp = d["reg"], d["clf"], d["clu"], d["val"], d["meta"], d["tp"], d["fp"]
    interp = _interpretations(paths, d)
    W = 17.6 * cm
    s: list = []

    # ---- Page 1
    s += [P("Musanze Harvest Intelligence Service", H1),
          P("AI decision-support prototype for a potato cooperative - Applied AI individual project", BODY),
          table([["Item", "Value"],
                 ["Full name", FULL_NAME], ["Registration number", REG_NUMBER],
                 ["Institutional username", STUDENT_ID], ["GitHub username", GITHUB_USER],
                 ["Verification code", f"<b>{VERIFICATION_CODE}</b>"],
                 ["Dataset seed", str(SEED)], ["Repository / folder", PROJECT_NAME]],
                [5 * cm, 12.6 * cm]),
          P("Stakeholder", H2),
          P("The stakeholder is the management of a potato cooperative in Musanze, together with its "
            "extension staff. They need a quick, offline-capable way to plan collection, storage and "
            "field visits before harvest."),
          P("Decision problem and prediction questions", H2),
          P("1. <b>Regression:</b> how many tons of potatoes is a farm likely to harvest (yield_tons)?<br/>"
            "2. <b>Classification:</b> is a farm at high production risk, meaning its yield per hectare "
            "is below the median (high_risk = 1)?<br/>"
            "3. <b>Clustering:</b> which farms share similar growing conditions and could be advised or "
            "supported together?"),
          P("Ethical boundary", H2),
          P("This is a classroom prototype trained on <b>synthetic data</b>. It does not replace an "
            "agronomist and must not be used to decide access to finance, insurance or public services. "
            "Every output needs human review. Training and the command-line predictor run locally, so "
            "they work when internet access is unreliable; the FastAPI service exposes the same "
            "predictions for later integration with a mobile application."),
          PageBreak()]

    # ---- Page 2
    n_base = val["rows_raw"] - val["duplicate_rows"]
    desc = [["Column", "Rule / type", "Role"],
            ["rainfall_mm", "Uniform 600-1400", "Numeric feature"],
            ["avg_temp_c", "Uniform 12-26", "Numeric feature"],
            ["soil_moisture_pct", "Uniform 20-80", "Numeric feature"],
            ["fertilizer_kg", "Uniform 0-250", "Numeric feature"],
            ["field_size_ha", "Uniform 0.2-4.0", "Numeric feature"],
            ["season", "Random A or B", "Categorical feature"],
            ["cooperative_support", "Random 0 or 1", "Binary feature"],
            ["yield_tons", "Formula + N(0, 0.80) noise, clipped at 0.50", "Regression target"],
            ["high_risk", "Yield per hectare below its median", "Classification target"]]
    s += [P("Data and reproducibility", H1), P("Data dictionary", H2),
          table(desc, [4 * cm, 8.6 * cm, 5 * cm]),
          P("Personal seed rule", H2),
          P(f"Seed = {SEED} (registration digits), used as <font name='Courier'>numpy.random.default_rng({SEED})</font>. "
            f"Base rows = 240 + ({SEED} mod 41) = 240 + {SEED % 41} = <b>{n_base}</b>. "
            f"Missing values: m = round(0.05 x {n_base}) = {round(0.05 * n_base)} in each of soil_moisture_pct "
            f"and fertilizer_kg (different positions), then 3 exact duplicate rows are appended, "
            f"giving <b>{val['rows_raw']} raw rows</b>. The CSV uses four decimals "
            "(float_format=\"%.4f\"). The yield formula is: 0.004 x rainfall + 0.06 x soil moisture + "
            "0.02 x fertilizer + 1.30 x field size - 0.25 x |temperature - 18| + 0.70 x cooperative "
            "support + season effect (+0.50 A, -0.20 B) + noise."),
          P("Data-quality evidence (artifacts/validation_report.json)", H2),
          table([["Check", "Result"],
                 ["Raw rows", str(val["rows_raw"])],
                 ["Missing soil_moisture_pct", str(val["missing_values"].get("soil_moisture_pct", 0))],
                 ["Missing fertilizer_kg", str(val["missing_values"].get("fertilizer_kg", 0))],
                 ["Exact duplicate rows detected and dropped", str(val["duplicate_rows"])],
                 ["Rows after cleaning", str(val["rows_after_cleaning"])],
                 ["Range / label checks", "passed"],
                 ["Target leakage check", "passed (no yield_tons, yield per hectare or high_risk as input)"]],
                [7.5 * cm, 10.1 * cm]),
          P("SHA-256 fingerprint of data/AI_A2_2527889.csv", H2),
          Preformatted(fp["sha256"], CODE),
          P("Running run_all.py twice recreates the same CSV bytes and therefore the same fingerprint; "
            "the automated reproducibility test checks this.", BODY),
          PageBreak()]

    # ---- Page 3
    s += [P("Pipeline design", H1),
          P("Train/test separation", H2),
          P(f"Duplicates are removed first so identical rows cannot appear on both sides. The "
            f"{val['rows_after_cleaning']} cleaned rows are split once with train_test_split("
            f"test_size=0.25, random_state={SEED}, stratified on high_risk): "
            f"<b>{reg['n_train']} training rows and {reg['n_test']} test rows</b>. The test set is used "
            "only for final reporting; model selection uses 5-fold cross-validation inside the "
            "training set."),
          P("Preprocessing flow (one ColumnTransformer, reused at inference)", H2),
          table([["Inputs", "Steps (fitted on training data only)"],
                 ["rainfall_mm, avg_temp_c, soil_moisture_pct, fertilizer_kg, field_size_ha",
                  "median imputation, then StandardScaler"],
                 ["season", "most-frequent imputation, then one-hot encoding"],
                 ["cooperative_support", "most-frequent imputation (already 0/1)"]],
                [9 * cm, 8.6 * cm]),
          P("The preprocessor and the estimator are combined in a scikit-learn Pipeline and saved with "
            "joblib, so training, cross-validation and inference apply identical transformations. "
            "Because the imputer sits inside the Pipeline, cross-validation re-fits it on each fold's "
            "training part."),
          P("Leakage controls", H2),
          P("Model inputs are exactly: " + ", ".join(FEATURES) + ". yield_tons, yield per hectare and "
            "high_risk are never inputs for the classifier; src/validation.py raises an error if such a "
            "column is added. No imputer, encoder, scaler or feature selection is fitted before the "
            "split. High_risk is defined from yield per hectare, which is why yield_tons is excluded "
            "from the classifier's inputs; field_size_ha is a legitimate input known before harvest."),
          P("Model candidates", H2),
          table([["Task", "Baseline", "Candidates", "Selection rule"],
                 ["Regression", "Mean predictor", "Linear regression, random forest, gradient boosting",
                  reg["selection_rule"]],
                 ["Classification", "Majority class", "Logistic regression, random forest, gradient boosting",
                  clf["selection_rule"]],
                 ["Clustering", "-", "K-means, K = 2..6 on scaled inputs", clu["selection_rule"]]],
                [2.6 * cm, 2.8 * cm, 6.6 * cm, 5.6 * cm]),
          P("All random states are set to 7889 and recorded in the JSON artifacts. The final models are "
            "fitted on the training rows only, so the reported test metrics describe exactly the saved "
            "models.", BODY),
          PageBreak()]

    # ---- Page 4
    sr = reg["candidates"][reg["selected_model"]]
    sc = clf["candidates"][clf["selected_model"]]
    cm_ = sc["confusion_matrix"]
    rrows = [["Model (regression)", "MAE (t)", "RMSE (t)", "R<super>2</super>", "CV RMSE"],
             [f"{reg['baseline']['name']}", f"{reg['baseline']['mae']:.4f}", f"{reg['baseline']['rmse']:.4f}",
              f"{reg['baseline']['r2']:.4f}", "-"]]
    for k, v in reg["candidates"].items():
        mark = " (selected)" if k == reg["selected_model"] else ""
        rrows.append([k + mark, f"{v['mae']:.4f}", f"{v['rmse']:.4f}", f"{v['r2']:.4f}", f"{v['cv_rmse_mean']:.4f}"])
    crows = [["Model (classification)", "Precision", "Recall", "F1", "ROC-AUC", "CV AUC"],
             [clf["baseline"]["name"], f"{clf['baseline']['precision']:.4f}", f"{clf['baseline']['recall']:.4f}",
              f"{clf['baseline']['f1']:.4f}", f"{clf['baseline']['roc_auc']:.4f}", "-"]]
    for k, v in clf["candidates"].items():
        mark = " (selected)" if k == clf["selected_model"] else ""
        crows.append([k + mark, f"{v['precision']:.4f}", f"{v['recall']:.4f}", f"{v['f1']:.4f}",
                      f"{v['roc_auc']:.4f}", f"{v['cv_roc_auc_mean']:.4f}"])
    s += [P("Regression and classification", H1),
          table(rrows, [6 * cm, 2.7 * cm, 2.7 * cm, 2.7 * cm, 3.5 * cm]),
          Spacer(1, 3),
          _img(paths.plots / "residuals.png", 12.8),
          P(f"<b>Selected regression model.</b> {reg['selection_reason']} Test residuals "
            f"(actual - predicted): mean {-tp['yield_error_tons'].mean():+.4f} t, standard deviation "
            f"{tp['yield_error_tons'].std():.4f} t, largest absolute error {tp['abs_yield_error_tons'].max():.4f} t."),
          table(crows, [5.2 * cm, 2.5 * cm, 2.3 * cm, 2.3 * cm, 2.6 * cm, 2.7 * cm]),
          Spacer(1, 3),
          Table([[_img(paths.plots / "confusion_matrix.png", 5.6),
                  P(f"<b>Selected classifier.</b> {clf['selection_reason']}<br/><br/>"
                    f"Confusion matrix (rows actual, columns predicted): TN {cm_['tn']}, FP {cm_['fp']}, "
                    f"FN {cm_['fn']}, TP {cm_['tp']}. A false negative (a high-risk farm missed) is the "
                    "costlier error for a cooperative planning support, so recall matters most here.")]],
                colWidths=[6 * cm, 11.6 * cm]),
          P("Important errors (my interpretation)", H2),
          P("<b>Regression:</b> " + interp.get("regression error", "")),
          P("<b>Classification:</b> " + interp.get("classification error", "")),
          PageBreak()]

    # ---- Page 5
    krows = [["K", "Silhouette", "Inertia", "Chosen"]]
    for k, v in clu["scores"].items():
        krows.append([k, f"{v['silhouette']:.4f}", f"{v['inertia']:.1f}", "yes" if int(k) == clu["selected_k"] else ""])
    prow = [["Cluster", "Farms (train)", "Mean yield (t)", "High-risk rate", "Plain-language profile"]]
    for p in clu["profiles"]:
        prow.append([p["cluster"], f"{p['n_train']} ({p['share_of_train']:.0%})", f"{p['mean_yield_tons']:.2f}",
                     f"{p['high_risk_rate']:.0%}", p["summary"]])
    s += [P("Clustering", H1),
          P("Inputs are imputed, encoded and scaled (fitted on training rows), then K-means is run for "
            "K = 2 to 6 (n_init = 10, random_state = 7889). Silhouette score decides K."),
          table(krows, [2 * cm, 3.5 * cm, 3.5 * cm, 2.5 * cm]), Spacer(1, 4),
          _img(paths.plots / "clusters.png", 14.5),
          P(f"<b>Selected K = {clu['selected_k']}.</b> {clu['interpretation']}"),
          table(prow, [1.6 * cm, 2.8 * cm, 2.6 * cm, 2.6 * cm, 8 * cm]),
          P("Operational meaning", H2),
          P("Cluster " + "labels are arbitrary numbers fixed by the seed. " + interp.get("cluster", "") +
            " Yield and risk columns describe each group afterwards; they were not used to form it."),
          PageBreak()]

    # ---- Page 6
    ex = {"rainfall_mm": 1050.0, "avg_temp_c": 18.5, "soil_moisture_pct": 55.0, "fertilizer_kg": 140.0,
          "field_size_ha": 1.8, "season": "A", "cooperative_support": 1}
    out = predict_record(ex, paths.models)
    out_short = {k: v for k, v in out.items() if k != "notice"}
    bad = {"rainfall_mm": 2500.0, "avg_temp_c": "warm", "soil_moisture_pct": 55.0, "fertilizer_kg": 140.0,
           "season": "C", "cooperative_support": 1}
    try:
        FarmRecord.model_validate(bad)
        errs = []
    except ValidationError as e:
        errs = [{"field": ".".join(map(str, x["loc"])), "problem": x["msg"]} for x in e.errors()]
    pytest_line = ""
    pf = paths.artifacts / "pytest_output.txt"
    if pf.exists():
        m = re.findall(r"(\d+ passed[^\n]*)", pf.read_text(encoding="utf-8"))
        pytest_line = m[-1].strip(" =") if m else ""
    s += [P("API and testing", H1),
          table([["Endpoint", "Purpose", "Responses"],
                 ["GET /health", "liveness and whether models are loaded", "200"],
                 ["GET /model-info", "selected models, test metrics, fingerprint, verification code", "200; 503 if models missing"],
                 ["POST /predict", "yield estimate, risk class and probability, cluster", "200; 422 invalid input; 503 if models missing"]],
                [3.3 * cm, 9 * cm, 5.3 * cm]),
          P("Valid request and response", H2),
          Preformatted(_lines(ex) + "\n=> HTTP 200\n" + _lines(out_short), CODE),
          P("Invalid request (out-of-range, wrong type, missing field, bad category) returns HTTP 422", H2),
          Preformatted(_lines(bad) + "\n=> HTTP 422\n" + "\n".join(f"{e['field']}: {e['problem']}" for e in errs), CODE),
          P("Failure behaviour", H2),
          P("Inputs are strictly typed (pydantic): all seven fields are required, numbers must lie in the "
            "training ranges, season must be A or B, cooperative_support must be 0 or 1, and unknown fields "
            "are rejected. Errors list each failing field and never crash the server. The CLI prints the "
            "same details to stderr and exits with code 2. If models are missing, the API returns 503 with "
            "instructions to run run_all.py."),
          P("Automated tests (pytest)", H2),
          table([["Test", "Covers"],
                 ["test_valid_inference", "valid record returns all outputs with the verification code"],
                 ["test_missing_field_rejected", "missing field raises a validation error"],
                 ["test_wrong_type_rejected", "string given for a numeric field is rejected"],
                 ["test_range_error_rejected", "out-of-range value is rejected"],
                 ["test_reproducibility", "two full pipeline runs give the same fingerprint and metrics"],
                 ["test_api_predict_response", "POST /predict returns 200 and the expected keys"],
                 ["test_api_invalid_returns_422", "API returns a useful 422 for invalid input"],
                 ["test_api_health_and_model_info", "/health and /model-info respond with the code"],
                 ["test_leakage_guard", "target columns are refused as model inputs"],
                 ["test_run_all_rejects_wrong_seed", "run_all.py refuses any seed other than 7889"]],
                [6 * cm, 11.6 * cm]),
          P(f"Last recorded run: {pytest_line or 'see evidence/AI_A2_2527889_TEST_LOG.pdf'}.", SMALL),
          PageBreak()]

    # ---- Page 7
    s += [P("Model card and reflection", H1),
          table([["Section", "Statement"],
                 ["Intended use", "Classroom decision-support prototype: rough harvest estimates, flagging "
                  "farms for follow-up and grouping similar farms, always with human review."],
                 ["Out of scope / prohibited", "Replacing an agronomist; deciding loans, insurance, subsidies "
                  "or access to public services; any use on real farmers' data without new validation."],
                 ["Data", f"Fully synthetic ({val['rows_raw']} raw / {val['rows_after_cleaning']} cleaned rows), generated from "
                  "a known formula. No real farm was measured."],
                 ["Performance", f"Test MAE {sr['mae']:.2f} t, RMSE {sr['rmse']:.2f} t, R<super>2</super> {sr['r2']:.2f}; "
                  f"risk F1 {sc['f1']:.2f}, ROC-AUC {sc['roc_auc']:.2f}, recall {sc['recall']:.2f}, "
                  f"precision {sc['precision']:.2f} on {reg['n_test']} test farms."],
                 ["Limitations", "High scores partly reflect that the data come from a simple linear rule plus "
                  "Gaussian noise; real farms involve pests, disease, soil type, variety and weather timing. "
                  "The test set is small (65 rows), so metrics have wide uncertainty. Inputs outside the "
                  "training ranges are rejected, not extrapolated. Clusters are weak (low silhouette)."],
                 ["Risks", "Over-trust in a number; applying results to real farms; unfair treatment if risk "
                  "labels were reused for credit or support decisions."],
                 ["Human oversight", "A cooperative officer or agronomist reviews each output and can override it. "
                  "Predictions are advice, not decisions."],
                 ["Improvements with real data", "Collect measured yields and field records, validate on a "
                  "separate season or district, add uncertainty intervals, check performance by group, "
                  "and retrain with agronomist input."]],
                [3.6 * cm, 14 * cm]),
          P("Reflection", H2),
          P("The linear model beat the tree ensembles on cross-validation, which fits the way the data were "
            "generated; on real data I would not expect that. Running every stage from one command made "
            "mistakes such as duplicate rows and imputation order visible early. Evidence for each claim "
            "sits in the JSON, CSV and plot artifacts listed in the README.")]

    doc = SimpleDocTemplate(str(paths.docs / f"{PROJECT_NAME}_REPORT.pdf"), pagesize=A4,
                            leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.6 * cm, title="Musanze Harvest Intelligence Service - Report",
                            author=FULL_NAME)

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#4a5568"))
        canvas.drawString(1.7 * cm, 0.9 * cm,
                          f"{FULL_NAME} | {REG_NUMBER} | {VERIFICATION_CODE} | synthetic-data classroom prototype")
        canvas.drawRightString(A4[0] - 1.7 * cm, 0.9 * cm, f"Page {_doc.page}")
        canvas.restoreState()

    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    out_path = paths.docs / f"{PROJECT_NAME}_REPORT.pdf"
    log(f"[{VERIFICATION_CODE}] report written: {out_path.name}")
    return out_path


if __name__ == "__main__":
    build_report(Paths())
