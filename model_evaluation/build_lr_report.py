from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "logistic_regression"
METRICS = json.loads((OUT / "metrics.json").read_text())
REPORT = OUT / "Logistic_Regression_Experiment_Report.docx"

NAVY = "17365D"
BLUE = "2F75B5"
PALE_BLUE = "EAF2F8"
LIGHT_GRAY = "D9E2F3"
MID_GRAY = "666666"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_border(cell, color="D9D9D9"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "6")
        element.set(qn("w:color"), color)


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn("w:" + side))
        if node is None:
            node = OxmlElement("w:" + side)
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def style_table(table, header=True, widths=None):
    table.style = "Table Grid"
    table.autofit = False
    for ri, row in enumerate(table.rows):
        for ci, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell)
            set_cell_border(cell)
            if widths:
                cell.width = Inches(widths[ci])
            for para in cell.paragraphs:
                para.paragraph_format.space_after = Pt(0)
                for run in para.runs:
                    run.font.name = "Aptos"
                    run._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
                    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
                    run.font.size = Pt(9.5)
            if header and ri == 0:
                set_cell_shading(cell, NAVY)
                for run in cell.paragraphs[0].runs:
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(255, 255, 255)
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif ri % 2 == 0:
                set_cell_shading(cell, PALE_BLUE)


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_body(doc, text, bold_start=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.12
    if bold_start and text.startswith(bold_start):
        run = p.add_run(bold_start)
        run.bold = True
        p.add_run(text[len(bold_start):])
    else:
        p.add_run(text)
    return p


def pct(x):
    return f"{100*x:.2f}%"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.72)
    sec.bottom_margin = Inches(0.68)
    sec.left_margin = Inches(0.78)
    sec.right_margin = Inches(0.78)

    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    styles["Normal"]._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    styles["Normal"].font.size = Pt(10.5)
    for name, size in (("Title", 22), ("Heading 1", 14), ("Heading 2", 11.5)):
        styles[name].font.name = "Aptos Display" if name != "Normal" else "Aptos"
        styles[name]._element.rPr.rFonts.set(qn("w:ascii"), styles[name].font.name)
        styles[name]._element.rPr.rFonts.set(qn("w:hAnsi"), styles[name].font.name)
        styles[name].font.size = Pt(size)
        styles[name].font.bold = True
        styles[name].font.color.rgb = RGBColor(0, 0, 0)

    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = header.add_run("PATHSNAP  |  MEMBER 4 MODEL EXPERIMENT")
    run.font.name = "Aptos"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor.from_string(MID_GRAY)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("Logistic Regression Experiment Report")
    run.font.name = "Aptos"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor.from_string(MID_GRAY)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.add_run("Logistic Regression Experiment Report")
    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(14)
    r = subtitle.add_run("PathSnap candidate level road segment matching  |  Member 4")
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor.from_string(MID_GRAY)

    add_heading(doc, "Experiment conclusion")
    add_body(doc, "The Logistic Regression baseline generalised consistently from grouped cross validation to the held out student test partition. It achieved a mean cross validation Macro F1 of 71.13% and a student test Macro F1 of 70.64%. The model is fast, interpretable, and produces probabilities that can later be used as HMM or Viterbi emission scores.")
    add_body(doc, "The main limitation is candidate coverage rather than only classifier quality. The reference road segment was absent from 25.82% of student test candidate sets. When the reference candidate was available, the model selected it as the highest probability candidate in 81.90% of observations.")

    add_heading(doc, "Key results")
    st = METRICS["student_test"]
    key_rows = [
        ("5 fold CV Macro F1", f"{METRICS['cv_macro_f1_mean']:.4f} ± {METRICS['cv_macro_f1_std']:.4f}", "Primary grouped validation result"),
        ("Student test Macro F1", f"{st['candidate_macro_f1']:.4f}", "Primary candidate classification result"),
        ("Student test Top 1 accuracy", pct(st["observation_top1_accuracy"]), "Correct highest scoring candidate over all observations"),
        ("Top 1 accuracy with reference candidate", pct(st["observation_top1_accuracy_when_reference_candidate_present"]), "Classifier decision quality when a correct candidate exists"),
        ("Candidate miss rate", pct(st["candidate_miss_rate"]), "Reference segment absent from the generated candidate set"),
        ("Student test log loss", f"{st['log_loss']:.4f}", "Probability quality lower is better"),
    ]
    table = doc.add_table(rows=1, cols=3)
    table.rows[0].cells[0].text = "Measure"
    table.rows[0].cells[1].text = "Result"
    table.rows[0].cells[2].text = "Interpretation"
    for row in key_rows:
        cells = table.add_row().cells
        for cell, val in zip(cells, row): cell.text = val
    style_table(table, widths=[1.9, 1.4, 3.85])

    add_heading(doc, "1  Objective and data contract")
    add_body(doc, "The task is binary candidate classification for map matching. Each row represents one possible road segment for one GPS observation and has an is_match label. A label of 1 identifies a reference candidate; a label of 0 identifies a non matching candidate. The model ranks the candidates for an observation by predicted match probability and selects the highest scoring one as the Top 1 road segment prediction.")
    add_body(doc, "The experiment used the supplied preprocessed files train_candidates.csv and student_test_candidates.csv. The training file contains 80 trips and the student test file contains 10 separate trips. No evaluator data or evaluator labels were used.")

    add_heading(doc, "2  Model and training configuration")
    config = METRICS["config"]
    model_rows = [
        ("Model", "Binary Logistic Regression"),
        ("Solver", config["solver"]),
        ("Regularisation", f"L2 regularisation with C = {config['C']}"),
        ("Class handling", "Balanced class weights to address rare positive candidates"),
        ("Feature scaling", "StandardScaler fitted within every training fold"),
        ("Objective", config["objective"]),
        ("Decision threshold", str(config["threshold"])),
        ("Maximum iterations and seed", f"{config['max_iter']} iterations; seed {config['seed']}"),
    ]
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Component"
    table.rows[0].cells[1].text = "Choice"
    for row in model_rows:
        cells = table.add_row().cells
        cells[0].text, cells[1].text = row
    style_table(table, widths=[2.0, 5.15])

    add_heading(doc, "Features used", level=2)
    feature_rows = [
        ("distance_m", "Distance from the GPS observation to the candidate road segment in metres"),
        ("distance_gap_m", "Distance relative to the closest candidate"),
        ("heading_cos", "Directional agreement between trajectory movement and candidate arc"),
        ("gps_speed_mps", "Observed vehicle speed"),
        ("candidate_rank", "Rank of this candidate by spatial closeness"),
    ]
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Feature"
    table.rows[0].cells[1].text = "Meaning"
    for row in feature_rows:
        cells = table.add_row().cells
        cells[0].text, cells[1].text = row
    style_table(table, widths=[1.75, 5.4])

    doc.add_page_break()
    add_heading(doc, "3  Evaluation protocol")
    add_body(doc, "Five fold GroupKFold cross validation was performed using trip_id as the grouping variable. All observations from a trip stayed in either the training partition or the validation partition for a fold. This prevents near duplicate observations from the same journey leaking into validation.")
    add_body(doc, "For each fold, scaling and model fitting used only that fold's training trips. After cross validation, a fresh pipeline was fitted on all 80 training trips and evaluated once on the held out 10 trip student test set. This evaluation partition was not used to fit scaling or model parameters.")

    add_heading(doc, "4  Cross validation results")
    add_body(doc, "Table 1 shows the actual results from each trip grouped validation fold. The validation partitions contain different groups of whole trips; no observations from a validation trip were used to fit that fold's scaler or classifier.")
    fold_table = doc.add_table(rows=1, cols=5)
    for c, v in zip(fold_table.rows[0].cells, ("Fold", "Validation trips", "Macro F1", "Top 1", "Log loss")): c.text = v
    for f in METRICS["cv_folds"]:
        cells = fold_table.add_row().cells
        values = (str(f["fold"]), str(f["validation_trips"]), f"{f['candidate_macro_f1']:.4f}", pct(f["observation_top1_accuracy"]), f"{f['log_loss']:.4f}")
        for c, v in zip(cells, values): c.text = v
    style_table(fold_table, widths=[0.65, 1.35, 1.15, 1.15, 1.15])
    add_body(doc, "The Macro F1 values range from 0.6455 to 0.7469, with a standard deviation of 0.0366. Variation partly aligns with different candidate coverage across folds, so this evidence does not indicate strong overfitting. The held out student test Macro F1 of 0.7064 is close to the cross validation mean of 0.7113.")

    add_heading(doc, "5  Student test results")
    cand = st["confusion_matrix_tn_fp_fn_tp"]
    metrics_rows = [
        ("Candidate accuracy", pct(st["candidate_accuracy"])),
        ("Macro F1", f"{st['candidate_macro_f1']:.4f}"),
        ("Positive candidate precision", pct(st["candidate_precision"])),
        ("Positive candidate recall", pct(st["candidate_recall"])),
        ("Log loss", f"{st['log_loss']:.4f}"),
        ("Observations", f"{st['observations']:,}"),
        ("Candidate rows", f"{sum(cand):,}"),
        ("Full train time", f"{METRICS['fit_seconds_full_train']:.3f} seconds"),
        ("Student inference time", f"{METRICS['student_inference_seconds']:.4f} seconds"),
    ]
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Metric"
    table.rows[0].cells[1].text = "Result"
    for row in metrics_rows:
        cells = table.add_row().cells
        cells[0].text, cells[1].text = row
    style_table(table, widths=[3.6, 3.55])

    add_heading(doc, "Confusion matrix", level=2)
    cm = doc.add_table(rows=3, cols=3)
    cm.cell(0, 0).text = "Actual / predicted"
    cm.cell(0, 1).text = "Non match 0"
    cm.cell(0, 2).text = "Match 1"
    cm.cell(1, 0).text = "Non match 0"
    cm.cell(1, 1).text = f"TN {cand[0]:,}"
    cm.cell(1, 2).text = f"FP {cand[1]:,}"
    cm.cell(2, 0).text = "Match 1"
    cm.cell(2, 1).text = f"FN {cand[2]:,}"
    cm.cell(2, 2).text = f"TP {cand[3]:,}"
    style_table(cm, widths=[2.2, 2.0, 2.0])

    add_heading(doc, "6  Interpretation and limitations")
    add_body(doc, "Macro F1 is the main classification metric because it gives equal importance to matching and non matching candidates. Ordinary candidate accuracy is less informative here because most candidate rows are non matches. The student test Macro F1 is close to grouped cross validation, which is the primary evidence that the baseline generalises to unseen trips.")
    add_body(doc, "Top 1 accuracy measures whether the single highest probability candidate for each GPS observation is a reference candidate. The overall result is 60.76%, but this includes 4,947 observations where no reference candidate was generated. Conditioning on the 14,213 observations where a reference candidate was available, Top 1 accuracy is 81.90%. Therefore, candidate generation coverage is an important upstream limitation.")
    add_body(doc, "The model has high recall (91.18%) and lower precision (36.74%) at the fixed 0.5 candidate threshold. This is consistent with balanced class weighting, which prioritises retaining positive candidates. The downstream route decoder should use the continuous predicted probabilities rather than only thresholded labels.")

    add_heading(doc, "7  Reproducibility")
    add_body(doc, "Run the following command from the repository root after placing the permitted preprocessed candidate files locally:")
    code = doc.add_paragraph()
    code.paragraph_format.left_indent = Inches(.25)
    code.paragraph_format.space_before = Pt(3)
    code.paragraph_format.space_after = Pt(6)
    rr = code.add_run("python models/logistic_regression.py --train train_candidates.csv --student-test student_test_candidates.csv --out results/logistic_regression")
    rr.font.name = "Courier New"
    rr._element.rPr.rFonts.set(qn("w:ascii"), "Courier New")
    rr.font.size = Pt(8.5)
    add_body(doc, "Evidence files: results/logistic_regression/metrics.json, fold_metrics.csv, student_top1_predictions.csv, and logistic_regression.joblib. The source script stores the fitted StandardScaler and Logistic Regression classifier together so that inference uses the same feature order and preprocessing.")
    add_body(doc, "Report scope: this document reports only the Member 4 Logistic Regression experiment. It does not select the team's final model and does not claim HMM or Viterbi route reconstruction performance.")

    doc.core_properties.title = "Logistic Regression Experiment Report"
    doc.core_properties.author = "PathSnap Member 4"
    doc.core_properties.subject = "PathSnap Logistic Regression experiment"
    doc.save(REPORT)
    print(REPORT)


if __name__ == "__main__":
    main()
