"""Pure-function tests for scripts/eval_suite.py metrics (no models, no datasets, no torch).
Run: python tests/test_eval_metrics.py   (or python -m pytest tests/test_eval_metrics.py -q)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.eval_suite import (  # noqa: E402
    apply_overrides, BENCHMARKS, exact_match, markdown_row, mme_scores, normalize_vqa, parse_choice,
    parse_yes_no, pope_metrics, sqa_prompt, vqa_soft_accuracy,
)


# ---- VQA normaliser -----------------------------------------------------------------------
def test_normalize_lowercase_and_punctuation():
    assert normalize_vqa("  Yes.  ") == "yes"
    assert normalize_vqa("Dakota Digital!") == "dakota digital"
    assert normalize_vqa("red, white") == "red white"
    assert normalize_vqa("stop-sign") == "stop sign"


def test_normalize_articles_and_numbers():
    assert normalize_vqa("a red car") == "red car"
    assert normalize_vqa("The two dogs") == "2 dogs"
    assert normalize_vqa("none") == "0"
    assert normalize_vqa("ten") == "10"
    assert normalize_vqa("An apple and the orange") == "apple and orange"


def test_normalize_keeps_numbers_intact():
    assert normalize_vqa("1.5") == "1.5"
    assert normalize_vqa("1,000") == "1000"
    assert normalize_vqa("3.") == "3"


def test_normalize_contractions():
    assert normalize_vqa("dont know") == "don't know"
    assert normalize_vqa("don't know") == "don't know"
    assert normalize_vqa("I'm") == "i'm"


# ---- soft accuracy -------------------------------------------------------------------------
def test_soft_accuracy_rule():
    answers = ["dakota"] * 5 + ["dakota digital"] * 3 + ["nous les gosses", "clos culombu"]
    assert vqa_soft_accuracy("Dakota", answers) == 1.0                 # 5 matches -> min(5/3, 1)
    assert vqa_soft_accuracy("dakota digital.", answers) == 1.0        # 3 matches -> exactly 1
    assert abs(vqa_soft_accuracy("nous les gosses", answers) - 1 / 3) < 1e-9
    assert vqa_soft_accuracy("canon", answers) == 0.0


def test_soft_accuracy_two_matches():
    answers = ["copenhagen"] * 2 + ["thursday"] * 8
    assert abs(vqa_soft_accuracy("Copenhagen", answers) - 2 / 3) < 1e-9
    assert vqa_soft_accuracy("thursday", answers) == 1.0


def test_exact_match():
    assert exact_match("The women", "women")
    assert exact_match("No.", "no")
    assert not exact_match("man", "women")


# ---- yes/no + POPE F1 ----------------------------------------------------------------------
def test_parse_yes_no():
    assert parse_yes_no("Yes, there is.") == "yes"
    assert parse_yes_no("No") == "no"
    assert parse_yes_no("no, but yes later") == "no"
    assert parse_yes_no("I cannot tell") == "no"


def test_pope_metrics_perfect():
    m = pope_metrics(["yes", "no", "yes", "no"], ["yes", "no", "yes", "no"])
    assert m["accuracy"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["yes_ratio"] == 0.5


def test_pope_metrics_f1():
    #             tp     fp     fn     tn
    preds = ["yes", "yes", "no", "no", "yes"]
    golds = ["yes", "no", "yes", "no", "yes"]
    m = pope_metrics(preds, golds)
    tp, fp, fn = 2, 1, 1
    p, r = tp / (tp + fp), tp / (tp + fn)
    assert abs(m["precision"] - p) < 1e-9
    assert abs(m["recall"] - r) < 1e-9
    assert abs(m["f1"] - 2 * p * r / (p + r)) < 1e-9
    assert abs(m["accuracy"] - 3 / 5) < 1e-9
    assert abs(m["yes_ratio"] - 3 / 5) < 1e-9


def test_pope_metrics_all_no_predictions():
    m = pope_metrics(["no", "no"], ["yes", "no"])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0 and m["accuracy"] == 0.5


# ---- multiple choice -----------------------------------------------------------------------
def test_parse_choice():
    opts = ["Maryland", "New Hampshire", "Rhode Island", "Vermont"]
    assert parse_choice("B", opts) == 1
    assert parse_choice("(C)", opts) == 2
    assert parse_choice("D. Vermont", opts) == 3
    assert parse_choice("The answer is B.", opts) == 1
    assert parse_choice("Rhode Island", opts) == 2
    assert parse_choice("It is New Hampshire, I think", opts) == 1
    assert parse_choice("E", opts) == -1
    assert parse_choice("Bananas", opts) == -1  # 'B' followed by letters is not a choice letter


def test_sqa_prompt():
    p = sqa_prompt("Which colony?", ["Maryland", "Vermont"], hint="")
    assert "Context:" not in p and "A. Maryland" in p and "B. Vermont" in p and p.endswith("directly.")
    assert sqa_prompt("q", ["x"], hint="some hint").startswith("Context: some hint")


# ---- MME ---------------------------------------------------------------------------------------
def test_mme_scores_pairing():
    rec = [
        {"pair_id": "OCR/1.jpg", "category": "OCR", "correct": True},
        {"pair_id": "OCR/1.jpg", "category": "OCR", "correct": True},      # full pair correct
        {"pair_id": "OCR/2.jpg", "category": "OCR", "correct": True},
        {"pair_id": "OCR/2.jpg", "category": "OCR", "correct": False},     # half
        {"pair_id": "code_reasoning/1.png", "category": "code_reasoning", "correct": False},
        {"pair_id": "code_reasoning/1.png", "category": "code_reasoning", "correct": False},
        {"pair_id": "count/9.jpg", "category": "count", "correct": True},  # incomplete pair (n-cut)
    ]
    m = mme_scores(rec)
    ocr = m["categories"]["OCR"]
    assert ocr["n"] == 4 and ocr["pairs"] == 2
    assert abs(ocr["acc"] - 75.0) < 1e-9 and abs(ocr["acc_plus"] - 50.0) < 1e-9 and abs(ocr["score"] - 125.0) < 1e-9
    cnt = m["categories"]["count"]
    assert cnt["pairs"] == 0 and cnt["acc"] == 100.0 and cnt["acc_plus"] == 0.0
    assert m["perception_categories"] == ["count", "OCR"]      # official ordering, only categories seen
    assert m["cognition_categories"] == ["code_reasoning"]
    assert abs(m["perception_score"] - (125.0 + 100.0)) < 1e-9
    assert m["cognition_score"] == 0.0
    assert m["perception_max"] == 400 and m["cognition_max"] == 200


# ---- glue ----------------------------------------------------------------------------------------
def test_apply_overrides():
    s = apply_overrides(BENCHMARKS, ["pope.cols.answer=label", "gqa.dataset=lmms-lab-encoder/GQA", "mme.spread=false", "sqa.config=none"])
    assert s["pope"]["cols"]["answer"] == "label" and BENCHMARKS["pope"]["cols"]["answer"] == "answer"
    assert s["gqa"]["dataset"] == "lmms-lab-encoder/GQA"
    assert s["mme"]["spread"] is False and s["sqa"]["config"] is None


def test_markdown_row():
    header, row = markdown_row("x", {"pope": pope_metrics(["yes"], ["yes"]),
                                     "mme": mme_scores([{"pair_id": "a", "category": "OCR", "correct": True}])})
    assert header.count("|") == row.count("|")
    assert row.startswith("| x | 100.0 / 100.0 | - | - | - | 100/200 / 0/0 |")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} tests passed")
