import os
import re
import csv

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import PredefinedSplit, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# =========================================================
# FILE FOR LOGGING
# =========================================================
os.makedirs("results", exist_ok=True)
LOG_FILE = os.path.join("results", "ml_baselines_tfidf.txt")

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("===== TF-IDF ML BASELINE RESULTS =====\n\n")


def log(text):
    print(text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# =========================================================
# CONFUSION MATRIX HELPER
# =========================================================
def save_confusion_matrix(y_true, y_pred, labels, model_key, task_name):
    """Save the confusion matrix as a PNG image and a CSV table."""
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    short_labels = [l[:20] + "…" if len(l) > 20 else l for l in labels]
    base = os.path.join(
        "results",
        f"cm_{model_key.replace(' ', '_')}_{task_name}",
    )

    fig_size = max(8, len(labels) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short_labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", va="top", rotation_mode="anchor"
    )

    ax.set_title(
        f"{model_key} — {task_name}\nConfusion Matrix (Test Set)", fontsize=13, pad=15
    )
    plt.tight_layout()

    png_name = f"{base}.png"
    fig.savefig(png_name, dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_name = f"{base}.csv"
    with open(csv_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true \\ pred"] + labels)
        for label, row in zip(labels, cm):
            writer.writerow([label] + row.tolist())

    log(f"  → Saved: {png_name}, {csv_name}")
    return png_name


# =========================================================
# STEP 0: LOAD + CLEAN DATA  (same pipeline as model_training.py)
# =========================================================
log("[STEP 0] Loading dataset...")

df = pd.read_csv("BanglaRhet.csv")

df = df[df["rhetorical_technique"] != "Parallelism"]
df["rhetorical_technique"] = df["rhetorical_technique"].fillna("None")
df["persuasion_technique"] = df["persuasion_technique"].fillna("None")

deletion_rules = [
    ("Contrast (Juxtaposition)", "Blame Assignment", 2000),
    ("Emotional Language", "Blame Assignment", 1571),
    ("Emotional Language", "Call to Action", 447),
    ("Exaggeration/Hyperbole", "Blame Assignment", 537),
    ("Repetition", "Blame Assignment", 375),
    ("Repetition", "Call to Action", 1100),
    ("None", "Call to Action", 300),
    ("None", "Blame Assignment", 685),
    ("None", "Unity Call", 300),
    ("Specific Example (Anecdote)", "Unity Call", 272),
]

for rhet, persu, n_del in deletion_rules:
    mask = (df["rhetorical_technique"] == rhet) & (df["persuasion_technique"] == persu)
    n_sample = min(n_del, mask.sum())
    df = df.drop(df[mask].sample(n=n_sample, random_state=42).index)

log(f"Dataset size after pruning: {len(df)}\n")


# =========================================================
# STEP 1: SPLITS + LABELS
# =========================================================
df = df.rename(
    columns={
        "segment_text": "text",
        "rhetorical_technique": "y1",
        "persuasion_technique": "y2",
    }
)

df["text"] = df["text"].apply(lambda t: re.sub(r"\s+", " ", str(t).strip()))

train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["y1"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["y1"], random_state=42
)

le_y1 = LabelEncoder().fit(train_df["y1"])
le_y2 = LabelEncoder().fit(train_df["y2"])

for d in [train_df, val_df, test_df]:
    d["y1_enc"] = le_y1.transform(d["y1"])
    d["y2_enc"] = le_y2.transform(d["y2"])

log(f"Train size: {len(train_df)}")
log(f"Val size:   {len(val_df)}")
log(f"Test size:  {len(test_df)}\n")


# =========================================================
# HYPERPARAMETER SEARCH
# Train+val with PredefinedSplit: score on val, never on test.
# =========================================================
tfidf = TfidfVectorizer(analyzer="word", sublinear_tf=True)

param_grids = {
    "TF-IDF + LR": {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 3],
        "tfidf__max_features": [30000, 50000, 100000],
        "clf__C": [0.1, 1.0, 10.0],
    },
    "TF-IDF + SVM": {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 3],
        "tfidf__max_features": [30000, 50000, 100000],
        "clf__C": [0.1, 1.0, 10.0],
    },
}

pipelines = {
    "TF-IDF + LR": Pipeline(
        [
            ("tfidf", tfidf),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    ),
    "TF-IDF + SVM": Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="word", sublinear_tf=True)),
            (
                "clf",
                LinearSVC(
                    class_weight="balanced",
                    max_iter=8000,
                    random_state=42,
                    dual="auto",
                ),
            ),
        ]
    ),
}

# -1 = train fold (fit only), 0 = validation fold (score only)
search_texts = pd.concat([train_df["text"], val_df["text"]], ignore_index=True)
test_fold = np.concatenate(
    [
        np.full(len(train_df), -1, dtype=int),
        np.zeros(len(val_df), dtype=int),
    ]
)
cv = PredefinedSplit(test_fold)

tasks = {
    "y1_enc": ("Rhetorical", le_y1),
    "y2_enc": ("Persuasion", le_y2),
}

results = {}


def log_search_table(search):
    rows = []
    for params, score, rank in zip(
        search.cv_results_["params"],
        search.cv_results_["mean_test_score"],
        search.cv_results_["rank_test_score"],
    ):
        rows.append((rank, score, params))
    rows.sort(key=lambda r: r[0])

    log("\nHyperparameter search (ranked by val macro-F1):")
    log(f"{'Rank':>4}  {'Val F1':>8}  Params")
    log("-" * 80)
    for rank, score, params in rows:
        compact = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in params.items())
        log(f"{rank:4d}  {score:8.4f}  {compact}")


# =========================================================
# TRAIN LOOP
# =========================================================
log(
    "[STEP 2] Grid-searching TF-IDF + classifier hyperparameters on the validation set...\n"
)

for model_key, pipe in pipelines.items():
    log(f"\n\n===== MODEL: {model_key} =====")
    results[model_key] = {}

    for task, (task_name, le) in tasks.items():
        log(f"\n--- TASK: {task_name} ({task}) ---")

        y_search = pd.concat([train_df[task], val_df[task]], ignore_index=True).values
        y_val = val_df[task].values
        y_test = test_df[task].values

        search = GridSearchCV(
            estimator=clone(pipe),
            param_grid=param_grids[model_key],
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=False,
            verbose=1,
        )
        search.fit(search_texts, y_search)

        log(f"\nBest params: {search.best_params_}")
        log(f"Best val macro-F1 (GridSearchCV): {search.best_score_:.4f}")
        log_search_table(search)

        # Refit the winning config on train only (same protocol as the transformers).
        best = clone(pipe).set_params(**search.best_params_)
        best.fit(train_df["text"], train_df[task].values)
        val_pred = best.predict(val_df["text"])
        test_pred = best.predict(test_df["text"])

        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average="macro")

        vocab_size = len(best.named_steps["tfidf"].vocabulary_)
        log(f"\nBest TF-IDF vocabulary size: {vocab_size}")
        log(f"VAL  RESULT → Acc={val_acc:.4f}, F1={val_f1:.4f}")
        log(f"TEST RESULT → Acc={test_acc:.4f}, F1={test_f1:.4f}")

        log("\nClassification report (test):")
        log(
            classification_report(
                y_test,
                test_pred,
                target_names=list(le.classes_),
                digits=4,
                zero_division=0,
            ).rstrip()
        )

        results[model_key][task] = {
            "eval_accuracy": test_acc,
            "eval_f1_macro": test_f1,
            "val_accuracy": val_acc,
            "val_f1_macro": val_f1,
            "best_params": search.best_params_,
        }

        save_confusion_matrix(
            y_true=y_test,
            y_pred=test_pred,
            labels=le.classes_,
            model_key=model_key,
            task_name=task_name,
        )


# =========================================================
# FINAL SUMMARY
# =========================================================
log("\n\n===== FINAL SUMMARY =====")
for m in results:
    y1 = results[m]["y1_enc"]
    y2 = results[m]["y2_enc"]
    log(
        f"{m} | "
        f"Rhetorical: Acc={y1['eval_accuracy']:.4f}, F1={y1['eval_f1_macro']:.4f} || "
        f"Persuasion: Acc={y2['eval_accuracy']:.4f}, F1={y2['eval_f1_macro']:.4f}"
    )
    log(f"  Rhetorical best params: {y1['best_params']}")
    log(f"  Persuasion best params: {y2['best_params']}")

log(f"\nFull log written to: {LOG_FILE}")
