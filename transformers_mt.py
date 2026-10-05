import re
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# =========================================================
# FILE FOR LOGGING
# =========================================================
LOG_FILE = "results.txt"
with open(LOG_FILE, "w") as f:
    f.write("===== MODEL TRAINING RESULTS =====\n\n")


def log(text):
    print(text)
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n")


# =========================================================
# CONFUSION MATRIX HELPER
# =========================================================
def save_confusion_matrix(y_true, y_pred, labels, model_key, task_name):
    """Save the confusion matrix as a PNG image and a CSV table."""
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    short_labels = [l[:20] + "…" if len(l) > 20 else l for l in labels]
    base = f"cm_{model_key.replace(' ', '_')}_{task_name}"

    # ---------- PNG ----------
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

    # ---------- CSV ----------
    csv_name = f"{base}.csv"
    with open(csv_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true \\ pred"] + labels)
        for label, row in zip(labels, cm):
            writer.writerow([label] + row.tolist())

    log(f"  → Saved: {png_name}, {csv_name}")
    return png_name


# =========================================================
# STEP 0: LOAD + CLEAN DATA
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


def make_dataset(df, label):
    return Dataset.from_pandas(df[["text", label]].rename(columns={label: "label"}))


ds = {
    "y1_enc": {
        "train": make_dataset(train_df, "y1_enc"),
        "val": make_dataset(val_df, "y1_enc"),
        "test": make_dataset(test_df, "y1_enc"),
    },
    "y2_enc": {
        "train": make_dataset(train_df, "y2_enc"),
        "val": make_dataset(val_df, "y2_enc"),
        "test": make_dataset(test_df, "y2_enc"),
    },
}


# =========================================================
# MODELS
# =========================================================
model_names = {
    "Bangla-BERT-Base": "sagorsarker/bangla-bert-base",
    "BanglaBERT": "csebuetnlp/banglabert",
    "XLM-RoBERTa": "xlm-roberta-base",
    "SahajBERT": "neuropark/sahajBERT",
}

results = {}


# =========================================================
# TRAINER WITH CLASS WEIGHT
# =========================================================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# =========================================================
# TRAIN LOOP
# =========================================================
for model_key, model_id in model_names.items():
    log(f"\n\n===== MODEL: {model_key} =====")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    for task in ds:
        for split in ds[task]:
            ds[task][split] = ds[task][split].map(
                lambda b: tokenizer(
                    b["text"], truncation=True, padding="max_length", max_length=128
                ),
                batched=True,
            )

    y1_w = torch.tensor(
        compute_class_weight(
            "balanced", classes=np.unique(train_df["y1_enc"]), y=train_df["y1_enc"]
        ),
        dtype=torch.float,
    )

    y2_w = torch.tensor(
        compute_class_weight(
            "balanced", classes=np.unique(train_df["y2_enc"]), y=train_df["y2_enc"]
        ),
        dtype=torch.float,
    )

    results[model_key] = {}

    for task, weight in [("y1_enc", y1_w), ("y2_enc", y2_w)]:
        log(f"\n--- TASK: {task} ---")

        training_args = TrainingArguments(
            output_dir="./tmp",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            logging_steps=100,
            report_to="none",
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=len(le_y1.classes_) if task == "y1_enc" else len(le_y2.classes_),
            ignore_mismatched_sizes=True,
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=ds[task]["train"],
            eval_dataset=ds[task]["val"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            class_weights=weight,
        )

        trainer.train()

        # ===== EPOCH TABLE =====
        history = trainer.state.log_history
        epoch_rows = [h for h in history if "eval_f1_macro" in h]

        log("\nEpoch | Val Acc | Val F1 | Val Loss")
        log("-----------------------------------")

        best_epoch = max(epoch_rows, key=lambda x: x["eval_f1_macro"])

        for h in epoch_rows:
            mark = " <-- BEST" if h == best_epoch else ""
            log(
                f"{int(h['epoch']):5d} | {h['eval_accuracy']:.4f} | {h['eval_f1_macro']:.4f} | {h['eval_loss']:.4f}{mark}"
            )

        # ===== TEST EVALUATION =====
        test_res = trainer.evaluate(ds[task]["test"])
        log(
            f"\nTEST RESULT → Acc={test_res['eval_accuracy']:.4f}, F1={test_res['eval_f1_macro']:.4f}"
        )

        results[model_key][task] = test_res

        # ===== CONFUSION MATRIX =====
        predictions = trainer.predict(ds[task]["test"])
        y_pred = predictions.predictions.argmax(axis=1)
        y_true = predictions.label_ids

        le = le_y1 if task == "y1_enc" else le_y2
        task_name = "Rhetorical" if task == "y1_enc" else "Persuasion"

        save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
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
