"""
MLTables Ontology Module
Domain: Machine Learning / NLP Results Tables
Task: Extract quantitative entries from ML-paper tables and classify into
      {Result | Data Stat. | Hyper-parameter/Architecture | Other}.

Ground-truth format (per numeric cell): {
    value, type,
    [Result]:        model, training_dataset, test_dataset, task, metric, subset
    [Data Stat.]:    dataset, attribute_name, subset
    [Hyper-parameter/Architecture]: parameter_name, model (opt.)
    [Other]:         (skipped in scoring)
}

Derived from the SchemaDrivenIE MLTables benchmark (Bai et al., EMNLP 2024).
"""

# =============================================================================
# Cell-Type Taxonomy  (4 canonical categories, mirrors the benchmark)
# =============================================================================
TYPE_TAXONOMY = {
    # Result synonyms
    "result": "Result", "results": "Result",
    "performance": "Result", "score": "Result", "scores": "Result",
    "accuracy result": "Result", "evaluation result": "Result",

    # Data statistics synonyms
    "data stat": "Data Stat.", "data stat.": "Data Stat.",
    "data statistics": "Data Stat.", "dataset statistics": "Data Stat.",
    "statistics": "Data Stat.", "statistics of": "Data Stat.",
    "split size": "Data Stat.", "corpus size": "Data Stat.",
    "number of samples": "Data Stat.", "sample size": "Data Stat.",

    # Hyper-parameter / architecture synonyms
    "hyperparameter": "Hyper-parameter/Architecture",
    "hyper-parameter": "Hyper-parameter/Architecture",
    "hyper parameter": "Hyper-parameter/Architecture",
    "architecture": "Hyper-parameter/Architecture",
    "model config": "Hyper-parameter/Architecture",
    "training setting": "Hyper-parameter/Architecture",
    "config": "Hyper-parameter/Architecture",
    "settings": "Hyper-parameter/Architecture",

    # Other
    "other": "Other", "misc": "Other", "miscellaneous": "Other",
}

VALID_CELL_TYPES = {"Result", "Data Stat.", "Hyper-parameter/Architecture", "Other"}

# =============================================================================
# Metric Taxonomy
# Canonicalises surface variants of evaluation metrics.
# (Left-hand key lowercased+space-stripped; right-hand is canonical form
#  as it appears most frequently in the gold.)
# =============================================================================
METRIC_TAXONOMY = {
    # Accuracy family
    "accuracy": "accuracy", "acc": "accuracy", "acc.": "accuracy",
    "avg accuracy": "Avg. accuracy", "avg. accuracy": "Avg. accuracy",
    "average accuracy": "Average accuracy", "mean accuracy": "Avg. accuracy",
    "top-1 accuracy": "accuracy", "top1 accuracy": "accuracy",
    "top-5 accuracy": "Top-5 accuracy", "top5 accuracy": "Top-5 accuracy",
    "lexical accuracy": "lexical accuracy",

    # Translation / generation
    "bleu": "BLEU", "sacrebleu": "BLEU",
    "delta bleu": "delta BLEU", "δ bleu": "delta BLEU",
    "nist": "NIST", "rouge": "ROUGE", "rouge-l": "ROUGE-L",
    "meteor": "METEOR", "chrf": "chrF", "chrf++": "chrF++",
    "ter": "TER", "bertscore": "BERTScore",

    # Detection / segmentation
    "ap": "AP",
    "ap50": "\\text{AP}^{50}", "ap^50": "\\text{AP}^{50}",
    "ap75": "\\text{AP}^{75}", "ap^75": "\\text{AP}^{75}",
    "ap^m": "\\text{AP}^{\\text{M}}", "ap^l": "\\text{AP}^{\\text{L}}",
    "ar": "AR", "map": "mAP", "m ap": "mAP",
    "miou": "mIOU", "mean iou": "mIOU",
    "iou": "IoU", "dice": "Dice",

    # Retrieval
    "recall@1": "Recall@1", "r@1": "Recall@1",
    "recall@5": "Recall@5", "r@5": "Recall@5",
    "recall@10": "Recall@10", "r@10": "Recall@10",
    "precision@1": "P@1", "p@1": "P@1",
    "mrr": "MRR", "ndcg": "NDCG", "ndcg@10": "NDCG@10",

    # Error / loss
    "mae": "MAE", "mse": "MSE", "rmse": "RMSE",
    "mnll": "MNLL", "nll": "NLL", "ppl": "perplexity",
    "perplexity": "perplexity",
    "epe": "end-point-error", "end point error": "end-point-error",
    "end-point-error": "end-point-error",

    # Misc
    "coverage": "Cov.", "cov.": "Cov.", "cov": "Cov.",
    "error": "Err.", "err.": "Err.", "err": "Err.",
    "f1": "F1", "f-1": "F1", "macro-f1": "macro-F1", "micro-f1": "micro-F1",
    "precision": "Precision", "recall": "Recall",
    "auc": "AUC", "auroc": "AUROC", "aupr": "AUPR",

    # Task-specific
    "sbwt": "semantic backward transfer",
    "semantic backward transfer": "semantic backward transfer",
}

# =============================================================================
# Task Taxonomy
# Common ML task names, with synonym merging.
# =============================================================================
TASK_TAXONOMY = {
    # Vision
    "image classification": "image classification",
    "image recognition": "image recognition",
    "object detection": "object detection",
    "semantic segmentation": "semantic segmentation",
    "instance segmentation": "instance segmentation",
    "depth prediction": "Depth Prediction",
    "depth estimation": "Depth Estimation",
    "optical flow": "Optical Flow",
    "human pose estimation": "human pose estimation",
    "2d human pose estimation": "2D human pose estimation",
    "pose estimation": "human pose estimation",

    # NLP
    "machine translation": "Machine Translation",
    "mt": "Machine Translation",
    "neural machine translation": "Machine Translation",
    "nmt": "Machine Translation",
    "few-shot region-aware machine translation": "Few-Shot Region-Aware Machine Translation",
    "frmt": "FRMT",
    "language modeling": "language modeling",
    "text classification": "text classification",
    "sentiment classification": "sentiment classification",
    "hate speech detection": "zero-shot hate speech detection",
    "definition generation": "Definition Generation",

    # Multimodal
    "vqa": "VQA",
    "visual question answering": "visual question answering",
    "image-speech retrieval": "image-speech retrieval",
    "image-text retrieval": "image-text retrieval",

    # Classification / FS
    "few-shot classification": "Few-shot classification",
    "selective classification": "Selective classification",
    "few-shot graph classification": "Few-shot graph classification",
    "graph classification": "graph classification",

    # Time-series
    "time-to-event modeling": "time-to-event modeling",
    "event modeling": "event modeling",
}

# =============================================================================
# Data-Statistic Attribute Names
# Maps surface forms to canonical "attribute name" values used in Data Stat. rows.
# =============================================================================
ATTR_TAXONOMY = {
    "number of samples": "number of samples",
    "#samples": "number of samples", "# samples": "number of samples",
    "n samples": "number of samples", "num samples": "number of samples",

    "number of sentence pairs": "Number of sentence pairs",
    "# sentence pairs": "# sentence pairs",
    "sentence pairs": "Number of sentence pairs",

    "number of examples": "number of examples",
    "# examples": "# examples", "#examples": "number of examples",

    "number of classes": "number of classes",
    "# classes": "number of classes", "num classes": "number of classes",

    "number of graphs": "number of Graph",
    "# graph": "# Graph",    "graph #": "Graph #",
    "number of nodes":  "number of Node",  "# node":  "# Node",
    "number of edges":  "number of Edge",  "# edge":  "# Edge",
    "number of features": "number of Feat.", "# feat.": "# Feat.",
    "number of labels": "number of Label", "# label": "# Label",

    "vocabulary size": "vocabulary size",
    "corpus size": "corpus size",
    "avg length": "average length",
    "average length": "average length",
    "max length": "max length",
}

# =============================================================================
# Hyper-parameter / Architecture Name Taxonomy
# =============================================================================
PARAM_TAXONOMY = {
    # Training
    "learning rate": "Learning rate", "lr": "Learning rate",
    "batch size": "Batch size", "bs": "Batch size",
    "epochs": "Training epochs", "training epochs": "Training epochs",
    "max-epoch": "Max-epoch", "max epoch": "Max-epoch",
    "early-stop": "Early-stop", "early stopping": "Early-stop",
    "dropout": "Dropout rate", "dropout rate": "Dropout rate",
    "optimizer": "Optimizer", "weight decay": "Weight decay",
    "momentum": "Momentum", "warmup": "Warmup steps",

    # Architecture
    "hidden size": "Hidden size", "hidden dim": "Hidden size",
    "num layers": "Number of layers", "layers": "Number of layers",
    "num heads": "Number of heads", "attention heads": "Number of heads",
    "embedding dim": "Embedding dim", "embedding size": "Embedding dim",
    "input resolution": "Input Resolution (pixels)",
    "image size": "Input Resolution (pixels)",

    # Capacity
    "params": "Params", "parameters": "number of parameters",
    "param (m)": "Param (M)", "number of parameters": "number of parameters",
    "model size": "model size",
    "total parameters": "Total parameters (M)",
    "inference per second": "Inference Per second",
    "fps": "Inference Per second",

    # Regularisation weights
    "lambda": "\\lambda", "λ": "\\lambda",
    "alpha": "\\alpha", "beta": "\\beta", "gamma": "\\gamma",
    "temperature": "temperature",
    "top-k": "top-k", "top-p": "top-p",
}

# =============================================================================
# Dataset / Benchmark Name Taxonomy (partial — seed list, extends from gold)
# =============================================================================
DATASET_TAXONOMY = {
    # Vision
    "vqa-v2": "VQA-v2", "vqav2": "VQA-v2",
    "coco": "COCO", "mscoco": "MS-COCO", "ms-coco": "MS-COCO",
    "imagenet": "ImageNet", "imagenet-1k": "ImageNet-1k",
    "cifar10": "CIFAR-10", "cifar-10": "CIFAR-10",
    "cifar100": "CIFAR-100", "cifar-100": "CIFAR-100",
    "pascal voc": "Pascal VOC", "voc": "Pascal VOC",
    "cityscapes": "Cityscapes", "ade20k": "ADE20K",
    "kitti": "KITTI", "nyu": "NYU-v2", "nyuv2": "NYU-v2",
    "mpii": "MPII", "coco keypoint": "COCO Keypoint",

    # NLP
    "wmt": "WMT", "wmt14": "WMT14", "wmt16": "WMT16", "wmt17": "WMT17",
    "glue": "GLUE", "superglue": "SuperGLUE",
    "squad": "SQuAD", "squad v1": "SQuAD-v1", "squad v2": "SQuAD-v2",
    "snli": "SNLI", "multinli": "MultiNLI", "mnli": "MultiNLI",
    "penn treebank": "Penn Treebank", "ptb": "Penn Treebank",
    "wikitext-103": "WikiText-103",
}

# =============================================================================
# Model / Architecture Name Taxonomy
# =============================================================================
MODEL_TAXONOMY = {
    # Transformers
    "transformer": "Transformer",
    "bert": "BERT", "bert-base": "BERT-base", "bert-large": "BERT-large",
    "roberta": "RoBERTa", "gpt-2": "GPT-2", "gpt2": "GPT-2",
    "gpt-3": "GPT-3", "t5": "T5", "bart": "BART",
    "xlnet": "XLNet", "electra": "ELECTRA", "albert": "ALBERT",

    # Vision
    "resnet": "ResNet", "resnet-50": "ResNet-50",
    "resnet-101": "ResNet-101", "resnet50": "ResNet-50",
    "vit": "ViT", "vit-b": "ViT-B", "vit-l": "ViT-L",
    "swin": "Swin", "swin-b": "Swin-B",
    "efficientnet": "EfficientNet",
    "yolo": "YOLO", "yolov5": "YOLOv5",
    "mask r-cnn": "Mask R-CNN", "faster r-cnn": "Faster R-CNN",

    # Generic
    "lstm": "LSTM", "bilstm": "BiLSTM", "gru": "GRU", "rnn": "RNN",
    "cnn": "CNN", "mlp": "MLP", "gcn": "GCN", "gat": "GAT",
    "gnn": "GNN", "graphsage": "GraphSAGE",
}

# =============================================================================
# Semantic Constraints
# =============================================================================
CONSTRAINTS = {
    # Null semantics — ML tables sometimes use these for "not reported"
    "null_conventions": {
        "not_reported": ["-", "--", "N/A", "n/a", "", "—", "–"],
    },

    # Value plausibility per metric (coarse sanity check)
    "plausibility": {
        "accuracy":   {"min": 0.0,    "max": 100.0},
        "Accuracy":   {"min": 0.0,    "max": 100.0},
        "BLEU":       {"min": 0.0,    "max": 100.0},
        "AP":         {"min": 0.0,    "max": 100.0},
        "mIOU":       {"min": 0.0,    "max": 100.0},
        "Recall@1":   {"min": 0.0,    "max": 100.0},
        "F1":         {"min": 0.0,    "max": 100.0},
        "MAE":        {"min": 0.0,    "max": 1e6},
        "MSE":        {"min": 0.0,    "max": 1e6},
        "perplexity": {"min": 1.0,    "max": 1e6},
        "Learning rate": {"min": 1e-8, "max": 1.0},
        "Batch size":    {"min": 1,    "max": 16384},
    },

    # Type→required-fields (used at extraction time to enforce schema)
    "type_required_fields": {
        "Result":                         ["model", "test_dataset", "metric"],
        "Data Stat.":                     ["dataset", "attribute_name"],
        "Hyper-parameter/Architecture":   ["parameter_name"],
        "Other":                          [],   # never scored
    },

    # Cells to skip entirely during extraction (obvious non-entries)
    "skip_patterns": [
        "table", "figure", "fig.", "eq.", "equation",
    ],
}


# =============================================================================
# Convenience functions (mirrors ChemTables / DiSCoMaT modules)
# =============================================================================
def _norm(s: str) -> str:
    return (s or "").strip().lower().replace("  ", " ")


def classify_type(header_text: str) -> str | None:
    """Best-effort mapping of a header/caption token to canonical cell-type."""
    t = _norm(header_text)
    for pat, canon in TYPE_TAXONOMY.items():
        if pat in t:
            return canon
    return None


def standardize_metric(raw: str) -> str:
    return METRIC_TAXONOMY.get(_norm(raw), raw.strip())


def standardize_task(raw: str) -> str:
    return TASK_TAXONOMY.get(_norm(raw), raw.strip())


def standardize_attr_name(raw: str) -> str:
    return ATTR_TAXONOMY.get(_norm(raw), raw.strip())


def standardize_param_name(raw: str) -> str:
    return PARAM_TAXONOMY.get(_norm(raw), raw.strip())


def standardize_dataset(raw: str) -> str:
    return DATASET_TAXONOMY.get(_norm(raw), raw.strip())


def standardize_model(raw: str) -> str:
    return MODEL_TAXONOMY.get(_norm(raw), raw.strip())


def is_null_value(raw: str) -> bool:
    return raw.strip() in CONSTRAINTS["null_conventions"]["not_reported"]


# =============================================================================
# Ontology metadata
# =============================================================================
ONTOLOGY_INFO = {
    "name": "MLTables Machine-Learning-Results Ontology",
    "domain": "Machine Learning / NLP evaluation tables",
    "version": "1.0",
    "total_entries": (
        len(TYPE_TAXONOMY) + len(METRIC_TAXONOMY) + len(TASK_TAXONOMY)
        + len(ATTR_TAXONOMY) + len(PARAM_TAXONOMY)
        + len(DATASET_TAXONOMY) + len(MODEL_TAXONOMY)
    ),
    "source_benchmark": "Schema-Driven IE / MLTables (Bai et al., EMNLP 2024)",
    "ground_truth_format": "{value, type, [model, training/test dataset, task, metric, subset | attribute name, dataset | parameter/architecture name]}",
    "valid_types": sorted(VALID_CELL_TYPES),
}


if __name__ == "__main__":
    print("MLTables Ontology Module")
    print(f"  Total entries : {ONTOLOGY_INFO['total_entries']}")
    print(f"  Cell types    : {len(TYPE_TAXONOMY)} → {sorted(VALID_CELL_TYPES)}")
    print(f"  Metrics       : {len(METRIC_TAXONOMY)}")
    print(f"  Tasks         : {len(TASK_TAXONOMY)}")
    print(f"  Attr names    : {len(ATTR_TAXONOMY)}")
    print(f"  Param names   : {len(PARAM_TAXONOMY)}")
    print(f"  Datasets      : {len(DATASET_TAXONOMY)}")
    print(f"  Models        : {len(MODEL_TAXONOMY)}")
