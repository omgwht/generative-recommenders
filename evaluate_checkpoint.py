#!/usr/bin/env python3

# pyre-unsafe

import argparse
import inspect
import time
from collections import defaultdict
from typing import Any, Dict

import gin
import torch
import fbgemm_gpu  # noqa: F401
from generative_recommenders.research.data.eval import (
    eval_metrics_v2_from_tensors,
    get_eval_state,
)
from generative_recommenders.research.data.reco_dataset import get_reco_dataset
from generative_recommenders.research.indexing.utils import get_top_k_module
from generative_recommenders.research.modeling.sequential.autoregressive_losses import (
    LocalNegativesSampler,
)
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    EmbeddingModule,
    LocalEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.encoder_utils import (
    get_sequential_encoder,
)
from generative_recommenders.research.modeling.sequential.features import (
    movielens_seq_features_from_row,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
    LayerNormEmbeddingPostprocessor,
)
from generative_recommenders.research.modeling.similarity_utils import (
    get_similarity_function,
)
from generative_recommenders.research.trainer.train import train_fn
from torch.utils.data import DataLoader


def _strip_ddp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not all(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key[len("module.") :]: value for key, value in state_dict.items()}


def _get_train_hparam(name: str) -> Any:
    binding_name = f"train_fn.{name}"
    try:
        return gin.query_parameter(binding_name)
    except ValueError:
        signature = inspect.signature(train_fn)
        if name not in signature.parameters:
            raise ValueError(f"Unknown train_fn parameter: {name}")
        default_value = signature.parameters[name].default
        if default_value is inspect._empty:
            raise ValueError(f"No default value for required parameter {name}")
        return default_value


def _build_model_and_dataset(device: str) -> tuple[torch.nn.Module, Any, Dict[str, Any]]:
    hparams = {
        "dataset_name": _get_train_hparam("dataset_name"),
        "max_sequence_length": _get_train_hparam("max_sequence_length"),
        "eval_batch_size": _get_train_hparam("eval_batch_size"),
        "eval_user_max_batch_size": _get_train_hparam("eval_user_max_batch_size"),
        "main_module": _get_train_hparam("main_module"),
        "main_module_bf16": _get_train_hparam("main_module_bf16"),
        "dropout_rate": _get_train_hparam("dropout_rate"),
        "user_embedding_norm": _get_train_hparam("user_embedding_norm"),
        "sampling_strategy": _get_train_hparam("sampling_strategy"),
        "item_l2_norm": _get_train_hparam("item_l2_norm"),
        "top_k_method": _get_train_hparam("top_k_method"),
        "embedding_module_type": _get_train_hparam("embedding_module_type"),
        "item_embedding_dim": _get_train_hparam("item_embedding_dim"),
        "interaction_module_type": _get_train_hparam("interaction_module_type"),
        "gr_output_length": _get_train_hparam("gr_output_length"),
        "l2_norm_eps": _get_train_hparam("l2_norm_eps"),
    }

    dataset = get_reco_dataset(
        dataset_name=hparams["dataset_name"],
        max_sequence_length=hparams["max_sequence_length"],
        chronological=True,
        positional_sampling_ratio=1.0,
    )

    if hparams["embedding_module_type"] != "local":
        raise ValueError(
            f"Unsupported embedding_module_type={hparams['embedding_module_type']}"
        )

    embedding_module: EmbeddingModule = LocalEmbeddingModule(
        num_items=dataset.max_item_id,
        item_embedding_dim=hparams["item_embedding_dim"],
    )

    interaction_module, _ = get_similarity_function(
        module_type=hparams["interaction_module_type"],
        query_embedding_dim=hparams["item_embedding_dim"],
        item_embedding_dim=hparams["item_embedding_dim"],
        bf16_training=hparams["main_module_bf16"],
    )

    if hparams["user_embedding_norm"] == "l2_norm":
        output_postproc_module = L2NormEmbeddingPostprocessor(
            embedding_dim=hparams["item_embedding_dim"],
            eps=1e-6,
        )
    elif hparams["user_embedding_norm"] == "layer_norm":
        output_postproc_module = LayerNormEmbeddingPostprocessor(
            embedding_dim=hparams["item_embedding_dim"],
            eps=1e-6,
        )
    else:
        raise ValueError(
            f"Unsupported user_embedding_norm={hparams['user_embedding_norm']}"
        )

    input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=dataset.max_sequence_length + hparams["gr_output_length"] + 1,
        embedding_dim=hparams["item_embedding_dim"],
        dropout_rate=hparams["dropout_rate"],
    )

    model = get_sequential_encoder(
        module_type=hparams["main_module"],
        max_sequence_length=dataset.max_sequence_length,
        max_output_length=hparams["gr_output_length"] + 1,
        embedding_module=embedding_module,
        interaction_module=interaction_module,
        input_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        verbose=False,
    )
    if hparams["main_module_bf16"]:
        model = model.to(torch.bfloat16)
    model = model.to(device)
    model.eval()

    return model, dataset, hparams


def evaluate(
    model: torch.nn.Module,
    dataset: Any,
    hparams: Dict[str, Any],
    device: str,
    max_batches: int,
) -> Dict[str, float]:
    # Force single-process loader to avoid semaphore permission issues in locked envs.
    eval_data_loader = DataLoader(
        dataset.eval_dataset,
        batch_size=hparams["eval_batch_size"],
        shuffle=False,
        num_workers=0,
    )

    negatives_sampler = LocalNegativesSampler(
        num_items=dataset.max_item_id,
        item_emb=model._embedding_module._item_emb,  # pyre-ignore [16]
        all_item_ids=dataset.all_item_ids,
        l2_norm=hparams["item_l2_norm"],
        l2_norm_eps=hparams["l2_norm_eps"],
    ).to(device)

    eval_state = get_eval_state(
        model=model,  # pyre-ignore [6]
        all_item_ids=dataset.all_item_ids,
        negatives_sampler=negatives_sampler,
        top_k_module_fn=lambda item_embeddings, item_ids: get_top_k_module(
            top_k_method=hparams["top_k_method"],
            model=model,  # pyre-ignore [6]
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        ),
        device=device,  # pyre-ignore [6]
        float_dtype=torch.bfloat16 if hparams["main_module_bf16"] else None,
    )

    metric_sums: Dict[str, float] = defaultdict(float)
    metric_counts: Dict[str, int] = defaultdict(int)
    start_time = time.time()

    for batch_idx, row in enumerate(iter(eval_data_loader)):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
            row,
            device=device,  # pyre-ignore [6]
            max_output_length=hparams["gr_output_length"] + 1,
        )
        eval_dict = eval_metrics_v2_from_tensors(
            eval_state=eval_state,
            model=model,  # pyre-ignore [6]
            seq_features=seq_features,
            target_ids=target_ids,
            target_ratings=target_ratings,
            user_max_batch_size=hparams["eval_user_max_batch_size"],
            dtype=torch.bfloat16 if hparams["main_module_bf16"] else None,
        )
        for key, value in eval_dict.items():
            value_f = value.float()
            metric_sums[key] += float(value_f.sum().item())
            metric_counts[key] += int(value_f.numel())

        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"[eval] batches={batch_idx + 1}, elapsed={elapsed:.1f}s, "
                f"examples={metric_counts.get('hr@10', 0)}"
            )

    metrics = {
        key: metric_sums[key] / metric_counts[key]
        for key in metric_sums.keys()
        if metric_counts[key] > 0
    }
    # In this one-positive-target setup, Recall@K == HR@K.
    if "hr@10" in metrics:
        metrics["recall@10"] = metrics["hr@10"]
    if "hr@20" in metrics:
        metrics["recall@20"] = metrics["hr@20"]

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GR checkpoint without retraining."
    )
    parser.add_argument("--gin_config_file", required=True, type=str)
    parser.add_argument("--checkpoint_path", required=True, type=str)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument(
        "--max_batches",
        default=0,
        type=int,
        help="For debugging only: evaluate first N batches (0 means full eval).",
    )
    args = parser.parse_args()

    # Register gin configurables before parsing config file.
    gin.parse_config_file(args.gin_config_file)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[info] device={device}")
    model, dataset, hparams = _build_model_and_dataset(device=device)
    print(
        "[info] dataset="
        f"{hparams['dataset_name']} eval_rows={len(dataset.eval_dataset)} "
        f"batch_size={hparams['eval_batch_size']}"
    )

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = _strip_ddp_prefix(checkpoint["model_state_dict"])
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint load mismatch. missing={missing_keys}, unexpected={unexpected_keys}"
        )
    print(f"[info] loaded checkpoint epoch={checkpoint.get('epoch', 'unknown')}")

    metrics = evaluate(
        model=model,
        dataset=dataset,
        hparams=hparams,
        device=device,
        max_batches=args.max_batches,
    )

    preferred = [
        "recall@10",
        "recall@20",
        "ndcg@10",
        "ndcg@20",
        "hr@10",
        "hr@20",
        "hr@50",
        "mrr",
    ]
    print("[result] metrics")
    for key in preferred:
        if key in metrics:
            print(f"{key}: {metrics[key]:.6f}")
    for key in sorted(metrics.keys()):
        if key not in preferred:
            print(f"{key}: {metrics[key]:.6f}")


if __name__ == "__main__":
    main()
