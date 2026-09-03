# SPDX-FileCopyrightText: Copyright (c) 2026 Sheba Medical Center. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This module does not embed third-party data download URLs.
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from gfmbench_api.tasks.base.base_gfm_model import BaseGFMModel


ProbeType = Union[LogisticRegression, RandomForestClassifier, Any]


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------


def _unwrap_repr(value: Any) -> Any:
    """
    Handle base-model APIs that may return either:
      - embedding representation directly
      - tuple/list where the representation is the last element

    This makes the probe code tolerant of both styles.
    """
    if isinstance(value, (tuple, list)):
        return value[-1]
    return value


def _to_numpy(
    value: Any,
    *,
    dtype: Optional[np.dtype] = np.float32,
) -> np.ndarray:
    """
    Convert a tensor or array-like object to a NumPy array.

    GPU tensors are moved to CPU exactly once here.
    """
    value = _unwrap_repr(value)

    if isinstance(value, torch.Tensor):
        if dtype is None:
            return value.detach().cpu().numpy()

        torch_dtype = torch.float32 if dtype == np.float32 else None

        if torch_dtype is not None and value.dtype != torch_dtype:
            return value.detach().to(dtype=torch_dtype, device="cpu").numpy()

        return value.detach().cpu().numpy()

    arr = np.asarray(value)

    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)

    return arr


def _concat_embeddings_to_numpy(
    left: Any,
    right: Optional[Any] = None,
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Convert one or two embedding tensors/arrays to NumPy.

    Optimization:
    If both embeddings are torch tensors, concatenate on the tensor side first,
    then transfer to CPU once.

    This avoids:
      - GPU -> CPU transfer for left
      - GPU -> CPU transfer for right
      - separate NumPy concatenation allocation

    Instead:
      - torch.cat on GPU or current tensor device
      - one GPU -> CPU transfer
    """
    left = _unwrap_repr(left)

    if right is None:
        return _to_numpy(left, dtype=dtype)

    right = _unwrap_repr(right)

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        combined = torch.cat([left, right], dim=1)

        if dtype == np.float32 and combined.dtype != torch.float32:
            combined = combined.to(dtype=torch.float32)

        return combined.detach().cpu().numpy()

    left_np = _to_numpy(left, dtype=dtype)
    right_np = _to_numpy(right, dtype=dtype)

    return np.concatenate([left_np, right_np], axis=1)


def _labels_to_numpy(labels: Any) -> np.ndarray:
    """
    Convert labels to a flat CPU NumPy array.
    """
    if isinstance(labels, torch.Tensor):
        return labels.view(-1).detach().cpu().numpy()

    return np.asarray(labels).reshape(-1)


def _try_get_dataset_len(loader: Any) -> Optional[int]:
    """
    Return len(loader.dataset) when available.

    Some iterable datasets do not expose __len__, in which case this returns None.
    """
    dataset = getattr(loader, "dataset", None)

    if dataset is None:
        return None

    try:
        return len(dataset)
    except TypeError:
        return None


def _build_probe(
    probe_type: str,
    probe_kwargs: Optional[Dict[str, Any]] = None,
) -> ProbeType:
    """
    Build a supported sklearn/xgboost classifier probe.

    Supported probe_type values:
      - "logistic_regression"
      - "random_forest"
      - "xgboost"
    """
    probe_kwargs = dict(probe_kwargs or {})

    if probe_type == "logistic_regression":
        default_kwargs = {
            "max_iter": 5000,
        }
        default_kwargs.update(probe_kwargs)
        return LogisticRegression(**default_kwargs)

    if probe_type == "random_forest":
        default_kwargs = {
            "n_estimators": 200,
            "random_state": 42,
            "n_jobs": -1,
        }
        default_kwargs.update(probe_kwargs)
        return RandomForestClassifier(**default_kwargs)

    if probe_type == "xgboost":
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed. Install it to use probe_type='xgboost'."
            )

        default_kwargs = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": 42,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }

        # Optional GPU example for recent XGBoost:
        # probe_kwargs = {
        #     "tree_method": "hist",
        #     "device": "cuda",
        # }
        #
        # For older XGBoost versions, this may be:
        # probe_kwargs = {
        #     "tree_method": "gpu_hist",
        # }

        default_kwargs.update(probe_kwargs)
        return XGBClassifier(**default_kwargs)

    raise ValueError(
        f"Unsupported probe_type={probe_type!r}. "
        "Choose from {'logistic_regression', 'random_forest', 'xgboost'}."
    )


# ---------------------------------------------------------------------
# Base wrapper
# ---------------------------------------------------------------------


class _BaseProbeWrapper(BaseGFMModel):
    """
    Shared behavior for probe-wrapped GLM models.
    """

    def __init__(self, base_model: BaseGFMModel, probe: Optional[ProbeType]) -> None:
        self.base_model = base_model
        self.device = base_model.device
        self.probe = probe

    def eval(self):
        self.base_model.eval()
        return self

    def train(self, mode: bool = True):
        self.base_model.train(mode)
        return self

    def to(self, device: str):
        self.base_model.to(device)
        self.device = device
        return self

    def get_hidden_dim(self):
        return self.base_model.get_hidden_dim()

    def infer_sequence_to_sequence(
        self,
        sequences: List[str],
        conditional_input=None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        raise NotImplementedError()

    def sequence_pos_to_prob_pos(self, sequences: List[str], pos: int) -> np.ndarray:
        raise NotImplementedError()

    def infer_masked_sequence_to_token_probs(
        self,
        sequences: List[str],
        variant_pos: int,
        variant_letters: List[str],
        reference_letters: List[str],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        raise NotImplementedError()


# ---------------------------------------------------------------------
# SNV wrapper: variant/reference sequence classification
# ---------------------------------------------------------------------


class SNVEmbeddingProbeModel(_BaseProbeWrapper):
    """
    Wrapper combining a BaseGFMModel with a classifier probe over
    concatenated variant/reference embeddings.

    The benchmark treats this as one unified model.
    """

    def infer_variant_ref_sequences_to_labels_probs(
        self,
        variant_sequences: List[str],
        ref_sequences: List[str],
        conditional_input=None,
    ) -> Optional[np.ndarray]:
        """
        Infer label probabilities from variant/reference sequence pairs.

        Embeddings are concatenated before being passed to the fitted probe.
        """
        if self.probe is None:
            return None

        with torch.inference_mode():
            _, _, var_repr = self.base_model.infer_sequence_to_sequence(
                variant_sequences,
                conditional_input,
            )
            _, _, ref_repr = self.base_model.infer_sequence_to_sequence(
                ref_sequences,
                conditional_input,
            )

            X = _concat_embeddings_to_numpy(var_repr, ref_repr)

        return self.probe.predict_proba(X)

    def infer_sequence_to_labels_probs(
        self,
        sequences: List[str],
        conditional_input=None,
    ) -> Optional[np.ndarray]:
        raise NotImplementedError(
            "SNVEmbeddingProbeModel expects variant/ref sequence pairs. "
            "Use infer_variant_ref_sequences_to_labels_probs()."
        )


# ---------------------------------------------------------------------
# Sequence wrapper: single-sequence classification
# ---------------------------------------------------------------------


class SequenceEmbeddingProbeModel(_BaseProbeWrapper):
    """
    Wrapper combining a BaseGFMModel with a classifier probe over
    single-sequence embeddings.

    The benchmark treats this as one unified model.
    """

    def infer_sequence_to_labels_probs(
        self,
        sequences: List[str],
        conditional_input=None,
    ) -> Optional[np.ndarray]:
        """
        Infer label probabilities from single-sequence embeddings.
        """
        if self.probe is None:
            return None

        with torch.inference_mode():
            _, _, seq_repr = self.base_model.infer_sequence_to_sequence(
                sequences,
                conditional_input,
            )

            X = _concat_embeddings_to_numpy(seq_repr)

        return self.probe.predict_proba(X)

    def infer_variant_ref_sequences_to_labels_probs(
        self,
        variant_sequences: List[str],
        ref_sequences: List[str],
        conditional_input=None,
    ) -> Optional[np.ndarray]:
        raise NotImplementedError(
            "SequenceEmbeddingProbeModel expects single sequences. "
            "Use infer_sequence_to_labels_probs()."
        )


# ---------------------------------------------------------------------
# Base fitter
# ---------------------------------------------------------------------


class _BaseEmbeddingProbeFitter:
    """
    Shared fitter logic for embedding probe classifiers.

    Subclasses implement _extract_embeddings().
    """

    wrapper_cls = None
    fit_message = "Fitting probe..."

    def __init__(
        self,
        model: BaseGFMModel,
        train_loader: Any,
        device: str = "cpu",
        probe_type: str = "logistic_regression",
        probe_kwargs: Optional[Dict[str, Any]] = None,
        embedding_dtype: np.dtype = np.float32,
        preallocate: bool = True,
    ) -> None:
        """
        Args:
            model:
                BaseGFMModel instance.

            train_loader:
                DataLoader yielding training batches.

            device:
                Device used by the base model for embedding extraction.

            probe_type:
                One of {"logistic_regression", "random_forest", "xgboost"}.

            probe_kwargs:
                Optional kwargs passed to the classifier constructor.

            embedding_dtype:
                NumPy dtype for extracted embeddings.
                np.float32 is usually preferred for memory efficiency.

            preallocate:
                If True and len(train_loader.dataset) is available, preallocate
                X/y arrays to avoid a large final np.concatenate memory spike.
        """
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.probe_type = probe_type
        self.probe_kwargs = probe_kwargs or {}
        self.embedding_dtype = embedding_dtype
        self.preallocate = preallocate

    def _extract_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def fit(self):
        X_train, y_train = self._extract_embeddings()
        probe = _build_probe(self.probe_type, self.probe_kwargs)

        print(
            f"{self.fit_message} "
            f"X shape={X_train.shape}, y shape={y_train.shape}, "
            f"X dtype={X_train.dtype}"
        )

        probe.fit(X_train, y_train)

        return self.wrapper_cls(self.model, probe)


# ---------------------------------------------------------------------
# SNV fitter
# ---------------------------------------------------------------------


class SNVEmbeddingProbeFitter(_BaseEmbeddingProbeFitter):
    """
    Fit a classifier probe on concatenated variant/reference embeddings.

    Expected train_loader batch format:
        variant_sequences, ref_sequences, labels, conditional_input
    """

    wrapper_cls = SNVEmbeddingProbeModel
    fit_message = "Fitting probe on combined variant/reference embeddings..."

    def _extract_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        self.model.to(self.device)

        dataset_len = _try_get_dataset_len(self.train_loader)

        if self.preallocate and dataset_len is not None:
            return self._extract_embeddings_preallocated(dataset_len)

        return self._extract_embeddings_chunked()

    def _extract_embeddings_chunked(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback extraction method.

        This is compatible with iterable datasets or loaders without a known length.
        It is less memory efficient than preallocation because final np.concatenate
        creates another full-size array.
        """
        X_chunks: List[np.ndarray] = []
        y_chunks: List[np.ndarray] = []

        with torch.inference_mode():
            for variant_sequences, ref_sequences, labels, conditional_input in tqdm(
                self.train_loader,
                desc="Generating SNV embeddings",
            ):
                var_repr = self.model.infer_sequence_to_sequence_representative(
                    variant_sequences,
                    conditional_input,
                )
                ref_repr = self.model.infer_sequence_to_sequence_representative(
                    ref_sequences,
                    conditional_input,
                )

                X_batch = _concat_embeddings_to_numpy(
                    var_repr,
                    ref_repr,
                    dtype=self.embedding_dtype,
                )
                y_batch = _labels_to_numpy(labels)

                X_chunks.append(X_batch)
                y_chunks.append(y_batch)

                del var_repr, ref_repr

        X_train = np.concatenate(X_chunks, axis=0)
        y_train = np.concatenate(y_chunks, axis=0)

        return X_train, y_train

    def _extract_embeddings_preallocated(
        self,
        dataset_len: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Memory-efficient extraction method.

        Allocates final X/y arrays once and fills them batch by batch.
        """
        X_train: Optional[np.ndarray] = None
        y_train = np.empty(dataset_len, dtype=np.int64)

        offset = 0

        with torch.inference_mode():
            for variant_sequences, ref_sequences, labels, conditional_input in tqdm(
                self.train_loader,
                desc="Generating SNV embeddings",
            ):
                var_repr = self.model.infer_sequence_to_sequence_representative(
                    variant_sequences,
                    conditional_input,
                )
                ref_repr = self.model.infer_sequence_to_sequence_representative(
                    ref_sequences,
                    conditional_input,
                )

                X_batch = _concat_embeddings_to_numpy(
                    var_repr,
                    ref_repr,
                    dtype=self.embedding_dtype,
                )
                y_batch = _labels_to_numpy(labels)

                batch_size = X_batch.shape[0]

                if X_train is None:
                    X_train = np.empty(
                        (dataset_len, X_batch.shape[1]),
                        dtype=X_batch.dtype,
                    )

                X_train[offset : offset + batch_size] = X_batch
                y_train[offset : offset + batch_size] = y_batch

                offset += batch_size

                del var_repr, ref_repr, X_batch, y_batch

        if X_train is None:
            raise ValueError("No embeddings were extracted. Is the train_loader empty?")

        return X_train[:offset], y_train[:offset]


# ---------------------------------------------------------------------
# Single-sequence fitter
# ---------------------------------------------------------------------


class SequenceEmbeddingProbeFitter(_BaseEmbeddingProbeFitter):
    """
    Fit a classifier probe on single-sequence embeddings.

    Expected train_loader batch format:
        sequences, labels, conditional_input
    """

    wrapper_cls = SequenceEmbeddingProbeModel
    fit_message = "Fitting probe on single-sequence embeddings..."

    def _extract_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        self.model.to(self.device)

        dataset_len = _try_get_dataset_len(self.train_loader)

        if self.preallocate and dataset_len is not None:
            return self._extract_embeddings_preallocated(dataset_len)

        return self._extract_embeddings_chunked()

    def _extract_embeddings_chunked(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback extraction method for unknown dataset lengths.
        """
        X_chunks: List[np.ndarray] = []
        y_chunks: List[np.ndarray] = []

        with torch.inference_mode():
            for sequences, labels, conditional_input in tqdm(
                self.train_loader,
                desc="Generating sequence embeddings",
            ):
                seq_repr = self.model.infer_sequence_to_sequence_representative(
                    sequences,
                    conditional_input,
                )

                X_batch = _concat_embeddings_to_numpy(
                    seq_repr,
                    dtype=self.embedding_dtype,
                )
                y_batch = _labels_to_numpy(labels)

                X_chunks.append(X_batch)
                y_chunks.append(y_batch)

                del seq_repr

        X_train = np.concatenate(X_chunks, axis=0)
        y_train = np.concatenate(y_chunks, axis=0)

        return X_train, y_train

    def _extract_embeddings_preallocated(
        self,
        dataset_len: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Memory-efficient extraction method.

        Allocates final X/y arrays once and fills them batch by batch.
        """
        X_train: Optional[np.ndarray] = None
        y_train = np.empty(dataset_len, dtype=np.int64)

        offset = 0

        with torch.inference_mode():
            for sequences, labels, conditional_input in tqdm(
                self.train_loader,
                desc="Generating sequence embeddings",
            ):
                seq_repr = self.model.infer_sequence_to_sequence_representative(
                    sequences,
                    conditional_input,
                )

                X_batch = _concat_embeddings_to_numpy(
                    seq_repr,
                    dtype=self.embedding_dtype,
                )
                y_batch = _labels_to_numpy(labels)

                batch_size = X_batch.shape[0]

                if X_train is None:
                    X_train = np.empty(
                        (dataset_len, X_batch.shape[1]),
                        dtype=X_batch.dtype,
                    )

                X_train[offset : offset + batch_size] = X_batch
                y_train[offset : offset + batch_size] = y_batch

                offset += batch_size

                del seq_repr, X_batch, y_batch

        if X_train is None:
            raise ValueError("No embeddings were extracted. Is the train_loader empty?")

        return X_train[:offset], y_train[:offset]