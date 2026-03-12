from typing import Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
import os
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from utils.parameter_handling import load_parameters
from utils.log_handling import log_info, log_warn, log_error
from utils.lm_inference import (
    RateLimitedAPIBase,
    OpenAICompatibleAPIBase,
    hf_remove_from_store,
    hf_clear_store,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool token embeddings weighted by the attention mask."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def maybe_normalize(embeddings: torch.Tensor, normalize: bool) -> torch.Tensor:
    if normalize:
        return F.normalize(embeddings, p=2, dim=-1)
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base classes
# ─────────────────────────────────────────────────────────────────────────────


class TextEmbeddingModel(ABC):
    """Abstract base for models that embed text into dense vectors."""

    @abstractmethod
    def do_embed_text(self, *, texts: list[str], normalize: bool) -> torch.Tensor:
        """
        Embed a validated, non-empty list of strings.

        :param texts: Pre-validated list of strings.
        :param normalize: If True, L2-normalise each embedding.
        :return: Float tensor of shape (N, dim).
        :rtype: torch.Tensor
        """
        pass

    def embed(
        self,
        *,
        texts: Union[str, list[str]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Embed one or more texts.

        :param texts: A single string or a list of strings.
        :param normalize: If True, L2-normalise the output embeddings. Default True.
        :return: Shape (dim,) for a single string input, (N, dim) for a list.
        :rtype: torch.Tensor
        """
        passed_str = isinstance(texts, str)
        if passed_str:
            if texts.strip() == "":
                log_error("texts cannot be empty.")
            texts = [texts]
        else:
            if not isinstance(texts, list):
                log_error(
                    f"texts must be a string or list of strings. Got {type(texts)}"
                )
            if len(texts) == 0:
                log_error("texts cannot be empty.")
            for i, item in enumerate(texts):
                if not isinstance(item, str):
                    log_error(f"Got {type(item)}:{item} instead of str at texts[{i}].")
        result = self.do_embed_text(texts=texts, normalize=normalize)
        return result[0] if passed_str else result


class ImageEmbeddingModel(ABC):
    """Abstract base for models that embed images into dense vectors."""

    @abstractmethod
    def do_embed_image(
        self, *, images: list[Image.Image], normalize: bool
    ) -> torch.Tensor:
        """
        Embed a validated, non-empty list of PIL images.

        :param images: Pre-validated list of PIL Images.
        :param normalize: If True, L2-normalise each embedding.
        :return: Float tensor of shape (N, dim).
        :rtype: torch.Tensor
        """
        pass

    def embed(
        self,
        *,
        images: Union[Image.Image, list[Image.Image]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Embed one or more images.

        :param images: A single PIL Image or a list of PIL Images.
        :param normalize: If True, L2-normalise the output embeddings. Default True.
        :return: Shape (dim,) for a single image input, (N, dim) for a list.
        :rtype: torch.Tensor
        """
        passed_single = isinstance(images, Image.Image)
        if passed_single:
            images = [images]
        else:
            if not isinstance(images, list):
                log_error(
                    f"images must be a PIL Image or list of PIL Images. Got {type(images)}"
                )
            if len(images) == 0:
                log_error("images cannot be empty.")
            for i, item in enumerate(images):
                if not isinstance(item, Image.Image):
                    log_error(f"Got {type(item)} instead of PIL Image at images[{i}].")
        result = self.do_embed_image(images=images, normalize=normalize)
        return result[0] if passed_single else result


class ImageTextEmbeddingModel(ABC):
    """Abstract base for models that embed (text, image) pairs into dense vectors."""

    @abstractmethod
    def do_embed_image_text(
        self,
        *,
        texts: list[str],
        images: list[Image.Image],
        normalize: bool,
    ) -> torch.Tensor:
        """
        Embed validated, paired (text, image) inputs.

        :param texts: Pre-validated list of strings, one per pair.
        :param images: Pre-validated list of PIL Images, one per pair.
        :param normalize: If True, L2-normalise each embedding.
        :return: Float tensor of shape (N, dim).
        :rtype: torch.Tensor
        """
        pass

    def embed(
        self,
        *,
        texts: Union[str, list[str]],
        images: Union[Image.Image, list[Image.Image]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Embed one or more (text, image) pairs.

        Both ``texts`` and ``images`` must be the same kind (single item or list).

        :param texts: A single string or a list of strings.
        :param images: A single PIL Image or a list of PIL Images.
        :param normalize: If True, L2-normalise the output embeddings. Default True.
        :return: Shape (dim,) for a single pair, (N, dim) for a list.
        :rtype: torch.Tensor
        """
        passed_str = isinstance(texts, str)
        passed_single_image = isinstance(images, Image.Image)
        if passed_str != passed_single_image:
            log_error(
                "texts and images must both be single items or both be lists. "
                f"Got {'str' if passed_str else 'list'} for texts and "
                f"{'Image' if passed_single_image else 'list'} for images."
            )
        if passed_str:
            if texts.strip() == "":
                log_error("texts cannot be empty.")
            texts = [texts]
            images = [images]
        else:
            if not isinstance(texts, list):
                log_error(
                    f"texts must be a string or list of strings. Got {type(texts)}"
                )
            if not isinstance(images, list):
                log_error(
                    f"images must be a PIL Image or list of PIL Images. Got {type(images)}"
                )
            if len(texts) == 0:
                log_error("texts cannot be empty.")
            if len(images) == 0:
                log_error("images cannot be empty.")
            if len(texts) != len(images):
                log_error(
                    f"texts and images must have the same length. "
                    f"Got {len(texts)} texts and {len(images)} images."
                )
            for i, item in enumerate(texts):
                if not isinstance(item, str):
                    log_error(f"Got {type(item)}:{item} instead of str at texts[{i}].")
            for i, item in enumerate(images):
                if not isinstance(item, Image.Image):
                    log_error(f"Got {type(item)} instead of PIL Image at images[{i}].")
        result = self.do_embed_image_text(
            texts=texts, images=images, normalize=normalize
        )
        return result[0] if passed_str else result


# ─────────────────────────────────────────────────────────────────────────────
# API text embedding
# ─────────────────────────────────────────────────────────────────────────────


class APITextEmbeddingModel(RateLimitedAPIBase, TextEmbeddingModel, ABC):
    """
    Abstract base for API-backed text embedding models.

    Handles rate limiting. Subclasses implement ``query_embedding_client``.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: Optional[str],
        api_key: Optional[str] = None,
        max_queries_per_minute: int = 60,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        :param model: Model identifier string.
        :param base_url: Base URL for the embeddings endpoint.
        :param api_key: API key for authentication. If None, uses environment variables.
        :param max_queries_per_minute: Maximum requests per minute. Must be >= 1.
        :param parameters: Loaded parameters dict. If None, loads from config.
        """
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )

    @abstractmethod
    def query_embedding_client(self, *, texts: list[str]) -> list[list[float]]:
        """
        Send texts to the embedding API and return raw embedding vectors.

        :param texts: List of strings to embed.
        :return: List of embedding vectors (each a list of floats), in input order.
        :rtype: list[list[float]]
        """
        pass

    def do_embed_text(self, *, texts: list[str], normalize: bool) -> torch.Tensor:
        # TODO: CHECK rate limiting is per-request here. Embedding APIs often impose
        # token-based or per-minute-request limits that vary by model tier. If you hit
        # 429s on large batches, lower max_queries_per_minute or split into smaller batches.
        self.wait()
        raw = self.query_embedding_client(texts=texts)
        embeddings = torch.tensor(raw, dtype=torch.float32)
        return maybe_normalize(embeddings, normalize)


class OpenAIAPITextEmbeddingModel(OpenAICompatibleAPIBase, APITextEmbeddingModel):
    """
    Text embedding backed by an OpenAI-compatible ``/v1/embeddings`` endpoint.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: Optional[str],
        api_key: Optional[str] = None,
        max_queries_per_minute: int = 60,
        parameters: dict[str, Any] = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )

    def query_embedding_client(self, *, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        # Sort by index to preserve input order regardless of API response ordering.
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


class OpenAITextEmbeddingModel(OpenAIAPITextEmbeddingModel):
    """
    Text embedding using the official OpenAI embeddings API (e.g. ``text-embedding-3-small``).
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: int = 60,
        parameters: dict[str, Any] = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=None,
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )


class OpenRouterTextEmbeddingModel(OpenAIAPITextEmbeddingModel):
    """
    Text embedding via OpenRouter's OpenAI-compatible embeddings endpoint.

    Reads ``OPENROUTER_API_KEY`` from the environment (same as ``OpenRouterModel``).

    # TODO: CHECK not all models listed on OpenRouter support the /v1/embeddings
    # endpoint. Verify the specific model you pass supports embeddings before use.
    # See the "API" tab on the model's OpenRouter page.
    """

    def __init__(
        self,
        *,
        model: str,
        max_queries_per_minute: int = 60,
        parameters: dict[str, Any] = None,
    ) -> None:
        api_key = os.environ["OPENROUTER_API_KEY"]
        super().__init__(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace text embedding store
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HuggingFaceTextEmbeddingModelStore:
    model: AutoModel
    tokenizer: Any
    model_kwargs: dict[str, Any]
    users: list = field(default_factory=list)


HUGGINGFACE_TEXT_EMBEDDING_MODEL_MAPPING: dict[
    str, HuggingFaceTextEmbeddingModelStore
] = {}


TEXT_STORE_LABEL = "text embedding store"


def hf_init_embedding_model(
    instance,
    *,
    model: str,
    parameters,
    model_kwargs: dict,
    mapping: dict,
    load_fn,
    remove_fn,
    model_type_label: str,
) -> None:
    """Shared init logic for all three HuggingFace embedding model classes."""
    instance.model = model
    instance.parameters = load_parameters(parameters)
    instance.model_kwargs = model_kwargs
    instance.is_defunct = False
    if model in mapping:
        existing_kwargs = mapping[model].model_kwargs
        if existing_kwargs != model_kwargs:
            log_warn(
                f"{model_type_label} model {model} already loaded with different kwargs. "
                f"Passed: {model_kwargs}, loaded: {existing_kwargs}. Reloading. "
                "Existing instances of this model will be marked defunct."
            )
            remove_fn(model_name=model)
            load_fn(model_name=model, model_kwargs=model_kwargs)
    else:
        load_fn(model_name=model, model_kwargs=model_kwargs)
    mapping[model].users.append(instance)


def remove_from_text_embedding_store(*, model_name: str, verbose: bool = False) -> None:
    """Remove a model from the text embedding store and free its GPU memory."""
    hf_remove_from_store(
        mapping=HUGGINGFACE_TEXT_EMBEDDING_MODEL_MAPPING,
        model_name=model_name,
        store_label=TEXT_STORE_LABEL,
        verbose=verbose,
    )


def clear_text_embedding_store() -> None:
    """Remove all models from the HuggingFace text embedding store and free GPU memory."""
    hf_clear_store(
        mapping=HUGGINGFACE_TEXT_EMBEDDING_MODEL_MAPPING, store_label=TEXT_STORE_LABEL
    )


def load_text_embedding_into_store(*, model_name: str, model_kwargs: dict) -> None:
    remove_from_text_embedding_store(model_name=model_name, verbose=False)
    log_info(
        f"Loading text embedding model {model_name} into store with kwargs {model_kwargs}"
    )
    model = AutoModel.from_pretrained(model_name, device_map="auto", **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    HUGGINGFACE_TEXT_EMBEDDING_MODEL_MAPPING[model_name] = (
        HuggingFaceTextEmbeddingModelStore(
            model=model, tokenizer=tokenizer, model_kwargs=model_kwargs
        )
    )


class HuggingFaceTextEmbeddingModel(TextEmbeddingModel):
    """
    HuggingFace encoder-based text embedding model.

    Uses mean pooling over token embeddings weighted by the attention mask.
    Suitable for models such as ``Qwen/Qwen3-Embedding-0.6B``, ``BAAI/bge-*``,
    ``intfloat/e5-*``, etc.
    """

    def __init__(
        self,
        *,
        model: str,
        parameters: dict[str, Any] = None,
        **model_kwargs,
    ) -> None:
        """
        :param model: HuggingFace model identifier.
        :param parameters: Loaded parameters dict. If None, loads from config.
        :param model_kwargs: Extra kwargs forwarded to ``AutoModel.from_pretrained``.
        """
        hf_init_embedding_model(
            self,
            model=model,
            parameters=parameters,
            model_kwargs=model_kwargs,
            mapping=HUGGINGFACE_TEXT_EMBEDDING_MODEL_MAPPING,
            load_fn=load_text_embedding_into_store,
            remove_fn=remove_from_text_embedding_store,
            model_type_label="Text embedding",
        )

    def do_embed_text(self, *, texts: list[str], normalize: bool) -> torch.Tensor:
        if self.is_defunct:
            log_error("Cannot embed with a defunct model.", parameters=self.parameters)
        store = HUGGINGFACE_TEXT_EMBEDDING_MODEL_MAPPING[self.model]
        # TODO: CHECK truncation=True silently drops tokens beyond the model's max_length.
        # For models with short context windows (e.g. 512 tokens) this loses information
        # without warning. If your texts may be long, add a length check before calling.
        inputs = store.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(store.model.device)
        with torch.no_grad():
            outputs = store.model(**inputs)
        # TODO: CHECK mean pooling over last_hidden_state is correct for most encoder
        # embedding models (bge, e5, nomic-embed, etc.). However some models (e.g.
        # Qwen3-Embedding) use the last token (EOS) as the pooling token rather than
        # mean pooling. Verify the recommended pooling strategy for your specific model.
        embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        return maybe_normalize(embeddings, normalize)


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace image embedding store
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HuggingFaceImageEmbeddingModelStore:
    model: AutoModel
    processor: Any
    model_kwargs: dict[str, Any]
    users: list = field(default_factory=list)


HUGGINGFACE_IMAGE_EMBEDDING_MODEL_MAPPING: dict[
    str, HuggingFaceImageEmbeddingModelStore
] = {}


IMAGE_STORE_LABEL = "image embedding store"


def remove_from_image_embedding_store(*, model_name: str, verbose: bool = False) -> None:
    """Remove a model from the image embedding store and free its GPU memory."""
    hf_remove_from_store(
        mapping=HUGGINGFACE_IMAGE_EMBEDDING_MODEL_MAPPING,
        model_name=model_name,
        store_label=IMAGE_STORE_LABEL,
        verbose=verbose,
    )


def clear_image_embedding_store() -> None:
    """Remove all models from the HuggingFace image embedding store and free GPU memory."""
    hf_clear_store(
        mapping=HUGGINGFACE_IMAGE_EMBEDDING_MODEL_MAPPING,
        store_label=IMAGE_STORE_LABEL,
    )


def load_image_embedding_into_store(*, model_name: str, model_kwargs: dict) -> None:
    remove_from_image_embedding_store(model_name=model_name, verbose=False)
    log_info(
        f"Loading image embedding model {model_name} into store with kwargs {model_kwargs}"
    )
    model = AutoModel.from_pretrained(model_name, device_map="auto", **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name)
    HUGGINGFACE_IMAGE_EMBEDDING_MODEL_MAPPING[model_name] = (
        HuggingFaceImageEmbeddingModelStore(
            model=model, processor=processor, model_kwargs=model_kwargs
        )
    )


class HuggingFaceImageEmbeddingModel(ImageEmbeddingModel):
    """
    HuggingFace image embedding model.

    Uses ``get_image_features()`` when available (CLIP and CLIP-like models).
    Falls back to ``pooler_output``, then to the CLS token of ``last_hidden_state``
    for models that don't expose ``get_image_features``.
    """

    def __init__(
        self,
        *,
        model: str,
        parameters: dict[str, Any] = None,
        **model_kwargs,
    ) -> None:
        """
        :param model: HuggingFace model identifier.
        :param parameters: Loaded parameters dict. If None, loads from config.
        :param model_kwargs: Extra kwargs forwarded to ``AutoModel.from_pretrained``.
        """
        hf_init_embedding_model(
            self,
            model=model,
            parameters=parameters,
            model_kwargs=model_kwargs,
            mapping=HUGGINGFACE_IMAGE_EMBEDDING_MODEL_MAPPING,
            load_fn=load_image_embedding_into_store,
            remove_fn=remove_from_image_embedding_store,
            model_type_label="Image embedding",
        )

    def do_embed_image(
        self, *, images: list[Image.Image], normalize: bool
    ) -> torch.Tensor:
        if self.is_defunct:
            log_error("Cannot embed with a defunct model.", parameters=self.parameters)
        store = HUGGINGFACE_IMAGE_EMBEDDING_MODEL_MAPPING[self.model]
        # TODO: CHECK padding=True may not be accepted by all image processors (it is
        # valid for CLIP's CLIPProcessor but some ViT processors silently ignore it or
        # raise). If you hit unexpected processor errors, try removing padding=True.
        inputs = store.processor(images=images, return_tensors="pt", padding=True).to(
            store.model.device
        )
        with torch.no_grad():
            if hasattr(store.model, "get_image_features"):
                # CLIP-style: get_image_features projects the vision encoder output
                # through the projection head, which is usually what you want for retrieval.
                embeddings = store.model.get_image_features(**inputs)
            else:
                # TODO: CHECK this fallback (pooler_output then CLS token) is a
                # best-guess for non-CLIP vision encoders (e.g. ViT, DINOv2). The right
                # pooling strategy is model-specific — check the model card.
                outputs = store.model(**inputs)
                if (
                    hasattr(outputs, "pooler_output")
                    and outputs.pooler_output is not None
                ):
                    embeddings = outputs.pooler_output
                else:
                    embeddings = outputs.last_hidden_state[:, 0]
        return maybe_normalize(embeddings, normalize)


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace image+text embedding store
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HuggingFaceImageTextEmbeddingModelStore:
    model: AutoModel
    processor: Any
    model_kwargs: dict[str, Any]
    users: list = field(default_factory=list)


HUGGINGFACE_IMAGE_TEXT_EMBEDDING_MODEL_MAPPING: dict[
    str, HuggingFaceImageTextEmbeddingModelStore
] = {}


IMAGETEXT_STORE_LABEL = "image+text embedding store"


def remove_from_image_text_embedding_store(
    *, model_name: str, verbose: bool = False
) -> None:
    """Remove a model from the image+text embedding store and free its GPU memory."""
    hf_remove_from_store(
        mapping=HUGGINGFACE_IMAGE_TEXT_EMBEDDING_MODEL_MAPPING,
        model_name=model_name,
        store_label=IMAGETEXT_STORE_LABEL,
        verbose=verbose,
    )


def clear_image_text_embedding_store() -> None:
    """Remove all models from the HuggingFace image+text embedding store and free GPU memory."""
    hf_clear_store(
        mapping=HUGGINGFACE_IMAGE_TEXT_EMBEDDING_MODEL_MAPPING,
        store_label=IMAGETEXT_STORE_LABEL,
    )


def load_image_text_embedding_into_store(
    *, model_name: str, model_kwargs: dict
) -> None:
    remove_from_image_text_embedding_store(model_name=model_name, verbose=False)
    log_info(
        f"Loading image+text embedding model {model_name} into store with kwargs {model_kwargs}"
    )
    model = AutoModel.from_pretrained(model_name, device_map="auto", **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name)
    HUGGINGFACE_IMAGE_TEXT_EMBEDDING_MODEL_MAPPING[model_name] = (
        HuggingFaceImageTextEmbeddingModelStore(
            model=model, processor=processor, model_kwargs=model_kwargs
        )
    )


class HuggingFaceImageTextEmbeddingModel(ImageTextEmbeddingModel):
    """
    HuggingFace multimodal (image + text) embedding model.

    Produces one joint embedding per (text, image) pair. Output selection priority:
    ``image_embeds`` → ``pooler_output`` → CLS token of ``last_hidden_state``.

    For models that expose a custom ``.encode()`` API (e.g. ``jinaai/jina-embeddings-v4``),
    subclass this and override ``do_embed_image_text``. Those models also typically
    require ``trust_remote_code=True`` passed as a ``model_kwarg``.
    """

    def __init__(
        self,
        *,
        model: str,
        parameters: dict[str, Any] = None,
        **model_kwargs,
    ) -> None:
        """
        :param model: HuggingFace model identifier.
        :param parameters: Loaded parameters dict. If None, loads from config.
        :param model_kwargs: Extra kwargs forwarded to ``AutoModel.from_pretrained``.
        """
        hf_init_embedding_model(
            self,
            model=model,
            parameters=parameters,
            model_kwargs=model_kwargs,
            mapping=HUGGINGFACE_IMAGE_TEXT_EMBEDDING_MODEL_MAPPING,
            load_fn=load_image_text_embedding_into_store,
            remove_fn=remove_from_image_text_embedding_store,
            model_type_label="Image+text embedding",
        )

    def do_embed_image_text(
        self,
        *,
        texts: list[str],
        images: list[Image.Image],
        normalize: bool,
    ) -> torch.Tensor:
        if self.is_defunct:
            log_error("Cannot embed with a defunct model.", parameters=self.parameters)
        store = HUGGINGFACE_IMAGE_TEXT_EMBEDDING_MODEL_MAPPING[self.model]
        inputs = store.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(store.model.device)
        with torch.no_grad():
            outputs = store.model(**inputs)
        # TODO: CHECK the output selection here is a best-effort heuristic and is likely
        # wrong for your specific model. `image_embeds` alone is NOT a joint text+image
        # embedding — it is just the image projection and ignores the text entirely.
        # True multimodal encoders (e.g. jina-embeddings-v4, ImageBind) produce a joint
        # embedding via a cross-modal forward pass; the right output field depends on the
        # model. Inspect `outputs.keys()` for your model and override this method in a
        # subclass. The fallback chain below is a placeholder only.
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            embeddings = outputs.image_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state[:, 0]
        return maybe_normalize(embeddings, normalize)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding utilities
# ─────────────────────────────────────────────────────────────────────────────


def cosine_similarity(*, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between two sets of embeddings.

    :param a: Tensor of shape (M, dim) or (dim,).
    :param b: Tensor of shape (N, dim) or (dim,).
    :return: Similarity matrix of shape (M, N). If both inputs are 1-D, returns a scalar tensor.
    :rtype: torch.Tensor
    """
    squeeze = a.dim() == 1 and b.dim() == 1
    a = F.normalize(a.unsqueeze(0) if a.dim() == 1 else a, p=2, dim=-1)
    b = F.normalize(b.unsqueeze(0) if b.dim() == 1 else b, p=2, dim=-1)
    sim = a @ b.T
    return sim.squeeze() if squeeze else sim


def get_top_k_similars(
    *,
    query: torch.Tensor,
    corpus: torch.Tensor,
    k: int,
    largest: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the top-k cosine similarity scores and their corpus indices for each query.

    :param query: Tensor of shape (Q, dim) or (dim,) for a single query.
    :param corpus: Tensor of shape (N, dim).
    :param k: Number of top results to return per query.
    :param largest: If True (default), return highest-similarity items.
    :return: Tuple of (scores, indices), each of shape (Q, k). Shape is (k,) when query is 1-D.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    squeeze = query.dim() == 1
    if squeeze:
        query = query.unsqueeze(0)
    if k > corpus.shape[0]:
        log_error(
            f"k={k} is larger than corpus size {corpus.shape[0]}. "
            "k must be <= number of corpus embeddings."
        )
    sim = cosine_similarity(a=query, b=corpus)  # (Q, N)
    result = torch.topk(sim, k=k, dim=-1, largest=largest)
    scores, indices = result.values, result.indices
    if squeeze:
        return scores.squeeze(0), indices.squeeze(0)
    return scores, indices
