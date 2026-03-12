# Embedding Module — Implementation Notes

Notes written during implementation of `utils/embedding.py`. These cover design decisions,
known limitations, and discrepancies found when cross-checking against real model docs and
the Tevatron retrieval toolkit.

---

## Mean pooling is wrong for decoder-based embedding models

`HuggingFaceTextEmbeddingModel` defaults to mean-pooling over `last_hidden_state`. This is
correct for encoder-only models (BGE-small, nomic-embed, jina-v2, most sentence-transformers).

**It is wrong for any decoder or decoder-based embedding model**, which use *last-token (EOS)
pooling* instead:

| Model | Correct pooling | Our default |
|---|---|---|
| `Qwen/Qwen3-Embedding-*` | Last token (EOS) | ✗ mean pool |
| `intfloat/e5-mistral-7b-instruct` | Last token (EOS) | ✗ mean pool |
| `BAAI/bge-m3` | CLS token (+ MCLS for long docs) | ✗ mean pool |
| `BAAI/bge-small-en-v1.5` | CLS token | ✓ mean pool ≈ fine |
| `jinaai/jina-embeddings-v2-base-en` | Mean pool | ✓ |

The correct last-token pooling logic (from Qwen3 and E5-Mistral model cards):

```python
def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]
```

**Recommendation**: Add a `pooling_strategy` parameter to `HuggingFaceTextEmbeddingModel`
with options `"mean"`, `"cls"`, `"last_token"` (default `"mean"`). Also note that
decoder-based models require `padding_side='left'` on the tokenizer.

---

## BGE-M3 cannot be loaded with AutoModel

`BAAI/bge-m3` requires the `BGEM3FlagModel` class from the `FlagEmbedding` package:

```python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
embeddings = model.encode(sentences)['dense_vecs']
```

Our current `HuggingFaceTextEmbeddingModel` will load it with `AutoModel` and use mean
pooling — both wrong. BGE-M3 also supports sparse retrieval and ColBERT multi-vector modes
that are inaccessible via `AutoModel`.

**Recommendation**: Either document that BGE-M3 is not supported by the generic class,
or add a `BGEM3TextEmbeddingModel` subclass that uses `FlagEmbedding` directly.

---

## Jina-embeddings-v4 requires custom encode methods

`jinaai/jina-embeddings-v4` loads with `AutoModel` + `trust_remote_code=True`, but the
correct forward API is model-specific:

```python
# Text:
query_embeddings = model.encode_text(texts=["query"], task="retrieval", prompt_name="query")
# Images:
image_embeddings = model.encode_image(images=[img], task="retrieval")
```

Calling `model(**inputs)` directly with a standard processor will not produce correct
embeddings. Our `HuggingFaceImageTextEmbeddingModel` forward pass is a placeholder and
will not work out-of-the-box for jina-v4.

The model also supports Matryoshka dimensions (128–2048), task adapters (retrieval,
text-matching, code), and multi-vector mode. Requires:
`transformers>=4.52.0`, `peft>=0.15.2`, `torch>=2.6.0`, `torchvision`.

**Recommendation**: Subclass `HuggingFaceImageTextEmbeddingModel` and override
`do_embed_image_text` for this model specifically.

---

## CLIP works correctly via AutoModel

`openai/clip-vit-large-patch14`: `AutoModel.from_pretrained(...)` returns a `CLIPModel`
which has `get_image_features()`. Our `hasattr(model, "get_image_features")` branch
therefore fires correctly for CLIP. The image-only path in `HuggingFaceImageEmbeddingModel`
is correct.

For joint image+text use of CLIP (i.e., via `HuggingFaceImageTextEmbeddingModel`), the
correct outputs are `text_embeds` and `image_embeds` separately — there is no single
fused vector. Our fallback that returns `image_embeds` alone silently drops the text.

---

## DINOv2 works correctly via AutoModel

`facebook/dinov2-base`: `AutoModel` works. The model card confirms CLS token (index 0 of
`last_hidden_state`) is the correct representation. Our fallback chain in
`HuggingFaceImageEmbeddingModel` reaches the CLS branch correctly.

---

## Tevatron uses EOS/CLS pooling, not mean pooling

The Tevatron dense retrieval toolkit (`texttron/tevatron`) defaults to EOS token pooling
(`--pooling eos` with `--append_eos_token`), not mean pooling. CLS pooling is the
alternative. Mean pooling is not the standard in dense retrieval pipelines.

This further supports adding a `pooling_strategy` parameter to `HuggingFaceTextEmbeddingModel`.

---

## Rate limiting is request-based, not token-based

`APITextEmbeddingModel.wait()` enforces one request per `60 / max_queries_per_minute`
seconds. Embedding API limits are often expressed as tokens-per-minute or
requests-per-minute depending on the provider. For large batches, you may hit token-per-
minute limits before hitting request-per-minute limits. If you receive 429s on large
batches, split into smaller sub-batches before calling `embed()`.

---

## Tokenizer truncation is silent

All HuggingFace text embedding paths use `truncation=True` without an explicit
`max_length`. Tokens beyond the model's default max will be silently dropped. For models
with short windows (512 tokens is common for older encoders), long documents will lose
content without any warning. Add a pre-check if this matters for your use case.

---

## image_embeds in multimodal forward pass is not a joint embedding

In `HuggingFaceImageTextEmbeddingModel.do_embed_image_text`, the output selection
fallback prefers `image_embeds`. For CLIP-like models this is the image projection only —
it ignores the text input. A true joint embedding does not exist in CLIP; you must compare
`image_embeds` and `text_embeds` separately (via dot product / cosine similarity).

The `do_embed_image_text` fallback chain is a placeholder. For any real use, subclass
and override it.

---

## processor padding=True may fail for some image processors

`AutoProcessor` for image-only models may not accept a `padding` kwarg. It is valid for
`CLIPProcessor` but other processors (e.g., plain `ViTImageProcessor`) may silently ignore
it or raise `TypeError`. If you hit processor errors, try removing `padding=True` from
`do_embed_image`.
