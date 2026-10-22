"""MOSS JSON captions (``caption_format = "moss_json"``), compatible with ai-toolkit's ``caption_format: moss_json``.

One JSON file per image (``001.png`` and ``001.json``)::

    {
      "caption": {"short": "...", "medium": "...", "long": "..."},
      "tags": ["1girl", "black hair"],
      "keep_tags": ["mychar"],
      "watermark": [],
      "borders": ["none"]
    }

* ``caption.short`` / ``caption.medium`` / ``caption.long`` are required non-empty natural-language captions. Line breaks
  and visible text are kept.
* ``keep_tags`` (optional) are protected tags: always placed first in the written order, never shuffled or dropped by
  ``caption_tag_dropout_rate``.
* ``tags`` (optional) are general tags: shuffled and dropped by the tag augmentation. Duplicates are removed while
  keeping the order, and a tag listed in both arrays is kept only in ``keep_tags``.
* Other fields (``watermark``, ``borders``, ...) are never added to the training caption.

For each sample one variant is chosen: ``tags`` (keep_tags + tags), ``nl`` (keep_tags + natural language), ``tags_nl``
(keep_tags + tags + natural language) or ``nl_tags`` (keep_tags + natural language + tags). With
``caption_mode = "mixed"`` the variant is drawn with ``mixed_weights``, otherwise ``tags`` is used. An image without
general tags always uses ``nl``. For the variants with natural language, the caption length is drawn independently
with ``caption_length_weights``.

The dataset stores the JSON text in ``ImageInfo.caption``; :func:`parse_moss_caption_text` parses it (cached).
"""

import functools
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tqdm import tqdm

CAPTION_FORMATS = ("text", "moss_json")
MIXED_CAPTION_VARIANTS = ("tags", "nl", "tags_nl", "nl_tags")
CAPTION_LENGTHS = ("short", "medium", "long")
DEFAULT_MOSS_JSON_CAPTION_EXTENSION = ".json"
PROTECTED_VARIANT = "protected"

_VARIANTS_WITH_TAGS = ("tags", "tags_nl", "nl_tags")
_VARIANTS_WITH_NL = ("nl", "tags_nl", "nl_tags")
_MAX_ERRORS_TO_SHOW = 20


def is_moss_json(subset: Any) -> bool:
    return getattr(subset, "caption_format", "text") == "moss_json"


@dataclass(frozen=True)
class MossCaption:
    """A parsed MOSS JSON caption. ``keep_tags`` and ``tags`` are distinct and stripped."""

    keep_tags: Tuple[str, ...]
    tags: Tuple[str, ...]
    short: str
    medium: str
    long: str

    def nl(self, length: str) -> str:
        return getattr(self, length)

    def tag_caption(self, separator: str = ",") -> str:
        """keep_tags and tags as one comma separated string, used for the tag frequency of the metadata."""
        return f"{separator} ".join(self.keep_tags + self.tags)


@dataclass(frozen=True)
class MossCaptionChoice:
    """A variant (and the caption length for variants with natural language) with its probability."""

    variant: str
    length: Optional[str]
    probability: float

    @property
    def label(self) -> str:
        return self.variant if self.length is None else f"{self.variant}:{self.length}"


def parse_moss_caption(value: Any, source: str = "MOSS JSON caption") -> MossCaption:
    """Validate a decoded MOSS JSON object. Raises ``ValueError`` with ``source`` in the message if invalid."""
    if not isinstance(value, dict) or not isinstance(value.get("caption"), dict):
        raise ValueError(f"{source}: expected a `caption` object with short, medium and long")

    captions = {}
    for length in CAPTION_LENGTHS:
        text = value["caption"].get(length)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{source}: caption.{length} must be a non-empty string")
        captions[length] = text

    def read_tags(key: str) -> Tuple[str, ...]:
        tags = value.get(key, [])
        if tags is None:
            tags = []
        if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise ValueError(f"{source}: {key} must be an array of non-empty strings")
        return tuple(dict.fromkeys(tag.strip() for tag in tags))

    keep_tags = read_tags("keep_tags")
    tags = tuple(tag for tag in read_tags("tags") if tag not in keep_tags)
    return MossCaption(keep_tags, tags, captions["short"], captions["medium"], captions["long"])


@functools.lru_cache(maxsize=1 << 16)
def parse_moss_caption_text(text: str) -> MossCaption:
    """Parse the JSON text stored in ``ImageInfo.caption``. The result is cached and must not be modified."""
    try:
        value = json.loads(text)
    except ValueError as e:
        raise ValueError(f"MOSS JSON caption is not valid JSON: {e}") from e
    return parse_moss_caption(value)


def get_moss_caption_paths(img_path: str, caption_extension: str) -> List[str]:
    """Candidate caption paths of an image, same as ``read_caption`` (including the face crop naming)."""
    base_name = os.path.splitext(img_path)[0]
    paths = [base_name + caption_extension]
    tokens = base_name.split("_")
    if len(tokens) >= 5:
        paths.append("_".join(tokens[:-4]) + caption_extension)
    return paths


def read_moss_caption_text(img_path: str, caption_extension: str) -> Optional[str]:
    """Read the MOSS JSON text of an image as is (UTF-8, a BOM is removed). Returns None if no file is found.
    Raises ``ValueError`` with the path if the file cannot be read."""
    for path in get_moss_caption_paths(img_path, caption_extension):
        if os.path.isfile(path):
            try:
                with open(path, "rt", encoding="utf-8-sig") as f:
                    return f.read()
            except UnicodeDecodeError as e:
                raise ValueError(f"{path}: not UTF-8 ({e})") from e
            except OSError as e:
                raise ValueError(f"{path}: cannot be read ({e})") from e
    return None


def load_moss_caption_texts(
    img_paths: Sequence[str], caption_extension: str, texts: Optional[Sequence[Optional[str]]] = None
) -> List[str]:
    """Read (unless ``texts`` are given, e.g. from the image info cache) and validate the MOSS JSON texts of all images.

    Raises ``ValueError`` listing every missing, unreadable or invalid file at once, so that the training stops before
    it starts. Returns the JSON texts."""
    errors = []
    loaded = []
    for i, img_path in enumerate(tqdm(img_paths, desc="read MOSS JSON caption", disable=texts is not None)):
        source = get_moss_caption_paths(img_path, caption_extension)[0]
        if texts is not None:
            text = texts[i]
        else:
            try:
                text = read_moss_caption_text(img_path, caption_extension)
            except ValueError as e:
                errors.append(str(e))
                continue
        loaded.append(text)
        if text is None:
            errors.append(f"{source}: file not found")
            continue
        try:
            value = json.loads(text)
        except ValueError as e:
            errors.append(f"{source}: invalid JSON ({e})")
            continue
        try:
            parse_moss_caption(value, source)
        except ValueError as e:
            errors.append(str(e))

    if errors:
        shown = "\n".join(errors[:_MAX_ERRORS_TO_SHOW])
        if len(errors) > _MAX_ERRORS_TO_SHOW:
            shown += f"\n... and {len(errors) - _MAX_ERRORS_TO_SHOW} more"
        raise ValueError(
            f"{len(errors)} MOSS JSON caption file(s) are missing or invalid"
            + f" / {len(errors)} 個の MOSS JSON キャプションファイルが見つからないか不正です:\n{shown}"
        )
    return loaded


def normalize_weights(name: str, weights: Optional[Mapping[str, Any]], keys: Sequence[str]) -> Dict[str, float]:
    """Validate relative weights. ``None`` gives equal weights. A key omitted from a given mapping is zero. The weights
    must be finite and non-negative, and at least one must be positive."""
    if weights is None:
        return {key: 1.0 for key in keys}
    if not isinstance(weights, Mapping):
        raise ValueError(f"{name} must be a mapping of {', '.join(keys)}")
    unknown = [key for key in weights if key not in keys]
    if unknown:
        raise ValueError(f"{name} only supports {', '.join(keys)}, got {unknown}")
    try:
        normalized = {key: float(weights.get(key, 0.0)) for key in keys}
    except (OverflowError, TypeError, ValueError) as e:
        raise ValueError(f"{name} must contain finite numeric values") from e
    if any(not math.isfinite(weight) for weight in normalized.values()) or not math.isfinite(sum(normalized.values())):
        raise ValueError(f"{name} must contain finite numeric values")
    if any(weight < 0 for weight in normalized.values()):
        raise ValueError(f"{name} cannot contain negative values")
    if sum(normalized.values()) <= 0:
        raise ValueError(f"{name} must contain at least one positive value")
    return normalized


def get_choices(
    caption: MossCaption,
    caption_mode: str,
    mixed_weights: Mapping[str, float],
    caption_length_weights: Mapping[str, float],
) -> List[MossCaptionChoice]:
    """All enabled (variant, length) choices of the caption with their probabilities (summing to 1)."""
    if not caption.tags:
        variant_probs = [("nl", 1.0)]
    elif caption_mode == "mixed":
        enabled = [(variant, float(mixed_weights.get(variant, 0))) for variant in MIXED_CAPTION_VARIANTS]
        enabled = [(variant, weight) for variant, weight in enabled if weight > 0]
        total = sum(weight for _, weight in enabled)
        variant_probs = [(variant, weight / total) for variant, weight in enabled]
    else:
        variant_probs = [("tags", 1.0)]

    lengths = [(length, float(caption_length_weights.get(length, 0))) for length in CAPTION_LENGTHS]
    lengths = [(length, weight) for length, weight in lengths if weight > 0]
    length_total = sum(weight for _, weight in lengths)

    choices = []
    for variant, variant_prob in variant_probs:
        if variant in _VARIANTS_WITH_NL:
            for length, weight in lengths:
                choices.append(MossCaptionChoice(variant, length, variant_prob * weight / length_total))
        else:
            choices.append(MossCaptionChoice(variant, None, variant_prob))
    return choices


def choose(choices: Sequence[MossCaptionChoice]) -> MossCaptionChoice:
    if len(choices) == 1:
        return choices[0]
    return random.choices(choices, weights=[choice.probability for choice in choices], k=1)[0]


def augment_tags(
    caption: MossCaption,
    keep_tokens: int,
    shuffle: bool,
    tag_dropout_rate: float,
    special_tag_dropout_rate: float,
    protected_tags: Optional[set] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Shuffle and drop the general tags. Returns (keep_tags, general tags, dropped tags).

    The first ``keep_tokens`` general tags are neither shuffled nor dropped by ``tag_dropout_rate``, same as the text
    captions. ``special_tag_dropout_rate`` drops keep_tags and those first general tags. Tags in ``protected_tags`` are
    never dropped. The natural-language caption is not changed.
    """
    protected_tags = protected_tags or set()
    dropped = []

    def drop(tags: List[str], rate: float) -> List[str]:
        kept = []
        for tag in tags:
            if tag in protected_tags or random.random() >= rate:
                kept.append(tag)
            else:
                dropped.append(tag)
        return kept

    keep_tags = list(caption.keep_tags)
    fixed = list(caption.tags[:keep_tokens])
    flex = list(caption.tags[keep_tokens:])
    if shuffle:
        random.shuffle(flex)
    if tag_dropout_rate > 0:
        flex = drop(flex, tag_dropout_rate)
    if special_tag_dropout_rate > 0:
        keep_tags = drop(keep_tags, special_tag_dropout_rate)
        fixed = drop(fixed, special_tag_dropout_rate)
    return keep_tags, fixed + flex, dropped


def render(
    caption: MossCaption,
    choice: MossCaptionChoice,
    keep_tags: Sequence[str],
    tags: Sequence[str],
    separator: str = ",",
) -> str:
    """Join keep_tags, tags and the natural-language caption in the order of the variant."""
    nl = [caption.nl(choice.length).strip()] if choice.variant in _VARIANTS_WITH_NL else []
    tags = list(tags) if choice.variant in _VARIANTS_WITH_TAGS else []
    if choice.variant == "nl_tags":
        parts = list(keep_tags) + nl + tags
    else:
        parts = list(keep_tags) + tags + nl
    return f"{separator} ".join(part for part in parts if part)


def render_protected(caption: MossCaption, keep_tokens: int, separator: str = ",") -> str:
    """The caption of protected caption dropout: keep_tags and the first ``keep_tokens`` general tags, without the
    natural language. Empty if there is nothing protected (the same as the full caption dropout)."""
    return f"{separator} ".join(list(caption.keep_tags) + list(caption.tags[:keep_tokens]))
