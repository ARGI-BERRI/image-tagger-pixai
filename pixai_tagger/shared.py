from typing import Any

from loguru import logger


def get_tagger(model_id: str, tagger: object | None = None) -> Any:
    """Lazy-load the tagger. Loaded model will be bound to `tagger`.

    If `tagger` is not None, this will return it as-is.

    Args:
        model_id (str): Model ID available in Hugging Face
        tagger (object | None, optional): Tagger object. Defaults to None.

    Returns:
        Any: Tagger
    """
    if tagger is not None:
        return tagger

    logger.debug("Loading tagger model (lazy loading)")
    from transformers import pipeline

    tagger = pipeline(
        model=model_id,
        image_processor=model_id,
        trust_remote_code=True,
    )  # pyright: ignore[reportCallIssue]

    logger.debug("Tagger model is loaded")
    return tagger
