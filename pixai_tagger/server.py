from asyncio import to_thread
from io import BytesIO
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile
from loguru import logger
from PIL import Image
from pydantic import BaseModel, ConfigDict

from pixai_tagger.shared import get_tagger

app = FastAPI()
tagger = get_tagger("pixai-labs/pixai-tagger-v1.0")


class TaggerResponse(BaseModel):
    """Tag categories mapping tag names to numeric scores."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "general": {"1girl": 0.98, "solo": 0.95},
                    "character": {"hatsune_miku": 0.92},
                    "copyright": {"vocaloid": 0.94},
                    "style": {"anime": 0.97},
                    "meta": {"highres": 0.91},
                    "rating": {
                        "rating:g": 0.97,
                        "rating:s": 0.02,
                        "rating:q": 0.009,
                        "rating:e": 0.001,
                    },
                }
            ]
        },
        use_attribute_docstrings=True,
    )

    general: dict[str, float]
    """Visible content, including clothing, poses, objects, and composition"""

    character: dict[str, float]
    """Named characters"""

    copyright: dict[str, float]
    """Source works and franchises, plus `original`"""

    style: dict[str, float]
    """Style labels"""

    meta: dict[str, float]
    """Medium, provenance, resolution, and status labels"""

    rating: dict[Literal["rating:g", "rating:s", "rating:q", "rating:e"], float]
    """`rating:g`, `rating:s`, `rating:q`, and `rating:e`"""


@app.post("/tag")
async def tag_image(image_bytes: UploadFile) -> TaggerResponse:
    """Give an image tags based on its content."""
    logger.info(
        f"[{image_bytes.filename}] Received a request ({image_bytes.size} bytes)"
    )

    try:
        image = Image.open(BytesIO(await image_bytes.read()))
        image.load()

    except Exception as e:  # noqa: BLE001
        logger.error(f"[{image_bytes.filename}] Failed to open image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to open image: {e}")

    results = await to_thread(tagger, image)
    tags = TaggerResponse.model_validate(results["results"])

    logger.info(f"[{image_bytes.filename}] Tags: {tags.model_dump()}")

    return tags
