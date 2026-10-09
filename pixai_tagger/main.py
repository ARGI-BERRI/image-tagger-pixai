import json
import sys
from pathlib import Path

from loguru import logger
from PIL import Image
from sixel import converter
from typer import run

from pixai_tagger.shared import get_tagger


def main(
    image_path: str | None = None,
    cached: bool = True,
    model_id: str = "pixai-labs/pixai-tagger-v1.0",
    preview_width: int = 180,
) -> None:
    images_dir = Path(__file__).parent.parent / "images"

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    image_paths = list(
        images_dir.iterdir() if image_path is None else [Path(image_path)]
    )

    # Load the tagger once it's needed
    # If the given images are all cached, the tagger will never be loaded
    tagger = None

    for img_path in image_paths:
        if img_path.name == ".gitkeep":
            continue

        image = Image.open(img_path)
        w, h = image.size

        if cached and (results_dir / f"{img_path.stem}.json").exists():
            results = json.loads((results_dir / f"{img_path.stem}.json").read_text())

        else:
            tagger = get_tagger(model_id, tagger)
            results = tagger(image)["results"]

        converter.SixelConverter(
            img_path.as_posix(), w=preview_width, h=int(h * preview_width / w)
        ).write(sys.stderr)

        logger.info(f"Image: {img_path.name}")
        logger.info(
            ", ".join(
                [
                    tag
                    for genre, tags in results.items()
                    if genre != "rating"
                    for tag in tags
                ]
            )
        )

        (results_dir / f"{img_path.stem}.json").write_text(
            json.dumps(results, indent=2)
        )


if __name__ == "__main__":
    run(main)
