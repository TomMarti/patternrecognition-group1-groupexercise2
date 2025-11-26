from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

# Target height for normalized word images (you can also import this from main_ex3.py)
TARGET_HEIGHT = 100


# ---------------------------------------------------------------------------
# Helper functions for loading metadata
# ---------------------------------------------------------------------------

def load_split_doc_ids(data_root: Path, split: str) -> List[str]:
    """
    Reads train.tsv or validation.tsv and returns a list of document IDs,
    e.g. ["270", "271", ...].

    split: "train" or "validation"
    """
    tsv_path = data_root / f"{split}.tsv"
    doc_ids: List[str] = []

    with tsv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # In the exercise, each line contains a document number (e.g., 270)
            doc_ids.append(line)

    return doc_ids


def load_transcriptions(data_root: Path) -> Dict[str, str]:
    """
    Reads transcription.tsv and returns a dict mapping word_id -> transcription,
    e.g. "277-02-06" -> "y-o-u".
    """
    trans_path = data_root / "transcription.tsv"
    transcriptions: Dict[str, str] = {}

    with trans_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Assumption: word_id <TAB> transcription
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            word_id, transcription = parts
            transcriptions[word_id] = transcription

    return transcriptions


# ---------------------------------------------------------------------------
# SVG parsing: extract polygons from locations/DDD.svg
# ---------------------------------------------------------------------------

def parse_svg_paths_for_doc(locations_dir: Path, doc_id: str) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """
    Reads locations/DDD.svg and returns a list of (word_id, polygon_points).

    polygon_points: list of (x, y) tuples in image coordinate space.
    """
    svg_path = locations_dir / f"{doc_id}.svg"
    tree = ET.parse(svg_path)
    root = tree.getroot()

    ns = {"svg": "http://www.w3.org/2000/svg"}  # sometimes necessary if <svg ...> contains namespaces

    word_polygons: List[Tuple[str, List[Tuple[float, float]]]] = []

    # We search for all <path> elements. If your files use <polygon>,
    # you can additionally use root.findall(".//{http://www.w3.org/2000/svg}polygon")
    for elem in root.iter():
        if elem.tag.endswith("path"):
            word_id = elem.get("id")
            d_attr = elem.get("d")

            if not word_id or not d_attr:
                continue

            points = _parse_svg_path_d_attribute(d_attr)
            if points:
                word_polygons.append((word_id, points))

    return word_polygons


def _parse_svg_path_d_attribute(d: str) -> List[Tuple[float, float]]:
    """
    Very simple parser for paths of the form:
    "M x y L x y L x y ... Z"

    Assumption: only M, L, and Z, separated by spaces.
    For Washington SVGs this is usually sufficient.
    """
    tokens = d.strip().split()

    points: List[Tuple[float, float]] = []
    i = 0
    current_cmd = None

    while i < len(tokens):
        token = tokens[i]

        if token in ("M", "L"):
            current_cmd = token
            i += 1
            continue
        elif token == "Z":
            # close path -> no extra coordinate needed
            break
        else:
            # Expecting a number (x or y)
            if current_cmd in ("M", "L"):
                # next token is x, then y
                x_str = token
                if i + 1 >= len(tokens):
                    break
                y_str = tokens[i + 1]
                try:
                    x = float(x_str)
                    y = float(y_str)
                    points.append((x, y))
                except ValueError:
                    pass
                i += 2
                continue
            else:
                i += 1

    return points


# ---------------------------------------------------------------------------
# Image processing: cropping, binarization, normalization
# ---------------------------------------------------------------------------

def crop_word_from_page(
    page_img: Image.Image,
    polygon: List[Tuple[float, float]],
    pad: int = 3,
) -> Image.Image:
    """
    Crops a word from the page image based on a polygon.

    - Computes bounding box of the polygon.
    - Crops this box from the page image.
    - Creates a mask and sets everything outside the polygon to white.
    """
    # bounding box
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    min_x = int(max(0, np.floor(min(xs)) - pad))
    max_x = int(min(page_img.width, np.ceil(max(xs)) + pad))
    min_y = int(max(0, np.floor(min(ys)) - pad))
    max_y = int(min(page_img.height, np.ceil(max(ys)) + pad))

    # crop from page
    crop = page_img.crop((min_x, min_y, max_x, max_y))

    # mask with same region
    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)

    # polygon points relative to crop-box
    rel_poly = [(x - min_x, y - min_y) for (x, y) in polygon]
    draw.polygon(rel_poly, outline=255, fill=255)

    # apply mask: background white
    # grayscale mode, so 255 = white
    result = Image.new("L", crop.size, 255)
    result.paste(crop, mask=mask)

    return result


def binarize_image(img: Image.Image) -> Image.Image:
    """
    Very simple binarization via a global threshold.
    (Optionally you can later use Otsu or Sauvola.)
    """
    # ensure grayscale
    if img.mode != "L":
        img = img.convert("L")

    # numpy array
    arr = np.array(img, dtype=np.uint8)
    # simple threshold = mean
    thresh = arr.mean()
    bw = (arr <= thresh).astype(np.uint8) * 255  # text dark -> 255 (black), background 0 -> invert later?

    # Optional: invert so that text is black (0) and background white (255)
    # Here we invert to ensure background is white:
    bw = 255 - bw

    return Image.fromarray(bw, mode="L")


def normalize_height(img: Image.Image, target_height: int = TARGET_HEIGHT) -> Image.Image:
    """
    Scales the image such that height = target_height.
    Width is adjusted proportionally.
    """
    w, h = img.size
    if h == 0:
        return img  # safety

    scale = target_height / float(h)
    new_w = max(1, int(round(w * scale)))
    resized = img.resize((new_w, target_height), Image.BILINEAR)
    return resized


# ---------------------------------------------------------------------------
# Main function for step 1: build all word images for a split
# ---------------------------------------------------------------------------

def build_word_image_index(data_root: Path, split: str) -> Dict[str, np.ndarray]:
    """
    Builds a dict: word_id -> normalized word image (as numpy array),
    for all words in the documents of the given split (train or validation).

    1. Read train.tsv / validation.tsv -> doc_ids
    2. For each doc_id: parse locations/DDD.svg -> (word_id, polygon)
    3. Load page image: images/DDD.jpg
    4. For each word polygon:
       - crop word
       - binarize
       - normalize
       - store as numpy array
    """
    images_dir = data_root / "images"
    locations_dir = data_root / "locations"

    doc_ids = load_split_doc_ids(data_root, split)
    word_images: Dict[str, np.ndarray] = {}

    for doc_id in doc_ids:
        page_path = images_dir / f"{doc_id}.jpg"
        page_img = Image.open(page_path).convert("L")

        word_polys = parse_svg_paths_for_doc(locations_dir, doc_id)

        for word_id, polygon in word_polys:
            # safety: only words that actually belong to this document
            if not word_id.startswith(doc_id + "-"):
                continue

            word_img = crop_word_from_page(page_img, polygon)
            word_img = binarize_image(word_img)
            word_img = normalize_height(word_img, TARGET_HEIGHT)

            word_arr = np.array(word_img, dtype=np.uint8)
            word_images[word_id] = word_arr

    return word_images
