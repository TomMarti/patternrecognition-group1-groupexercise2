from typing import Dict
import numpy as np


def extract_features_for_word(
    word_img: np.ndarray,
    window_width: int = 1,
    step: int = 1,
) -> np.ndarray:
    """
    Extracts sliding-window features from a single word image.

    word_img: 2D array (H, W) with uint8, background = 255, text = 0 (from preprocess.py)
    window_width: width of the sliding window in pixels
    step: step size in pixels

    Return: feature matrix with shape (num_windows, num_features).
    """

    # Safety: only 2D image
    if word_img.ndim != 2:
        raise ValueError("word_img must be 2D (H, W).")

    H, W = word_img.shape
    if W < 1:
        # empty image -> empty feature matrix
        return np.zeros((0, 7), dtype=np.float32)

    # list for feature vectors
    feature_list = []

    # we store UC/LC of the previous column for gradients
    prev_uc_norm = 0.0
    prev_lc_norm = 0.0
    first_window = True

    # sliding window over width
    x = 0
    while x < W:
        # window region
        x_start = x
        x_end = min(W, x + window_width)

        window = word_img[:, x_start:x_end]  # shape (H, window_width)

        # 1) upper / lower contour
        uc_norm, lc_norm = _compute_upper_lower_contours(window)

        # 2) fraction of black pixels (in entire window)
        frac_black = _compute_fraction_black(window)

        # 3) fraction of black pixels between UC and LC
        frac_black_band = _compute_fraction_black_between_contours(window, uc_norm, lc_norm)

        # 4) black/white transitions (in middle column)
        transitions = _compute_vertical_transitions(window)

        # 5) gradients of UC and LC compared to previous position
        if first_window:
            d_uc = 0.0
            d_lc = 0.0
            first_window = False
        else:
            d_uc = uc_norm - prev_uc_norm
            d_lc = lc_norm - prev_lc_norm

        prev_uc_norm = uc_norm
        prev_lc_norm = lc_norm

        # assemble feature vector
        feature_vec = np.array(
            [uc_norm, lc_norm, frac_black, frac_black_band, transitions, d_uc, d_lc],
            dtype=np.float32,
        )
        feature_list.append(feature_vec)

        # next window
        x += step

    if not feature_list:
        return np.zeros((0, 7), dtype=np.float32)

    features = np.stack(feature_list, axis=0)
    return features


def _compute_upper_lower_contours(window: np.ndarray) -> tuple[float, float]:
    """
    Determines the upper (UC) and lower (LC) contour as normalized positions in [0, 1].
    Background: 255, text: 0 (black).
    """
    H, W = window.shape
    # mask: True where text (black)
    black_mask = window < 128  # or == 0, depending on how strict you want to be

    # rows that contain at least one black pixel
    rows_with_black = np.where(black_mask.any(axis=1))[0]

    if rows_with_black.size == 0:
        # no stroke in this window
        uc_norm = 1.0  # "upper contour very low"
        lc_norm = 1.0
    else:
        uc = rows_with_black[0]
        lc = rows_with_black[-1]
        uc_norm = uc / float(H)
        lc_norm = lc / float(H)

    return float(uc_norm), float(lc_norm)


def _compute_fraction_black(window: np.ndarray) -> float:
    """
    Fraction of black pixels (text) in the entire window.
    """
    H, W = window.shape
    total_pixels = H * W
    if total_pixels == 0:
        return 0.0

    black_pixels = np.sum(window < 128)
    return float(black_pixels) / float(total_pixels)


def _compute_fraction_black_between_contours(
    window: np.ndarray,
    uc_norm: float,
    lc_norm: float,
) -> float:
    """
    Fraction of black pixels between UC and LC.
    If no contour exists, 0.0.
    """
    H, W = window.shape
    if H == 0 or uc_norm >= 1.0 and lc_norm >= 1.0:
        return 0.0

    uc = int(round(uc_norm * H))
    lc = int(round(lc_norm * H))

    uc = max(0, min(H - 1, uc))
    lc = max(0, min(H - 1, lc))

    if lc <= uc:
        return 0.0

    band = window[uc:lc + 1, :]
    total_pixels = band.size
    if total_pixels == 0:
        return 0.0

    black_pixels = np.sum(band < 128)
    return float(black_pixels) / float(total_pixels)


def _compute_vertical_transitions(window: np.ndarray) -> float:
    """
    Counts black/white transitions along a vertical line.
    We take the middle column of the window.
    """
    H, W = window.shape
    if H <= 1:
        return 0.0

    col_idx = W // 2
    col = window[:, col_idx]

    # binarize: 1 = black, 0 = white
    is_black = (col < 128).astype(np.int32)

    # count transitions where the value changes
    diffs = np.abs(np.diff(is_black))
    transitions = np.sum(diffs)

    # optional: normalize (e.g., by max possible transitions = H-1)
    max_transitions = H - 1
    if max_transitions <= 0:
        return 0.0

    return float(transitions) / float(max_transitions)


# ---------------------------------------------------------------------------
# Build features for all words
# ---------------------------------------------------------------------------

def build_feature_index(
    word_images: Dict[str, np.ndarray],
    window_width: int = 1,
    step: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Takes the dict word_id -> word image (from preprocess.py)
    and returns a dict word_id -> feature matrix.

    Feature matrix: shape (num_windows, num_features)
    """
    features_index: Dict[str, np.ndarray] = {}

    for word_id, img in word_images.items():
        feats = extract_features_for_word(img, window_width=window_width, step=step)
        features_index[word_id] = feats

    return features_index
