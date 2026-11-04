"""Lightweight image-processing engines used by the agent's tools."""

from __future__ import annotations

import io
import logging
import os
import tempfile
from typing import Optional, Union

import numpy as np
from PIL import Image

from . import image_io


from . import config


logger = logging.getLogger(__name__)


try:  # OpenCV is optional; image enhancement degrades to no-op without it.
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    cv2 = None  # type: ignore[assignment]
    CV2_AVAILABLE = False


class ImageToolEngine:
    """Crop and OCR helper backed by PIL and the layout-parsing API."""

    def __init__(self) -> None:
        self.current_image: Optional[Image.Image] = None

    def load_image(self, source: Union[str, bytes, Image.Image]):
        if isinstance(source, Image.Image):
            self.current_image = source
            return self
        if isinstance(source, bytes):
            self.current_image = Image.open(io.BytesIO(source))
            return self
        if isinstance(source, str):
            if image_io.looks_like_base64(source):
                raise ValueError(
                    "Received a base64 string where a path was expected."
                )
            if os.path.exists(source):
                self.current_image = Image.open(source)
                return self
            raise FileNotFoundError(f"Image file not found: {source}")
        raise TypeError(f"Unsupported image source: {type(source)!r}")

    def crop(self, x: int, y: int, width: int, height: int) -> Image.Image:
        if self.current_image is None:
            raise RuntimeError("No image loaded.")
        box = (x, y, x + width, y + height)
        cropped = self.current_image.crop(box)
        self.current_image = cropped
        return cropped

    def save_current(self, path: str) -> None:
        if self.current_image is not None:
            self.current_image.save(path)

    def ocr_via_layout_api(self, parser_callable) -> dict:
        """Save the current image to a temp file and call ``parser_callable``.

        The callable should accept a path and return a dict with at least
        ``rec_texts``, ``formatted_text`` and ``blocks``. We isolate the
        HTTP call here so the engine does not depend on the network module.
        """

        if self.current_image is None:
            raise RuntimeError("No image loaded.")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            self.current_image.save(tmp.name)
            tmp_path = tmp.name
        try:
            return parser_callable(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class ImageEnhancementEngine:
    """OpenCV-based perspective correction, super resolution and sharpen."""

    def __init__(self) -> None:
        self.current_image: Optional[np.ndarray] = None

    @staticmethod
    def available() -> bool:
        return CV2_AVAILABLE

    def load_image(self, source: Union[str, bytes, Image.Image, np.ndarray]):
        if not CV2_AVAILABLE:
            raise RuntimeError(
                "OpenCV is not installed; image enhancement tools are disabled."
            )
        if isinstance(source, np.ndarray):
            self.current_image = source
            return self
        if isinstance(source, Image.Image):
            self.current_image = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
            return self
        if isinstance(source, bytes):
            pil = Image.open(io.BytesIO(source))
            self.current_image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            return self
        if isinstance(source, str):
            if image_io.looks_like_base64(source):
                raise ValueError(
                    "Received a base64 string where a path was expected."
                )
            if not os.path.exists(source):
                raise FileNotFoundError(f"Image file not found: {source}")
            buf = np.fromfile(source, dtype=np.uint8)
            self.current_image = cv2.imdecode(buf, -1)
            return self
        raise TypeError(f"Unsupported image source: {type(source)!r}")

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def auto_correct_perspective(self) -> Optional[np.ndarray]:
        if self.current_image is None:
            return None
        if (
            self.current_image.ndim == 3
            and self.current_image.shape[2] == 4
        ):
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        contours, _ = cv2.findContours(
            edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        screen_cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screen_cnt = approx
                break
        if screen_cnt is None:
            return self.current_image
        pts = screen_cnt.reshape(4, 2)
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        max_width = max(
            int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl))
        )
        max_height = max(
            int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl))
        )
        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(
            self.current_image, matrix, (max_width, max_height)
        )
        self.current_image = warped
        return warped

    def apply_super_resolution(
        self, model_path: str = "EDSR_x4.pb", scale: int = 4
    ) -> Optional[np.ndarray]:
        """EDSR x4 on CPU. Raises on failure so a no-op is never reported as success.

        Input is first capped to ``SR_MAX_INPUT_EDGE``; only x4 weights exist,
        so scale 2/3 are x4 plus an area downscale.
        """
        if self.current_image is None:
            return None
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"EDSR weights not found at {model_path!r} (set SR_MODEL_PATH)"
            )
        image = self.current_image
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        scale = scale if scale in (2, 3, 4) else 4
        h, w = image.shape[:2]
        if max(h, w) > config.SR_MAX_INPUT_EDGE:
            factor = config.SR_MAX_INPUT_EDGE / max(h, w)
            image = cv2.resize(
                image,
                (max(1, int(w * factor)), max(1, int(h * factor))),
                interpolation=cv2.INTER_AREA,
            )
            h, w = image.shape[:2]
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("edsr", 4)
        out = sr.upsample(image)
        if scale != 4:
            out = cv2.resize(out, (w * scale, h * scale), interpolation=cv2.INTER_AREA)
        self.current_image = out
        return self.current_image

    def enhance_sharpness(self, amount: float = 1.5) -> Optional[np.ndarray]:
        if self.current_image is None:
            return None
        blurred = cv2.GaussianBlur(self.current_image, (0, 0), 3)
        sharpened = cv2.addWeighted(
            self.current_image, 1.0 + amount, blurred, -amount, 0
        )
        self.current_image = sharpened
        return sharpened

    def to_pil(self) -> Optional[Image.Image]:
        if self.current_image is None:
            return None
        rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
