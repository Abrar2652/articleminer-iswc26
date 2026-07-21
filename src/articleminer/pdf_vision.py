"""
pdf_vision.py - Render PDF pages as images for vision-based table extraction.

Uses PyMuPDF (fitz) for rendering. Designed for scale: smart page selection
sends only data-dense pages to the LLM vision API, minimizing cost.

Handles edge cases: landscape/rotated pages, multi-page table spans,
adaptive page count based on expected sample volume.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pdf_reader import PDFContent

logger = logging.getLogger(__name__)

# Lazy import — PyMuPDF is optional
_fitz = None


def _get_fitz():
    """Lazy-import PyMuPDF (fitz). Raises ImportError if not installed."""
    global _fitz
    if _fitz is None:
        import fitz
        _fitz = fitz
    return _fitz


def detect_page_orientations(pdf_path: str | Path) -> list[dict]:
    """Detect orientation and dimensions for every page in a PDF.

    Returns a list of dicts, one per page:
        {
            "page_index": int,
            "width": float,    # effective width in points after rotation
            "height": float,   # effective height in points after rotation
            "rotation": int,   # PDF /Rotate value (0, 90, 180, 270)
            "is_landscape": bool,  # True when width > height (page-level rotation)
        }

    Landscape pages are common in geochemistry papers for wide data tables
    with many element columns.  Text extraction often fails on these pages
    because pdfplumber/pdftext don't always handle the rotation, producing
    garbled or empty text — so they need special treatment.
    """
    try:
        fitz = _get_fitz()
    except ImportError:
        return []

    results = []
    try:
        doc = fitz.open(str(pdf_path))
        for i, page in enumerate(doc):
            # page.rect gives the effective rectangle after rotation
            rect = page.rect
            w, h = rect.width, rect.height
            results.append({
                "page_index": i,
                "width": w,
                "height": h,
                "rotation": page.rotation,
                "is_landscape": w > h * 1.15,  # 15% margin avoids near-square false positives
            })
        doc.close()
    except Exception as exc:
        logger.warning("Failed to detect page orientations: %s", exc)
    return results


def detect_content_rotated_pages(pdf_path: str | Path) -> set[int]:
    """Detect pages where the TEXT CONTENT is rotated within a portrait page.

    Many journals place landscape tables on portrait pages by rotating the
    table content 90°. The PDF page dimensions stay portrait, and
    page.rotation is 0, but the individual text characters have
    upright=False in pdfplumber.

    These pages are the #1 cause of text extraction failure for geochem
    papers — pdfplumber/pdftext extract garbled/reversed text, making
    text-based table parsing and page scoring impossible.

    Returns:
        Set of 0-based page indices with >50% rotated characters.
    """
    try:
        import pdfplumber
    except ImportError:
        return set()

    rotated_pages: set[int] = set()
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                chars = page.chars
                if not chars or len(chars) < 20:
                    continue
                n_rotated = sum(1 for c in chars if not c.get("upright", True))
                ratio = n_rotated / len(chars)
                if ratio > 0.5:
                    rotated_pages.add(i)
    except Exception as exc:
        logger.warning("Failed to detect content-rotated pages: %s", exc)

    if rotated_pages:
        logger.info(
            "Detected content-rotated pages (landscape table within portrait page): %s",
            sorted(rotated_pages),
        )
    return rotated_pages


def detect_text_rotation(pdf_path: str | Path, page_index: int) -> float:
    """Detect the page's text rotation and return the PRECISE degrees the
    RENDERED IMAGE must be rotated (clockwise) to bring text upright.

    NOT limited to 0/90/180/270: it returns the exact angle, so an arbitrarily
    rotated or skewed table (e.g. a 273.4-degree scan-skewed landscape table) is
    corrected precisely. Method:
      1. born-digital PDFs: take the per-line writing-direction vector (pymupdf
         'dir'), pick the dominant coarse orientation (0/90/180/270 by weight),
         then the length-weighted CIRCULAR MEAN of the exact angles within that
         orientation -- giving coarse rotation + fine skew in one number.
         Mixed-orientation pages (no >=50% majority) return 0.0 (de-rotating the
         whole page would flip the upright content -- needs per-region deskew).
      2. scanned pages (no text layer): fall back to a cv2 projection-profile
         skew estimate.
    Returns 0.0 when already upright or undetectable.
    """
    import collections
    import math
    try:
        fitz = _get_fitz()
    except ImportError:
        return 0.0
    try:
        doc = fitz.open(str(pdf_path))
        if page_index >= len(doc):
            doc.close()
            return 0.0
        lines = []  # (exact_angle_deg, weight)
        for b in doc[page_index].get_text("dict").get("blocks", []):
            for ln in b.get("lines", []):
                dx, dy = ln.get("dir", (1, 0))
                ang = math.degrees(math.atan2(dy, dx)) % 360
                w = sum(len(s.get("text", "")) for s in ln.get("spans", []))
                if w:
                    lines.append((ang, w))
        doc.close()
        if not lines:
            return _estimate_scan_skew(pdf_path, page_index)
        buckets = collections.Counter()
        for ang, w in lines:
            b = min((0, 90, 180, 270), key=lambda a: min((ang - a) % 360, (a - ang) % 360))
            buckets[b] += w
        dom, domw = buckets.most_common(1)[0]
        total = sum(w for _, w in lines)
        if total == 0 or domw / total < 0.5:
            return 0.0   # mixed orientations: not safe to de-rotate the whole page
        sx = sum(w * math.cos(math.radians(a)) for a, w in lines
                 if min((a - dom) % 360, (dom - a) % 360) <= 45)
        sy = sum(w * math.sin(math.radians(a)) for a, w in lines
                 if min((a - dom) % 360, (dom - a) % 360) <= 45)
        precise = math.degrees(math.atan2(sy, sx)) % 360
        corr = (360 - precise) % 360
        return 0.0 if min(corr, 360 - corr) < 0.5 else round(corr, 2)
    except Exception as exc:
        logger.warning("detect_text_rotation failed for page %d: %s", page_index, exc)
        return 0.0


def _estimate_scan_skew(pdf_path: str | Path, page_index: int, dpi: int = 150) -> float:
    """Best-effort skew estimate for a SCANNED (text-less) page via a
    projection-profile sweep: the correct deskew angle maximizes the variance of
    the horizontal pixel-row projection (text rows line up). Covers fine skew
    (+/-12 deg). Coarse 90-degree orientation of scans is left to the caller's
    landscape/aspect detection. Returns clockwise correction degrees."""
    try:
        import io
        import cv2
        import numpy as np
        from PIL import Image
        fitz = _get_fitz()
        page = fitz.open(str(pdf_path))[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0))
        gray = np.array(Image.open(io.BytesIO(pix.tobytes("png"))).convert("L"))
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        if thr.sum() < 255 * 50:
            return 0.0
        best_a, best_score = 0.0, -1.0
        h, w = thr.shape
        for a in np.arange(-12, 12.01, 0.5):
            M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
            rot = cv2.warpAffine(thr, M, (w, h), flags=cv2.INTER_NEAREST)
            proj = rot.sum(axis=1, dtype=np.float64)
            score = float(np.var(proj))
            if score > best_score:
                best_score, best_a = score, float(a)
        return 0.0 if abs(best_a) < 0.5 else round(-best_a % 360, 2)
    except Exception:
        return 0.0


def render_page_to_image(
    pdf_path: str | Path,
    page_index: int,
    dpi: int = 150,
    max_bytes: int = 1_500_000,
    rotate_deg: int = 0,
) -> tuple[bytes, str]:
    """Render a single PDF page to a PNG image.

    Args:
        pdf_path: Path to the PDF file.
        page_index: 0-based page index.
        dpi: Rendering resolution. Auto-reduced if image exceeds max_bytes.
        max_bytes: Maximum image size in bytes (default 1.5MB — increased
            from 1MB to accommodate landscape pages which are wider).
        rotate_deg: If non-zero, de-rotate the rendered image by this many
            degrees clockwise (used to bring rotated-table content upright for
            vision — see detect_text_rotation). Default 0 = unchanged behavior.

    Returns:
        (image_bytes, media_type) tuple. Returns (b"", "") on failure.
    """
    try:
        fitz = _get_fitz()
    except ImportError:
        logger.warning("PyMuPDF (fitz) not installed — cannot render PDF pages")
        return b"", ""

    def _maybe_deskew(image_bytes: bytes) -> bytes:
        if not rotate_deg:
            return image_bytes
        try:
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes)).rotate(-rotate_deg, expand=True)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return buf.getvalue()
        except Exception as exc:
            logger.warning("deskew rotate failed (page %d): %s", page_index, exc)
            return image_bytes

    try:
        doc = fitz.open(str(pdf_path))
        if page_index >= len(doc):
            doc.close()
            return b"", ""

        page = doc[page_index]

        # Try progressively lower DPI until image fits within max_bytes
        for try_dpi in [dpi, 120, 100, 72]:
            zoom = try_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            image_bytes = _maybe_deskew(pix.tobytes("png"))
            if len(image_bytes) <= max_bytes:
                doc.close()
                return image_bytes, "image/png"

        # Even at 72 DPI it's too large — return it anyway
        doc.close()
        return image_bytes, "image/png"

    except Exception as exc:
        logger.warning("Failed to render page %d of %s: %s", page_index, pdf_path, exc)
        return b"", ""


def get_data_page_indices(
    content: "PDFContent",
    pdf_path: str | Path | None = None,
    max_pages: int = 5,
    expected_samples: int = 0,
) -> list[int]:
    """Select page indices most likely to contain geochemical data tables.

    Uses a multi-signal approach (priority order):
    1. Content-rotated pages (landscape tables within portrait pages —
       these are invisible to text scoring but contain critical data)
    2. Page-level landscape detection (width > height)
    3. Text-based scoring heuristic (element symbols + numeric density)
    4. Adaptive page budget based on expected sample count

    Args:
        content: PDFContent with per-page text already extracted.
        pdf_path: Path to PDF for orientation detection (optional).
        max_pages: Maximum number of pages to return.
        expected_samples: Expected sample count from paper intelligence.
            Used to adapt the number of pages rendered.

    Returns:
        List of 0-based page indices, sorted by page order.
    """
    from .pdf_reader import score_pages_for_data

    # Adaptive page budget: more expected samples → more pages
    if expected_samples > 200:
        max_pages = max(max_pages, 15)
    elif expected_samples > 100:
        max_pages = max(max_pages, 12)
    elif expected_samples > 50:
        max_pages = max(max_pages, 10)
    elif expected_samples > 20:
        max_pages = max(max_pages, 8)

    scored = score_pages_for_data(content)

    # Detect pages with rotated content (landscape tables in portrait pages).
    # These are HIGHEST PRIORITY — text extraction produces garbled output
    # so text-based scoring completely misses them.
    content_rotated_indices: set[int] = set()
    landscape_indices: set[int] = set()
    if pdf_path:
        content_rotated_indices = detect_content_rotated_pages(pdf_path)

        orientations = detect_page_orientations(pdf_path)
        for info in orientations:
            if info["is_landscape"]:
                landscape_indices.add(info["page_index"])
        if landscape_indices:
            logger.info("Detected page-level landscape: %s", sorted(landscape_indices))

    # All rotated/landscape pages (text extraction fails on these)
    must_include = content_rotated_indices | landscape_indices

    # Build combined set: must-include first, then scored pages
    combined = set(must_include)

    # Add scored pages up to budget
    for idx, _score in scored:
        if len(combined) >= max_pages:
            break
        combined.add(idx)

    if combined:
        return sorted(combined)

    # Fallback: select pages from the second half of the document
    # (tables are typically in results/appendix sections)
    total = len(content.pages)
    if total == 0:
        return []
    start = max(total // 2, 0)
    fallback = list(range(start, min(start + max_pages, total)))
    return fallback


def render_data_pages(
    pdf_path: str | Path,
    content: "PDFContent",
    max_pages: int = 5,
    dpi: int = 150,
    expected_samples: int = 0,
    deskew: bool | None = None,
) -> list[dict]:
    """Render the most data-dense PDF pages as images.

    Combines smart page selection (text scoring + landscape detection)
    with image rendering.

    Each result dict includes:
        page_index: 0-based index
        image_bytes: PNG bytes
        media_type: "image/png"
        is_landscape: whether the page is landscape-oriented

    Args:
        pdf_path: Path to the PDF file.
        content: PDFContent with per-page text already extracted.
        max_pages: Maximum number of pages to render.
        dpi: Rendering resolution.
        expected_samples: Expected sample count for adaptive budget.

    Returns:
        List of dicts. Empty list if PyMuPDF is not installed or rendering fails.
    """
    try:
        _get_fitz()
    except ImportError:
        logger.warning("PyMuPDF (fitz) not installed — skipping vision extraction")
        return []

    indices = get_data_page_indices(
        content,
        pdf_path=pdf_path,
        max_pages=max_pages,
        expected_samples=expected_samples,
    )
    if not indices:
        return []

    # Get orientation info for labeling — both page-level AND content-level
    orientations = detect_page_orientations(pdf_path)
    page_landscape_set = {
        info["page_index"]
        for info in orientations
        if info["is_landscape"]
    }
    content_rotated_set = detect_content_rotated_pages(pdf_path)
    # Both types are "landscape" for vision purposes
    landscape_set = page_landscape_set | content_rotated_set

    logger.info(
        "Rendering %d data-dense pages: %s (landscape/rotated: %s)",
        len(indices), indices,
        sorted(landscape_set & set(indices)) or "none",
    )

    # Opt-in de-rotation: bring content-rotated pages upright before vision.
    # Default OFF (deskew=None -> read ARTICLEMINER_DESKEW env) so existing
    # behavior and benchmark numbers are unchanged until explicitly enabled.
    if deskew is None:
        deskew = os.environ.get("ARTICLEMINER_DESKEW", "").lower() in ("1", "true", "yes")

    results = []
    for idx in indices:
        # Landscape/rotated pages benefit from slightly higher DPI
        page_dpi = dpi + 30 if idx in landscape_set else dpi
        # De-rotate only content-rotated pages (rotated text within a portrait
        # page); page-level landscape is already read correctly by vision.
        rotate_deg = 0
        if deskew and idx in content_rotated_set:
            rotate_deg = detect_text_rotation(pdf_path, idx)
            if rotate_deg:
                logger.info("Deskew: de-rotating content-rotated page %d by %d deg",
                            idx, rotate_deg)
        image_bytes, media_type = render_page_to_image(
            pdf_path, idx, dpi=page_dpi, rotate_deg=rotate_deg)
        if image_bytes:
            results.append({
                "page_index": idx,
                "image_bytes": image_bytes,
                "media_type": media_type,
                "is_landscape": idx in landscape_set,
                "deskew_applied": rotate_deg,
            })

    logger.info(
        "Rendered %d/%d pages (total %.1f KB)",
        len(results), len(indices),
        sum(len(r["image_bytes"]) for r in results) / 1024,
    )
    return results


def has_pdfplumber_tables(content: "PDFContent", min_tables: int = 1) -> bool:
    """Check if pdfplumber extracted any structured tables from the PDF."""
    return len(content.tables_text) >= min_tables


# ──────────────────────────────────────────────────────────────────────────────
# Marker-based PDF extraction (layout-aware, handles rotated content)
# ──────────────────────────────────────────────────────────────────────────────

_marker_converter = None
_marker_available: bool | None = None


class _DummyRecognitionModel:
    """Stub replacing the real recognition model that segfaults on
    torch 2.10 + Python 3.14.  Absorbs attribute writes (e.g.
    ``disable_tqdm = True``) and returns empty results when called,
    so processors that reference it don't crash.  For digital PDFs
    with disable_ocr=True, the real model is never actually needed."""

    def __setattr__(self, name, value):
        pass  # absorb e.g. disable_tqdm = True

    def __getattr__(self, name):
        return None

    def __call__(self, *args, **kwargs):
        # Return empty list — matches expected List[OCRResult] signature
        return []


def _get_marker_converter():
    """Lazy-init the Marker PDF converter.

    Skips the recognition model (segfaults on torch 2.10 + Python 3.14)
    and disables OCR — both are unnecessary for digital PDFs where text
    is already embedded.

    Returns None if marker-pdf is not installed or fails to init.
    """
    global _marker_converter, _marker_available

    if _marker_available is False:
        return None
    if _marker_converter is not None:
        return _marker_converter

    try:
        from marker.models import (
            LayoutPredictor, FoundationPredictor,
            TableRecPredictor, DetectionPredictor, OCRErrorPredictor,
        )
        from surya.settings import settings as surya_settings
        from marker.converters.pdf import PdfConverter

        artifact_dict = {
            "layout_model": LayoutPredictor(
                FoundationPredictor(checkpoint=surya_settings.LAYOUT_MODEL_CHECKPOINT)
            ),
            "recognition_model": _DummyRecognitionModel(),
            "table_rec_model": TableRecPredictor(),
            "detection_model": DetectionPredictor(),
            "ocr_error_model": OCRErrorPredictor(),
        }

        # Exclude EquationProcessor — it calls recognition_model for LaTeX
        # which we don't need. TableProcessor also references it but only
        # in assign_ocr_lines which is guarded by needs_ocr() returning
        # empty for digital PDFs; _DummyRecognitionModel handles the rest.
        _SKIP_PROCESSORS = {"EquationProcessor"}
        processor_list = [
            f"{p.__module__}.{p.__name__}"
            for p in PdfConverter.default_processors
            if p.__name__ not in _SKIP_PROCESSORS
        ]

        _marker_converter = PdfConverter(
            artifact_dict=artifact_dict,
            processor_list=processor_list,
            config={"disable_ocr": True},
        )
        _marker_available = True
        logger.info("Marker PDF converter initialized (OCR disabled — digital PDF mode)")
        return _marker_converter

    except ImportError:
        logger.debug("marker-pdf not installed — skipping Marker backend")
        _marker_available = False
        return None
    except Exception as exc:
        logger.warning("Failed to initialize Marker converter: %s", exc)
        _marker_available = False
        return None


def extract_tables_with_marker(
    pdf_path: str | Path,
    page_indices: list[int] | None = None,
) -> list[tuple[str, int]]:
    """Extract tables from a PDF using Marker's layout-aware pipeline.

    Marker uses ML models for layout detection + table structure recognition,
    correctly handling rotated content, multi-column layouts, and complex
    table structures that pdftext/pdfplumber miss.

    Args:
        pdf_path: Path to the PDF file.
        page_indices: If provided, only extract from these 0-based page indices.
            If None, processes all pages.

    Returns:
        List of (markdown_table_text, page_index) tuples.
        Empty list if Marker is not available or extraction fails.
    """
    converter = _get_marker_converter()
    if converter is None:
        return []

    try:
        import re as _re

        config_overrides = {}
        if page_indices is not None:
            # Marker uses 0-based page ranges
            config_overrides["page_range"] = page_indices

        if config_overrides:
            # Update converter config for this run
            for k, v in config_overrides.items():
                converter.config[k] = v

        result = converter(str(Path(pdf_path).resolve()))
        md = result.markdown

        # Extract table blocks from markdown
        # Marker outputs tables as markdown pipe tables
        tables: list[tuple[str, int]] = []
        current_table_lines: list[str] = []
        # Track approximate page by counting page breaks or sections
        # Marker doesn't give per-page info easily, so we return page_index=0
        # for all tables when page mapping isn't available
        for line in md.split("\n"):
            if "|" in line and line.strip().startswith("|"):
                current_table_lines.append(line)
            else:
                if len(current_table_lines) >= 3:  # header + separator + at least 1 row
                    table_text = "\n".join(current_table_lines)
                    tables.append((table_text, 0))
                current_table_lines = []

        # Don't forget last table
        if len(current_table_lines) >= 3:
            table_text = "\n".join(current_table_lines)
            tables.append((table_text, 0))

        if tables:
            logger.info(
                "Marker extracted %d tables from %s (%.1f KB markdown)",
                len(tables), Path(pdf_path).name, len(md) / 1024,
            )
        else:
            logger.debug("Marker produced no tables from %s", Path(pdf_path).name)

        return tables

    except Exception as exc:
        logger.warning("Marker extraction failed for %s: %s", pdf_path, exc)
        return []


def marker_tables_to_dataframes(
    tables: list[tuple[str, int]],
) -> list[tuple["pd.DataFrame", int]]:
    """Convert Marker markdown tables to pandas DataFrames.

    Args:
        tables: List of (markdown_table_text, page_index) from extract_tables_with_marker.

    Returns:
        List of (DataFrame, page_index) tuples. Skips tables that fail to parse.
    """
    try:
        import pandas as pd
    except ImportError:
        return []

    import io

    results = []
    for table_md, page_idx in tables:
        try:
            lines = [l.strip() for l in table_md.strip().split("\n") if l.strip()]
            if len(lines) < 3:
                continue

            # Parse markdown pipe table
            # Line 0: header, Line 1: separator (---|---), Lines 2+: data
            def parse_row(line: str) -> list[str]:
                cells = [c.strip() for c in line.split("|")]
                # Remove empty first/last from leading/trailing |
                if cells and cells[0] == "":
                    cells = cells[1:]
                if cells and cells[-1] == "":
                    cells = cells[:-1]
                return cells

            header = parse_row(lines[0])
            # Skip separator line
            data_rows = [parse_row(l) for l in lines[2:]]

            if not header or not data_rows:
                continue

            # Pad/truncate rows to header length
            n_cols = len(header)
            data_rows = [
                row[:n_cols] + [""] * max(0, n_cols - len(row))
                for row in data_rows
            ]

            df = pd.DataFrame(data_rows, columns=header)
            if len(df) > 0:
                results.append((df, page_idx))
        except Exception:
            continue

    return results
