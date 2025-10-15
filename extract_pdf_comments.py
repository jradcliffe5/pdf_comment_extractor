#!/usr/bin/env python3
"""
extract_pdf_comments.py

Extracts PDF annotations (comments/highlights/etc.) to CSV/JSON with page, location,
and an estimated text line number.

Requires: PyMuPDF (fitz)
    pip install pymupdf
"""

import argparse
import csv
import json
import re
import sys
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, cast

import fitz  # PyMuPDF


@dataclass
class LineInfo:
    idx: int  # 1-based line number on the page (fallback ordering)
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    text: str
    pdf_number: Optional[int] = None  # explicit line number printed in the PDF if detected


@dataclass
class AnnotRecord:
    page: Optional[int]  # 1-based in PDF; None for manual entries
    type: str
    author: Optional[str]
    comment_text: Optional[str]
    quoted_text: Optional[str]
    line_number: Optional[int]
    context_line_text: Optional[str]
    bbox: Tuple[float, float, float, float]


def _page_get_text(page: fitz.Page, mode: str, **kwargs: Any) -> Any:
    """Compatibility wrapper for Page.get_text across PyMuPDF versions."""
    page_any = cast(Any, page)
    if hasattr(page_any, "get_text"):
        return page_any.get_text(mode, **kwargs)

    clip = kwargs.get("clip")
    textpage = page.get_textpage(clip=clip)
    mode_lower = mode.lower()
    if mode_lower == "dict":
        return textpage.extractDICT()
    if mode_lower == "text":
        return textpage.extractTEXT()
    raise ValueError(f"Unsupported text extraction mode: {mode}")


def _flatten_lines(page: fitz.Page) -> List[LineInfo]:
    """
    Return a reading-order list of line boxes and text for a page.
    Uses page.get_text('dict') to access blocks/lines with bounding boxes.

    Attempts to detect explicit line numbers printed in the PDF so we can
    propagate those instead of synthetic ordering when possible.
    """
    raw = _page_get_text(page, "dict")
    lines: List[LineInfo] = []
    markers: List[Tuple[float, int]] = []  # (vertical center, line number)
    # We'll remap explicit line-number-only blocks onto nearby text lines.
    line_idx = 1

    # Iterate blocks -> lines (skip images)
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue  # not text

        for line in block.get("lines", []):
            # Concatenate span texts in a line
            txt = "".join(span.get("text", "") for span in line.get("spans", []))
            bbox = tuple(line.get("bbox", [0, 0, 0, 0]))  # x0, y0, x1, y1
            # Normalize whitespace
            txt_norm = " ".join(txt.split())

            if not txt_norm:
                continue

            detected_number: Optional[int] = None
            # Check for a leading integer (common line-number format)
            prefix_len = 0
            while prefix_len < len(txt_norm) and txt_norm[prefix_len].isdigit():
                prefix_len += 1

            if prefix_len > 0 and (prefix_len == len(txt_norm) or txt_norm[prefix_len].isspace()):
                try:
                    detected_number = int(txt_norm[:prefix_len])
                except ValueError:
                    detected_number = None

                remainder = txt_norm[prefix_len:].lstrip()
                if remainder:
                    txt_norm = remainder
                else:
                    if detected_number is not None:
                        # Pure number line (likely the printed line-number in margin)
                        y_center = (bbox[1] + bbox[3]) / 2.0
                        markers.append((y_center, detected_number))
                        continue
                    # Otherwise leave txt_norm as-is to avoid losing numeric-only content

            lines.append(LineInfo(idx=line_idx, bbox=bbox, text=txt_norm, pdf_number=detected_number))
            line_idx += 1

    if markers:
        # Reassign detected margin line numbers to the closest text lines.
        markers.sort(key=lambda item: item[0])
        remaining_markers = list(markers)
        for ln in lines:
            if ln.pdf_number is not None:
                continue

            if not remaining_markers:
                break

            y_center = (ln.bbox[1] + ln.bbox[3]) / 2.0
            line_height = max(ln.bbox[3] - ln.bbox[1], 1.0)

            best_idx = None
            best_delta = None
            for idx, (marker_y, marker_value) in enumerate(remaining_markers):
                delta = abs(marker_y - y_center)
                if best_delta is None or delta < best_delta:
                    best_idx = idx
                    best_delta = delta

            if best_idx is not None and best_delta is not None and best_delta <= line_height:
                _, marker_value = remaining_markers.pop(best_idx)
                ln.pdf_number = marker_value

    return lines


def _rect_from_quads(quads: List[fitz.Quad]) -> fitz.Rect:
    """
    Compute a bounding rectangle that covers all quad points.
    """
    # Union all quad rectangles to cover multi-span highlights.
    rect = None
    for q in quads:
        r = q.rect
        rect = r if rect is None else rect | r
    return rect if rect is not None else fitz.Rect(0, 0, 0, 0)


def _best_line_for_rect(lines: List[LineInfo], rect: fitz.Rect) -> Optional[LineInfo]:
    """
    Find the line that best matches a rectangle:
    - Prefer maximum vertical overlap (IoU-ish on Y),
    - Break ties with horizontal overlap, then nearest center distance.
    """
    best = None
    best_score = -1.0

    def overlap_1d(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))

    for ln in lines:
        x0, y0, x1, y1 = ln.bbox
        ly0, ly1 = y0, y1

        # Score vertical overlap first so we align to the right text row.
        vy = overlap_1d(rect.y0, rect.y1, ly0, ly1)
        vh = rect.y1 - rect.y0
        lh = ly1 - ly0
        denom = max(vh, lh, 1e-6)
        vy_iou = vy / denom  # normalized vertical overlap

        # Horizontal overlap helps separate adjacent columns.
        hx = overlap_1d(rect.x0, rect.x1, x0, x1)
        wx = rect.x1 - rect.x0
        lx = x1 - x0
        denomx = max(wx, lx, 1e-6)
        hx_iou = hx / denomx

        # score: prioritize vertical overlap, then horizontal
        score = vy_iou * 0.8 + hx_iou * 0.2

        if score > best_score:
            best_score = score
            best = ln

    return best


def _annotation_type(annot: fitz.Annot) -> str:
    """Normalize annotation subtype to a friendly label."""
    sub = (annot.type[1] if isinstance(annot.type, tuple) else str(annot.type)) or ""
    sub = sub.upper()
    # Map PyMuPDF subtype codes to human readable tokens.
    mapping = {
        "TEXT": "Text",
        "HIGHLIGHT": "Highlight",
        "UNDERLINE": "Underline",
        "SQUIGGLY": "Squiggly",
        "STRIKEOUT": "Strikeout",
        "FREETEXT": "FreeText",
        "INK": "Ink",
        "STAMP": "Stamp",
        "POLYGON": "Polygon",
        "POLYLINE": "Polyline",
        "SQUARE": "Square",
        "CIRCLE": "Circle",
        "LINE": "Line",
        "CARETS": "Caret",
        "FILEATTACH": "FileAttachment",
        "REDACT": "Redact",
    }
    return mapping.get(sub, sub.title() or "Unknown")


def _extract_popup_text(annot: fitz.Annot) -> Optional[str]:
    """
    Get the comment / popup content for the annotation if present.
    In PyMuPDF: annot.info.get('content') usually holds the text.
    """
    # Favor content, but fall back to subject for old-style PDFs.
    info = annot.info or {}
    raw = info.get("content") or info.get("subject")
    if not raw:
        return None
    txt = " ".join(str(raw).split())
    return txt if txt else None


def _extract_author(annot: fitz.Annot) -> Optional[str]:
    # Pull author from either title or name metadata.
    info = annot.info or {}
    author = info.get("title") or info.get("name")
    if not author:
        return None
    author = " ".join(str(author).split())
    return author if author else None


def _strip_margin_numbers(text: str) -> str:
    """
    Remove standalone page/line numbers that often get captured at the edges of highlights.
    Only trims numeric tokens at the boundaries so inline numeric content is preserved.
    """

    def is_margin_number(token: str) -> bool:
        stripped = token.strip("()[]:;,.")
        if not stripped:
            return False
        if stripped.isdigit():
            return True
        if "-" in stripped:
            parts = stripped.split("-")
            if all(part.isdigit() for part in parts if part):
                return True
        return False

    tokens = text.split()
    start = 0
    end = len(tokens)

    while start < end and is_margin_number(tokens[start]):
        start += 1
    while end > start and is_margin_number(tokens[end - 1]):
        end -= 1

    cleaned = tokens[start:end]
    return " ".join(cleaned)


def _normalize_snippet(text: str) -> str:
    """
    Normalize raw text extracted from highlight regions:
    - collapse line breaks,
    - stitch hyphenated line breaks,
    - strip stray margin numbers per line and at boundaries.
    """
    if not text:
        return ""
    cleaned = text.replace("\u00ad", "")  # soft hyphen
    cleaned = cleaned.replace("\r\n", "\n")
    # Drop standalone line/page numbers that sit on their own lines.
    cleaned = re.sub(r"^\s*\d{1,4}\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", cleaned)
    # Handle hyphenated breaks that had a numeric line number in between.
    cleaned = re.sub(r"(\w)-\s*\n\s*\d{1,4}\s*\n\s*(\w)", r"\1\2", cleaned)

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    filtered: List[str] = []
    for line in lines:
        if re.fullmatch(r"\d{1,4}", line):
            continue
        filtered.append(line)

    segments: List[str] = []
    carry: Optional[str] = None
    pending_symbols: List[str] = []
    for line in filtered:
        if carry is not None:
            if re.fullmatch(r"[^\w]+", line):
                pending_symbols.append(line)
                continue
            line = carry + line
            carry = None
        if line.endswith("-"):
            carry = line[:-1]
            continue
        if re.fullmatch(r"[^\w]+", line):
            segments.append(line)
            continue
        segments.append(line)
        if pending_symbols:
            segments.extend(pending_symbols)
            pending_symbols.clear()
    if carry:
        segments.append(carry)
    if pending_symbols:
        segments.extend(pending_symbols)

    cleaned = " ".join(segments)
    cleaned = " ".join(cleaned.split())
    cleaned = _strip_margin_numbers(cleaned)
    return cleaned


def _extract_quoted_text_for_quads(page: fitz.Page, quads: List[fitz.Quad]) -> str:
    """
    For highlight/underline/etc., extract the text under each quad and join.
    """
    raw_snippets = []
    for q in quads:
        r = q.rect
        t = _page_get_text(page, "text", clip=r)  # raw text within the clip
        if not t:
            continue
        t = t.strip()
        if not t:
            continue
        raw_snippets.append(t)
    # Deduplicate while keeping order
    seen = set()
    ordered: List[str] = []
    for snippet in raw_snippets:
        if snippet not in seen:
            ordered.append(snippet)
            seen.add(snippet)
    combined = "\n".join(ordered)
    return _normalize_snippet(combined)


def _quads_from_annot(annot: fitz.Annot) -> List[fitz.Quad]:
    """
    Return a list of fitz.Quad for annotation if it has vertices (highlight, etc.).
    """
    quads = []
    try:
        v = annot.vertices
        if not v:
            return []
        # Vertices arrive flat; regroup them in sets of four for each quad.
        pts = [fitz.Point(p) for p in v]
        for i in range(0, len(pts), 4):
            quad = fitz.Quad(pts[i], pts[i + 1], pts[i + 2], pts[i + 3])
            quads.append(quad)
    except Exception:
        pass
    return quads


def extract_annotations(doc: fitz.Document) -> List[AnnotRecord]:
    records: List[AnnotRecord] = []

    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        lines = _flatten_lines(page)

        # Walk every annotation on the page and capture its metadata.
        annot = page.first_annot
        while annot:
            a_type = _annotation_type(annot)
            author = _extract_author(annot)
            comment = _extract_popup_text(annot)

            quoted_text = None
            rect = annot.rect

            # For highlight-style annots, use quads to both locate and extract the quoted text
            if a_type in {"Highlight", "Underline", "Strikeout", "Squiggly"}:
                quads = _quads_from_annot(annot)
                if quads:
                    rect = _rect_from_quads(quads)
                    quoted_text = _extract_quoted_text_for_quads(page, quads)
                else:
                    # fallback to rect clip
                    qt = _page_get_text(page, "text", clip=rect)
                    if qt:
                        qt_norm = _normalize_snippet(qt)
                        quoted_text = qt_norm if qt_norm else None
                    else:
                        quoted_text = None

            # Find best-matching line for the annotation rectangle
            best_line = _best_line_for_rect(lines, rect) if lines else None
            if best_line:
                line_num = best_line.pdf_number if best_line.pdf_number is not None else best_line.idx
            else:
                line_num = None
            context_text = best_line.text if best_line else None

            rec = AnnotRecord(
                page=pno + 1,
                type=a_type,
                author=author,
                comment_text=comment,
                quoted_text=quoted_text,
                line_number=line_num,
                context_line_text=context_text,
                bbox=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
            )
            records.append(rec)

            annot = annot.next

    return records


def write_csv(path: str, records: List[AnnotRecord]) -> None:
    fieldnames = [
        "page",
        "type",
        "author",
        "comment_text",
        "quoted_text",
        "line_number",
        "context_line_text",
        "bbox",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = asdict(r)
            # pretty-print bbox
            row["bbox"] = ",".join(f"{v:.2f}" for v in r.bbox)
            # CSV library handles quoting automatically here.
            writer.writerow(row)


def write_json(path: str, records: List[AnnotRecord]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)


def _quote_for_display(text: str) -> str:
    # Quote text in a stable way for single-line referee bullets.
    sanitized = text.replace('"', "'")
    return f'"{sanitized}"'


def _report_line_for_record(record: AnnotRecord) -> str:
    location_parts: List[str] = []
    if record.page is not None:
        location_parts.append(f"Page {record.page}")
    if record.line_number is not None:
        location_parts.append(f"line {record.line_number}")
    location = ", ".join(location_parts)

    def normalize(text: Optional[str]) -> str:
        return " ".join(text.split()) if text else ""

    comment = normalize(record.comment_text)
    quote = normalize(record.quoted_text)
    context = normalize(record.context_line_text)

    quote_display = _quote_for_display(quote) if quote else ""
    annot_type = record.type or "Annotation"

    body: str
    if annot_type == "Strikeout" and quote:
        body = f"Remove {quote_display}"
        if comment:
            body = f"{body}: {comment}"
    elif annot_type == "Highlight":
        if comment:
            body = comment
            if quote and quote_display not in comment:
                body = f"{body} (Quote: {quote_display})"
        elif quote:
            body = quote
        elif context:
            body = context
        else:
            body = "(No annotation text)"
    elif annot_type == "Underline" and quote:
        body = f"Underline {quote_display}"
        if comment:
            body = f"{body}: {comment}"
    elif annot_type == "Squiggly" and quote:
        body = f"Reconsider {quote_display}"
        if comment:
            body = f"{body}: {comment}"
    elif annot_type == "Manual":
        body = comment or context or "(No annotation text)"
    elif annot_type == "Caret":
        if comment:
            body = f"Insert {_quote_for_display(comment)}"
            if quote:
                body = f"{body} (Context: {quote_display})"
        elif quote:
            body = f"Insert {quote_display}"
        elif context:
            body = f"Insert text near: {context}"
        else:
            body = "Insert text"
    elif annot_type in {"Text", "FreeText"}:
        body = comment or context or "(No annotation text)"
    else:
        if comment and quote:
            body = f"{annot_type}: {comment} (Quote: {quote_display})"
        elif comment:
            body = f"{annot_type}: {comment}" if annot_type else comment
        elif quote:
            body = f"{annot_type}: {quote_display}" if annot_type else quote_display
        elif context:
            body = context
        else:
            body = "(No annotation text)"

    body = body.strip()

    return f"{location}: {body}" if location else body


def write_text_report(path: Optional[str], records: List[AnnotRecord]) -> None:
    # Collapse each annotation into a single referee-style line.
    lines = [_report_line_for_record(r) for r in records]

    if path:
        with open(path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n\n".join(lines))
                f.write("\n")
    else:
        for idx, line in enumerate(lines):
            if idx:
                sys.stdout.write("\n")
            sys.stdout.write(f"{line}\n")


def _split_manual_blocks(text: str) -> List[List[str]]:
    blocks: List[List[str]] = []
    current: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.strip():
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def load_manual_comments(paths: List[str]) -> List[AnnotRecord]:
    manual_records: List[AnnotRecord] = []
    pattern = re.compile(r"^Page\s+(\d+)(?:\s*,?\s*line\s+(\d+))?\s*:\s*(.*)$", re.IGNORECASE)

    for manual_path in paths:
        path = Path(manual_path)
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"Warning: manual comment file not found: {manual_path}", file=sys.stderr)
            continue
        except OSError as exc:
            print(f"Warning: could not read manual comment file {manual_path}: {exc}", file=sys.stderr)
            continue

        for block in _split_manual_blocks(content):
            if not block:
                continue

            first = block[0].strip()
            page: Optional[int] = None
            line_number: Optional[int] = None
            body_parts: List[str]

            match = pattern.match(first)
            if match:
                page = int(match.group(1))
                if match.group(2):
                    line_number = int(match.group(2))
                first_body = match.group(3).strip()
                remaining = block[1:]
                if first_body:
                    body_parts = [first_body, *remaining]
                else:
                    body_parts = remaining
            else:
                body_parts = block

            comment_text = " ".join(part.strip() for part in body_parts if part.strip())
            if not comment_text:
                continue

            manual_records.append(
                AnnotRecord(
                    page=page,
                    type="Manual",
                    author=None,
                    comment_text=comment_text,
                    quoted_text=None,
                    line_number=line_number,
                    context_line_text=comment_text,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                )
            )

    return manual_records


def _auto_output_path(pdf_path: str, kind: str) -> Path:
    pdf = Path(pdf_path)
    stem = pdf.stem or "output"
    if kind == "json":
        filename = f"{stem}_comments.json"
    elif kind == "text":
        filename = f"{stem}_report.txt"
    else:
        filename = f"{stem}_comments.csv"
    return pdf.with_name(filename)


def main():
    usage_examples = textwrap.dedent(
        """\
        Examples:
          python extract_pdf_comments.py input.pdf -o comments.csv
          python extract_pdf_comments.py input.pdf --json -o comments.json
          python extract_pdf_comments.py input.pdf --text-report -o report.txt
          python extract_pdf_comments.py input.pdf --text-report --manual-text notes.txt
          python extract_pdf_comments.py input.pdf > comments.csv
        """
    )

    ap = argparse.ArgumentParser(
        description="Extract PDF comments / annotations with line numbers and locations.",
        epilog=usage_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("pdf", help="Input PDF file")
    ap.add_argument("-o", "--output", help="Output file path (default: stdout CSV)", default=None)
    ap.add_argument(
        "--manual-text",
        metavar="PATH",
        action="append",
        default=[],
        help="Add manual comments from text files; repeat to include multiple files."
        " Each block separated by blank lines becomes a manual annotation.",
    )
    format_group = ap.add_mutually_exclusive_group()
    format_group.add_argument("--json", action="store_true", help="Write JSON instead of CSV")
    format_group.add_argument(
        "--text-report",
        action="store_true",
        help="Write a plain-text report with one annotation per line",
    )
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    pdf_records = extract_annotations(doc)

    manual_records: List[AnnotRecord] = []
    if args.manual_text:
        manual_records = load_manual_comments(args.manual_text)

    records = manual_records + pdf_records

    if args.text_report:
        if args.output:
            write_text_report(args.output, records)
        else:
            auto_path = _auto_output_path(args.pdf, "text")
            write_text_report(str(auto_path), records)
            print(f"No output specified; wrote text report to {auto_path}")
        return

    if args.json:
        payload = [asdict(r) for r in records]
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            auto_path = _auto_output_path(args.pdf, "json")
            with open(auto_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"No output specified; wrote JSON to {auto_path}")
        return

    if args.output:
        write_csv(args.output, records)
        return

    auto_path = _auto_output_path(args.pdf, "csv")
    write_csv(str(auto_path), records)
    print(f"No output specified; wrote CSV to {auto_path}")


if __name__ == "__main__":
    main()
