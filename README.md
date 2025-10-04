# PDF comment extractor

`extract_pdf_comments.py` pulls annotations (comments, highlights, strikeouts, etc.) from a PDF and converts them into reviewer-friendly output formats.

## Features

- Captures per-annotation metadata: page, type, author, inline comment, quoted text, context line, and bounding box.
- Detects printed line numbers in the PDF and maps annotations to those numbers when available.
- Exports to CSV, JSON, or a referee-style text report (one line per comment).
- Automatically chooses an output filename when `-o/--output` is omitted, printing where the file was written.
- Supports PyMuPDF-compatible PDFs (`pip install pymupdf`).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pymupdf
```

## Usage

```bash
python extract_pdf_comments.py INPUT.pdf [options]
```

Options:

- `-o`, `--output PATH` – Explicit output file destination.
- `--json` – Write the results as JSON (pretty-printed).
- `--text-report` – Produce a plain-text referee report (one annotation per line).
- Omitting both `--json` and `--text-report` defaults to CSV output.

### Examples

```bash
# CSV, explicit output
python extract_pdf_comments.py paper.pdf -o comments.csv

# JSON, automatic filename (prints the path)
python extract_pdf_comments.py paper.pdf --json

# Referee-style report
python extract_pdf_comments.py paper.pdf --text-report -o report.txt

# Default CSV with auto-named file
python extract_pdf_comments.py paper.pdf
```

## Text report format

Each line in the text report summarises a comment in reviewer prose, e.g.

```
Page 12, line 204: Remove "the quoted phrase": Please rephrase to clarify the intent.
```

The wording adapts to the annotation type (`Strikeout`, `Caret`, `Highlight`, etc.) and includes the inline comment when provided.

## CSV / JSON schema

Field | Description
:-----|:-----------
`page` | 1-based page number
`type` | Annotation type (e.g., `Text`, `Highlight`, `Strikeout`)
`author` | Annotating user, if available
`comment_text` | Popup/comment text
`quoted_text` | Text region associated with the annotation
`line_number` | Printed line number (or layout order fallback)
`context_line_text` | Nearby full line of text
`bbox` | Bounding box (`x0,y0,x1,y1` in PDF coordinates)

## Limitations

- Requires PyMuPDF (`fitz`) and PDFs with extractable text.
- Line-number detection depends on numbers being present in the margin.
- Ink/stamp annotations are recorded but not converted into bespoke prose.

## License

MIT (supply your chosen license details here).

