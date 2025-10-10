# PDF comment extractor

`extract_pdf_comments.py` pulls annotations (comments, highlights, strikeouts, etc.) from a PDF and converts them into reviewer-friendly output formats.

## Features

- Captures per-annotation metadata: page, type, author, inline comment, quoted text, context line, and bounding box.
- Detects printed line numbers in the PDF and maps annotations to those numbers when available.
- Exports to CSV, JSON, or a referee-style text report (one line per comment).
- Merges additional manual notes from plain-text files (ideal for free-form reviewer comments).
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
- `--manual-text PATH` – Append manual comments loaded from a text file. Repeat to include multiple files.
- Omitting both `--json` and `--text-report` defaults to CSV output.

### Examples

```bash
# CSV, explicit output
python extract_pdf_comments.py paper.pdf -o comments.csv

# JSON, automatic filename (prints the path)
python extract_pdf_comments.py paper.pdf --json

# Referee-style report
python extract_pdf_comments.py paper.pdf --text-report -o report.txt

# Referee report with extra manual notes
python extract_pdf_comments.py paper.pdf --text-report --manual-text extra_comments.txt -o report.txt

# Default CSV with auto-named file
python extract_pdf_comments.py paper.pdf
```

## Manual comments

You can provide extra reviewer notes that are not embedded in the PDF by supplying one or more plain-text files with `--manual-text`.

Each file is parsed into comment blocks separated by blank lines. Blocks may optionally begin with a location header on the first line:

```
Page 5 line 12: tighten the derivation here

Double-check the appendix: numbering resets unexpectedly.
```

- The `Page ... line ...:` prefix (case insensitive) is stripped from the output and used to tag the note with a location.
- Lines inside a block are concatenated with spaces; keep separate thoughts in separate blocks.
- Manual comments appear before PDF-derived annotations in the text report, with a blank line between entries for readability.

## Text report format

Each entry in the text report summarises a comment in reviewer prose, e.g.

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

GNU General Public License-3.0.
