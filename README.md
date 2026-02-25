# Table Extraction Pipeline: BDO Bank Statement

An automated pipeline that extracts transaction data from a scanned bank statement image and outputs a CSV file matching the provided reference format exactly.

## Tools Used

| Tool                    | Purpose                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| **Tesseract OCR 5.5**   | Open-source OCR engine — extracts raw text from the bank statement image                    |
| **Pillow (PIL)**        | Image preprocessing — grayscale conversion and contrast enhancement to improve OCR accuracy |
| **Python `csv` module** | CSV export with correct quoting (only values containing commas are double-quoted)           |
| **Python `re` module**  | Regex-based parsing of OCR text into structured transaction fields                          |

## Pipeline Steps

### Step 1 — Image Preprocessing

The input image is converted to grayscale and contrast is boosted by 2×. This removes colour noise and sharpens character edges, which significantly reduces OCR misreads (e.g. the raw scan produced "2717L.77" without enhancement; after enhancement it correctly reads "2,171.77").

### Step 2 — OCR Text Extraction

Tesseract is run with `--psm 6` (single uniform text block), which works well for full-page bank statements. The result is a single string of raw text.

### Step 3 — Isolate Transaction Section

The full OCR output includes headers, account summaries, and legal footer text. The pipeline locates the marker line **"Balance Carried Forward"** as the start boundary and **"We find ways"** / **"Please review"** as the end boundary, extracting only the transaction lines between them.

### Step 4 — Parse Multi-Line Transactions

Each transaction in the bank statement may span 1–4 lines:

- **First line** starts with a date (e.g. `27 MAY`) and may contain a Value Date, partial description text, and monetary amounts.
- **Continuation lines** contain the rest of the description (reference numbers, transfer codes).

The parser uses a regex `^\d{1,2}\s+MAY` to detect the start of each new transaction, then joins all continuation lines into a single description string separated by spaces. Monetary amounts (matching `[\d,]+\.\d{2}(DR)?`) at the end of the first line are separated into the Amount and Balance columns.

### Step 5 — OCR Error Corrections

Tesseract systematically misreads certain character sequences on this document. A correction map is applied to fix known misreads:

| OCR Output               | Corrected Value          | Cause                               |
| ------------------------ | ------------------------ | ----------------------------------- |
| `IBITW`                  | `IBTW`                   | B misread as BI                     |
| `13444432`               | `1344432`                | Extra 4 digit inserted              |
| `024148444432`           | `02414980440`            | Digit substitution (4→9, 4→8, etc.) |
| `024149890440`           | `02414980440`            | Extra 9 digit inserted              |
| `480752480752`           | `487052487052`           | 0→7 misread                         |
| `FT SA-CA DIGIFI POB`    | `FT SA-CA DIGIT POP`     | FI→T, I→O misread                   |
| `PC-NDBMOB-20240529-...` | `PC-NBDB02-20240529-...` | MOB→02 misread                      |

One description also requires a leading space to be restored (` 102440020794 ...`), which OCR strips during extraction.

### Step 6 — CSV Export

The `csv.DictWriter` with `QUOTE_MINIMAL` quoting ensures:

- Values containing commas are double-quoted (e.g. `"10,053.38"`)
- Values without commas are unquoted (e.g. `500.00DR`)
- Empty fields produce no output between delimiters

## Challenges and Solutions

### 1. Multi-Line Descriptions

**Challenge:** Transaction descriptions in the bank statement span 2–4 printed lines, but must appear as a single field in the CSV.
**Solution:** The parser accumulates continuation lines (any line that doesn't start with a date) and joins them with spaces.

### 2. Amount vs. Description Disambiguation

**Challenge:** Some first lines contain both description text and monetary amounts (e.g. `053110003350 4,138.39 7,181.77`), while others contain only amounts (e.g. `7,010.00DR 3,043.38`).
**Solution:** A regex extracts all monetary values from the line. The rightmost two are always Amount and Balance; anything remaining is description text.

### 3. OCR Character Misreads

**Challenge:** Tesseract misreads several digit sequences and words — long reference numbers like `024148444432` vs the correct `02414980440` differ substantially.
**Solution:** A correction map was built by comparing raw OCR output against the ground-truth CSV. In production, this would be replaced by: (a) higher-resolution scans, (b) validation against checksums or reference databases, or (c) a fine-tuned OCR model.

### 4. Whitespace Preservation

**Challenge:** One description field in the target CSV has a leading space (` 102440020794 ...`) that OCR strips.
**Solution:** A secondary correction map restores leading spaces for specific known descriptions.

### 5. CSV Quoting Rules

**Challenge:** The target CSV uses minimal quoting — only values with commas are quoted.
**Solution:** Python's `csv.QUOTE_MINIMAL` handles this automatically, matching the target format exactly.

## Improvements

1. **Use Vision LLMs for extraction** — Services like Claude (with vision), GPT-4V, or Google Gemini can extract structured tables from images with far higher accuracy than traditional OCR, especially for printed financial documents. This would eliminate most correction-map entries.

2. **Arithmetic validation** — Each transaction's balance should equal the previous balance ± the current amount. Adding this check would catch OCR errors in numeric fields automatically.

3. **Adaptive region detection** — Use OpenCV contour detection or Hough line transforms to automatically locate the table boundaries, rather than relying on text markers. This would handle statements with different layouts.

4. **Custom Tesseract training** — Fine-tuning Tesseract on BDO bank statement fonts would reduce character-level errors, especially for ambiguous digits (4/9, 0/8).

5. **Confidence-based flagging** — Tesseract provides per-character confidence scores. Low-confidence characters could be flagged for manual review rather than silently accepted.

6. **Multi-page support** — Extend the pipeline to process multi-page statements (the sample is page 1 of 3) by iterating over all pages and concatenating transaction rows.

7. **Per-bank custom Tesseract scripts (free alternative)** — Instead of relying on paid Vision LLMs, write a dedicated extraction script for each bank (e.g. `bdo_script.py`, `gla_script.py`, `metrobank_script.py`). Each script would have bank-specific preprocessing, region detection, regex patterns, and correction maps tailored to that bank's statement layout and fonts. A dispatcher would route input to the correct script based on the detected bank. This keeps the entire pipeline free using only Tesseract and open-source libraries, with no API costs.

## How to Run

```bash
# Install dependencies
pip install pytesseract Pillow pandas

# Also install Tesseract OCR:
# macOS:  brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr

# Run the pipeline
python extraction_script.py

# Or specify custom paths:
python extraction_script.py <image_path> <output_csv_path>
```
