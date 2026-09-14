# PII Detection Agent

A personal learning project by [Nimrod Shaham](https://github.com/NimLordia), built in October 2025 to explore how regex, named entity recognition (NER), and LLM refinement can work together to identify potential personally identifiable information (PII) in PDF documents.

The project turns extracted document text into page-linked detections and JSON summaries. It combines conventional text processing with spaCy and Google Gemini, with a focus on entities found in legal documents such as NDAs.

This repository preserves the original implementation as a record of the concepts I explored and the tools I learned to combine.

## What I built and learned

- Extracting text from PDFs with PyMuPDF while preserving page numbers.
- Combining regex for emails and phone numbers with spaCy's `PERSON` and `ORG` entities.
- Filtering common legal phrases and prompting Gemini to refine candidate entities into structured JSON.
- Aggregating repeated entities by value and type, with occurrence counts and page references.
- Wrapping PDF loading and detection as LangChain tools and experimenting with a ReAct agent.

The command-line entry point calls the detection pipeline directly. The separate `build_agent()` function contains the LangChain agent experiment.

## How it works

```text
PDF → page text → regex + spaCy NER → confidence filter → Gemini refinement → JSON + summary
```

1. **Extract:** PyMuPDF reads the text from each PDF page.
2. **Detect:** Regex finds email addresses, phone numbers, and address hints; spaCy finds people and organizations.
3. **Filter:** Candidates below the default `0.5` threshold are removed.
4. **Refine:** Gemini receives candidate values, types, and page numbers and is prompted to remove false positives and correct categories.
5. **Summarize:** Results are grouped by entity value and type, sorted by occurrence count.

## Setup

The instructions below document the original workflow. The refinement step uses `gemini-2.0-flash-exp`, which Google has since [retired](https://ai.google.dev/gemini-api/docs/models/gemini-2.0-flash); reproducing that step today requires a supported model integration.

Use Python 3.11 or newer; the pinned dependencies require at least Python 3.11. A Google API key is required for the Gemini integration. Dependencies are preserved from the original project.

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/NimLordia/PII-and-NER-agent.git
cd PII-and-NER-agent
python -m venv env
```

Activate it on Windows PowerShell:

```powershell
.\env\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source env/bin/activate
```

Install the pinned dependencies, including the `en_core_web_sm` spaCy model:

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file in the repository root using [`.env.example`](.env.example) as a template:

```dotenv
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

Run the pipeline on a PDF with extractable text:

```bash
python pii_agent.py "path/to/document.pdf"
```

The program prints JSON to standard output. To save it:

```bash
python pii_agent.py "path/to/document.pdf" > results.json
```

The output contains `detections` and a `summary`. Example structure using fictional data:

```json
{
  "detections": [
    {"value": "Alex Morgan", "type": "person_name", "page": 1, "confidence": 0.9},
    {"value": "Alex Morgan", "type": "person_name", "page": 2, "confidence": 0.9},
    {"value": "alex@example.com", "type": "email", "page": 2, "confidence": 0.99}
  ],
  "summary": [
    {"value": "Alex Morgan", "type": "person_name", "count": 2, "pages": [1, 2]},
    {"value": "alex@example.com", "type": "email", "count": 1, "pages": [2]}
  ]
}
```

## Scope and limitations

- Uses an English spaCy model and requires text-based PDFs; scanned documents need OCR before use.
- Detector confidence values are fixed heuristics. NER and LLM refinement can miss entities or return false positives, so results need review.
- Address detection only produces keyword hints with confidence `0.40`, so the default `0.5` threshold excludes them before refinement.
- PDF extraction and initial detection happen locally. Candidate entity values, types, and page numbers are sent to Google's Gemini API for refinement.
- If the model invocation or response parsing fails, the code prints a warning and returns the candidates from before refinement.

## Files

| File | Purpose |
| --- | --- |
| [`pii_agent.py`](pii_agent.py) | PDF extraction, detection, refinement, LangChain tools, agent setup, and CLI |
| [`requirements.txt`](requirements.txt) | Original pinned Python dependencies and spaCy model |
| [`LICENSE`](LICENSE) | MIT license |

## Acknowledgments and license

Thanks to [ApplicaAI](https://github.com/applicaai) for the [Kleister-NDA dataset](https://github.com/applicaai/kleister-nda), a source of NDA documents for document-processing experiments. See the dataset repository for its contents and usage terms.

Released under the [MIT License](LICENSE). Copyright © 2025 Nimrod Shaham.
