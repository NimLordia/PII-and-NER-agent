# PII Detection Agent

An intelligent PII (Personally Identifiable Information) detection tool for PDF documents that combines regex patterns, Named Entity Recognition (NER), and LLM-based refinement to accurately identify and categorize sensitive information.

## Features

- **Multi-layered Detection**: Uses regex patterns, spaCy NER, and LLM refinement for accurate PII detection
- **Entity Types Detected**:
  - Person names
  - Organizations
  - Email addresses
  - Phone numbers
  - Address components
- **Smart Filtering**: LLM-based post-processing to remove false positives
- **Summary Statistics**: Aggregated view showing entity counts and page locations
- **PDF Support**: Extract and analyze text from PDF documents

## Installation

### Prerequisites

- Python 3.9 or higher
- Google API key for Gemini

### Setup

1. Clone the repository:
```bash
git clone https://github.com/NimLordia/PII-and-NER-agent.git
cd piiAgentRAG
```

2. Create a virtual environment:
```bash
python -m venv env
```

3. Activate the virtual environment:
   - Windows:
     ```bash
     env\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source env/bin/activate
     ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Download the spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

6. Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

Run the PII detection on a PDF document:

```bash
python pii_agent.py "path/to/document.pdf"
```

### Windows Example:
```bash
python pii_agent.py "C:\Documents\contract.pdf"
```

### Output Format

The tool outputs JSON with two main sections:

1. **detections**: Full list of all detected entities with their locations
2. **summary**: Aggregated statistics showing:
   - Entity value
   - Entity type
   - Total count of occurrences
   - Pages where it appears

Example output:
```json
{
  "detections": [
    {
      "value": "John Doe",
      "type": "person_name",
      "page": 1
    },
    ...
  ],
  "summary": [
    {
      "value": "John Doe",
      "type": "person_name",
      "count": 3,
      "pages": [1, 2, 5]
    },
    ...
  ]
}
```

## How It Works

1. **PDF Text Extraction**: Extracts text from PDF documents using PyMuPDF
2. **Regex Detection**: Applies regex patterns to detect:
   - Email addresses
   - Phone numbers
   - Address components
3. **NER Detection**: Uses spaCy's NER model to detect:
   - Person names (PERSON entities)
   - Organizations (ORG entities)
4. **LLM Refinement**: Sends detected entities to Google's Gemini model to:
   - Filter out false positives (generic legal terms, document sections)
   - Correctly categorize ambiguous entities
   - Return only genuine PII
5. **Summary Generation**: Aggregates results and counts occurrences

## Configuration

### Minimum Confidence Threshold

By default, the tool filters entities with confidence below 0.5. You can modify this in the code:

```python
result = _detect_pii_impl(pdf_path, minimum_confidence=0.7)  # Stricter filtering
```

### LLM Model

The tool uses `gemini-2.0-flash-exp` by default. You can change the model in `refine_detections_with_llm()`:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",  # Change this
    temperature=0.0,
    api_key=os.getenv("GOOGLE_API_KEY")
)
```

### Custom Regex Patterns

Add or modify regex patterns in the global variables section:

```python
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b")
```

## Project Structure

```
piiAgentRAG/
├── pii_agent.py          # Main application file
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not in git)
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Dependencies

- **PyMuPDF (fitz)**: PDF text extraction
- **spaCy**: Named Entity Recognition
- **LangChain**: LLM integration and agent framework
- **Google Generative AI**: LLM for refinement
- **python-dotenv**: Environment variable management

## Limitations

- Currently optimized for English text
- Legal documents may contain many entity-like terms that require careful filtering
- Brand names may sometimes be misclassified as person names
- OCR is not performed on scanned PDFs (text must be extractable)

## Future Improvements

- [ ] Add support for more PII types (SSN, credit cards, dates of birth)
- [ ] Implement OCR for scanned documents
- [ ] Add batch processing for multiple PDFs
- [ ] Create web interface
- [ ] Add support for other document formats (DOCX, TXT)
- [ ] Implement custom entity training

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Testing Dataset

For testing this tool, we recommend using the **Kleister-NDA dataset** - a collection of Non-Disclosure Agreement (NDA) documents perfect for PII detection testing.

### Download the Kleister-NDA Dataset

```bash
git clone https://github.com/applicaai/kleister-nda.git
```

Then test the PII detection on any document:

```bash
python pii_agent.py "kleister-nda/documents/00a1d238e37ac225b8045a97953e845d.pdf"
```

**Thank you to [applicaai](https://github.com/applicaai) for providing this excellent dataset!**

## Troubleshooting

### "No module named 'fitz'"
Run: `pip install PyMuPDF`

### "Can't find model 'en_core_web_sm'"
Run: `python -m spacy download en_core_web_sm`

### "Your default credentials were not found"
Make sure you have a `.env` file with a valid `GOOGLE_API_KEY`

### Path issues on Windows
Always use quotes around file paths: `python pii_agent.py "C:\path\to\file.pdf"`
