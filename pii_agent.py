"""
PII Detection Agent

A multi-layered PII detection tool for PDF documents that combines:
- Regex pattern matching for structured data (emails, phones)
- spaCy NER for entity recognition (names, organizations)
- LLM-based refinement to filter false positives and improve categorization

Author: Nimrod Shaham
Date: 2025
License: Free for public use. Please credit the author if you use this code or ideas from it.
"""

import json
import os
import re
import sys
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import fitz  # PyMuPDF
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_react_agent
import spacy
from langchain_google_genai import ChatGoogleGenerativeAI



nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Regex patterns for PII
# -----------------------------
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# International-ish phone (very permissive), tweak for your locale as needed:
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b")
# Credit card candidates (Luhn will filter later):
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){12,19}\b")

# Basic address-ish hints (very heuristic; feel free to expand or replace):
ADDRESS_HINT_REGEX = re.compile(r"\b(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Blvd\.|Lane|Ln\.|House|Apartment|Apt\.|City|Zip|Postal)\b", re.IGNORECASE)
COMPANY_REGEX = re.compile(r"\b(?:Ltd\.|Limited|Inc\.|LLC|Corporation|Company|S\.A\.|GmbH)\b", re.IGNORECASE)
ADDRESS_REGEX = re.compile(r"\b\d{1,4}\s+[A-Za-z0-9.\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Way)\b", re.IGNORECASE)


# -----------------------------
# Detector
# -----------------------------
def detect_pii_in_text(page_text: str, page_number: int) -> List[Dict[str, Any]]:
    detections: List[Dict[swhtr, Any]] = []

    for match in EMAIL_REGEX.finditer(page_text):
        span_text = match.group(0)
        detections.append({
            "type": "email",
            "value": span_text,
            "page": page_number,
            "start": match.start(),
            "end": match.end(),
            "confidence": 0.99
        })

    for match in PHONE_REGEX.finditer(page_text):
        span_text = match.group(0)
        normalized = re.sub(r"\D", "", span_text)
        if len(normalized) >= 9:
            detections.append({
                "type": "phone",
                "value": span_text,
                "page": page_number,
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.85
            })


    # Address hints are weak; mark lower confidence and let the LLM classify if needed
    for match in ADDRESS_HINT_REGEX.finditer(page_text):
        span_text = match.group(0)
        detections.append({
            "type": "address_hint",
            "value": span_text,
            "page": page_number,
            "start": match.start(),
            "end": match.end(),
            "confidence": 0.40
        })

    return detections


def detect_names(page_text: str, page_number: int):
    doc = nlp(page_text)
    detections = []

    # Common words to filter out (false positives)
    false_positives = {
        "subsequent employer", "employee", "employer", "agreement",
        "parties", "recitals", "protected information", "restriction period",
        "competitor"
    }

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Filter out false positives
            if ent.text.lower() not in false_positives:
                detections.append({
                    "type": "person_name",
                    "value": ent.text,
                    "page": page_number,
                    "confidence": 0.9
                })
        elif ent.label_ == "ORG":
            # Detect organizations
            if ent.text.lower() not in false_positives:
                detections.append({
                    "type": "organization",
                    "value": ent.text,
                    "page": page_number,
                    "confidence": 0.85
                })
    return detections


def extract_pdf_text_by_page(pdf_path: str) -> List[Tuple[int, str]]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

    pages: List[Tuple[int, str]] = []
    document = fitz.open(pdf_path)
    page_index = 0
    while page_index < document.page_count:
        page = document.load_page(page_index)
        text = page.get_text("text")
        pages.append((page_index + 1, text))
        page_index = page_index + 1
    document.close()
    return pages


# -----------------------------
# LangChain Tools
# -----------------------------
class LoadPdfInput(BaseModel):
    pdf_path: str = Field(..., description="Absolute or relative path to the PDF file on disk")


@tool("load_pdf_text", args_schema=LoadPdfInput)
def load_pdf_text_tool(pdf_path: str) -> str:
    """
    Load a PDF from disk and return a JSON with page_number and text per page.
    """
    # Handle if pdf_path is passed as a dict or dict-like string
    if isinstance(pdf_path, str) and pdf_path.startswith('{'):
        import ast
        try:
            parsed = ast.literal_eval(pdf_path)
            if isinstance(parsed, dict):
                pdf_path = parsed.get('pdf_path', '')
        except:
            pass
    elif isinstance(pdf_path, dict):
        pdf_path = pdf_path.get('pdf_path', '')

    # Normalize Windows paths
    pdf_path = os.path.normpath(str(pdf_path))

    pages = extract_pdf_text_by_page(pdf_path)
    result = [{"page": p, "text": t} for p, t in pages]
    return json.dumps({"pages": result}, ensure_ascii=False)


class DetectPiiInput(BaseModel):
    pdf_path: str = Field(..., description="Path to the same PDF. The tool will read and detect PII")
    minimum_confidence: float = Field(0.5, description="Minimum confidence to keep matches (0.0-1.0)")


def refine_detections_with_llm(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use LLM to refine and categorize the detected entities, filtering out false positives.
    """
    if not detections:
        return []

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.9,
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Create a summary of detections for the LLM
    detection_summary = []
    for det in detections:
        detection_summary.append({
            "value": det.get("value"),
            "type": det.get("type"),
            "page": det.get("page")
        })

    prompt = f"""You are a PII detection expert analyzing entities extracted from a legal document (NDA/Non-Compete agreement).

Below is a list of detected entities. Your task is to:
1. Filter out FALSE POSITIVES (generic legal terms, document sections, phrases that are not actual entities)
2. Correctly categorize remaining entities as:
   - "person_name": Actual people's names
   - "organization": Company/brand names
   - "email": Email addresses
   - "phone": Phone numbers
   - "address_hint": Address components

FALSE POSITIVES to remove:
- Generic legal terms like "Restriction Period", "Protected Information", "Non-Compete", "Agreement"
- Document section names
- Generic role names like "Employee", "Employer" (unless part of an actual name)
- Phrases or partial sentences

KEEP as valid entities:
- Actual person names (e.g., "Eric Dean Sprunk", "Jeffrey M. Cava")
- Company/organization names (e.g., "NIKE", "Adidas", "Reebok")
- Contact information (emails, phones, addresses)

Detected entities:
{json.dumps(detection_summary, indent=2)}

Return ONLY a JSON array of the valid entities in this exact format:
[
  {{"value": "entity value", "type": "category", "page": page_number, "confidence": 0.0-1.0}}
]

Be strict about filtering. When in doubt about whether something is a false positive, exclude it.
"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        refined = json.loads(content)
        return refined
    except Exception as e:
        print(f"Warning: LLM refinement failed: {e}. Returning original detections.", file=sys.stderr)
        return detections


def _detect_pii_impl(pdf_path: str, minimum_confidence: float = 0.5) -> str:
    """
    Detect PII in the PDF using regex + Luhn.
    Returns JSON with a 'detections' list (type, value, page, start, end, confidence).
    """
    # Handle if arguments are passed as a dict or dict-like string
    if isinstance(pdf_path, str) and pdf_path.startswith('{'):
        # It's a string representation of a dict, try to parse it
        import ast
        try:
            parsed = ast.literal_eval(pdf_path)
            if isinstance(parsed, dict):
                minimum_confidence = parsed.get('minimum_confidence', 0.5)
                pdf_path = parsed.get('pdf_path', '')
        except:
            pass
    elif isinstance(pdf_path, dict):
        minimum_confidence = pdf_path.get('minimum_confidence', 0.5)
        pdf_path = pdf_path.get('pdf_path', '')

    # Normalize Windows paths
    pdf_path = os.path.normpath(str(pdf_path))

    pages = extract_pdf_text_by_page(pdf_path)
    all_detections: List[Dict[str, Any]] = []
    for page_number, text in pages:
        # Detect regex-based PII
        page_detections = detect_pii_in_text(text, page_number)
        all_detections.extend(page_detections)

        # Detect names using NER
        name_detections = detect_names(text, page_number)
        all_detections.extend(name_detections)

    filtered: List[Dict[str, Any]] = []
    for item in all_detections:
        conf = item.get("confidence", 0.0)
        if float(conf) >= float(minimum_confidence):
            filtered.append(item)

    # Use LLM to refine and categorize detections
    refined_detections = refine_detections_with_llm(filtered)

    # Create summary: count occurrences and track pages for each entity
    summary = {}
    for det in refined_detections:
        value = det.get("value")
        entity_type = det.get("type")
        page = det.get("page")

        key = f"{value}||{entity_type}"  # Use a unique key combining value and type

        if key not in summary:
            summary[key] = {
                "value": value,
                "type": entity_type,
                "count": 0,
                "pages": []
            }

        summary[key]["count"] += 1
        if page not in summary[key]["pages"]:
            summary[key]["pages"].append(page)

    # Convert summary to list and sort by count (descending)
    summary_list = list(summary.values())
    summary_list.sort(key=lambda x: x["count"], reverse=True)

    return json.dumps({
        "detections": refined_detections,
        "summary": summary_list
    }, ensure_ascii=False)


# Create the tool wrapper for the agent
@tool("detect_pii", args_schema=DetectPiiInput)
def detect_pii_tool(pdf_path: str, minimum_confidence: float = 0.5) -> str:
    """
    Detect PII in the PDF using regex + Luhn.
    Returns JSON with a 'detections' list (type, value, page, start, end, confidence).
    """
    return _detect_pii_impl(pdf_path, minimum_confidence)


# -----------------------------
# Agent setup
# -----------------------------
SYSTEM_PROMPT = """You are a helpful PII-extraction assistant.
Your job is to:
1) Load the PDF.
2) Detect likely PII using the provided tools.
3) Return concise JSON summaries and file paths.

Important:
- Prefer the 'detect_pii' tool for finding PII.
- Do not invent file paths; only use what the user provides.
- If a step fails, report the error message and suggest a fix.
"""

def build_agent() -> AgentExecutor:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, api_key=os.getenv("GOOGLE_API_KEY"))

    tools = [load_pdf_text_tool, detect_pii_tool]

    # Create ReAct prompt template with required variables
    template = """You are a helpful PII-extraction assistant.
Your job is to:
1) Load the PDF.
2) Detect likely PII using the provided tools.
3) Return concise JSON summaries and file paths.

Important:
- Prefer the 'detect_pii' tool for finding PII.
- Do not invent file paths; only use what the user provides.
- If a step fails, report the error message and suggest a fix.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = ChatPromptTemplate.from_template(template)

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    return executor


# -----------------------------
# CLI Runner
# -----------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pii_agent.py <pdf_path>")
        print("")
        print("Examples:")
        print("  python pii_agent.py sample.pdf")
        print('  python pii_agent.py "C:\\path\\to\\document.pdf"')
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Call the implementation directly
    try:
        result = _detect_pii_impl(pdf_path, minimum_confidence=0.5)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
