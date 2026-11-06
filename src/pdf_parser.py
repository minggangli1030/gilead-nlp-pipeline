#!/usr/bin/env python3
"""
Reusable PDF extraction, text cleaning, sentence segmentation, and metadata helpers.
Use scripts/parse_documents.py as the batch entry point.
"""

import re
import sys
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not installed. Install with: pip install PyPDF2")
    sys.exit(1)

def sent_tokenize(text):
    """Robust sentence tokenization with fallback."""
    try:
        from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
        return nltk_sent_tokenize(text)
    except:
        # Fallback to regex-based splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None
    return text


STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'in',
    'is', 'it', 'of', 'on', 'or', 'our', 'that', 'the', 'their', 'this', 'to',
    'was', 'we', 'with'
}

NON_ASCII_RE = re.compile(r'[^\x00-\x7F]+')

TABLE_INLINE_WORDS = {
    'rating', 'target', 'total', 'rtn', 'price', 'outperform', 'underperform',
    'buy', 'sell', 'neutral', 'overweight', 'underweight', 'perform', 'market'
}

TABLE_TOKEN_RE = re.compile(
    r"^(?:"
    r"EPS|EBITDA|REV(?:ENUE)?|GM|NM|OM|Q[1-4]|FY\d{2}|20\d{2}[A-Z]?|\d{1,2}%|"
    r"\$?\(?-?\d[\d,]*(?:\.\d+)?[A-Za-z]?\)?"
    r")$",
    re.IGNORECASE
)


def clean_and_reconstruct_text(text):
    """Normalise PDF text while preserving complete sentences."""
    raw_lines = text.split('\n')

    cleaned_lines = []
    for raw_line in raw_lines:
        line = raw_line.strip()

        if not line:
            cleaned_lines.append('')
            continue

        if (
            len(line) < 3 or
            re.match(r'^[\d\s\-\|\.]+$', line) or  # Pure numbers/symbols
            re.match(r'^[A-Z\s\-]{10,}$', line) or  # All caps headers
            ('Page' in line and re.search(r'\d+', line)) or
            line.count('|') > 3
        ):
            continue
        if (
            '@' in line
            or re.search(r'\(\d{3}\)\s*\d{3}', line)
            or (sum(ch.isdigit() for ch in line) > 0 and not re.search(r'[A-Za-z]', line))
        ):
            continue

        cleaned_lines.append(line)

    paragraphs = []
    buffer = []
    for line in cleaned_lines:
        if not line:
            if buffer:
                paragraphs.append(' '.join(buffer).strip())
                buffer = []
            continue

        if buffer and buffer[-1].endswith('-'):
            buffer[-1] = buffer[-1][:-1] + line
        else:
            buffer.append(line)

    if buffer:
        paragraphs.append(' '.join(buffer).strip())

    reconstructed_text = '\n'.join(paragraphs)
    return re.sub(r'\s+', ' ', reconstructed_text).strip()


def is_numeric_sentence(sentence):
    """Return True if a sentence is effectively numeric noise (tables, tickers)."""
    stripped = re.sub(r'[\s\.,:%$\-\u2022]+', '', sentence)
    if not stripped:
        return True
    return not re.search(r'[A-Za-z]', stripped)


def _is_table_like_token(token: str) -> bool:
    clean = re.sub(r'[^A-Za-z0-9]', '', token)
    if not clean:
        return True
    lower = clean.lower()
    if lower in STOPWORDS:
        return False
    if TABLE_TOKEN_RE.match(clean):
        return True
    if any(ch.isdigit() for ch in clean):
        return True
    if clean.isupper() and len(clean) > 2:
        return True
    if re.search(r'[A-Z][a-z]+[A-Z]', clean):
        return True
    return False


def strip_leading_table_tokens(sentence):
    """Remove leading table-like tokens from a sentence."""
    tokens = sentence.split()
    idx = 0

    while idx < len(tokens) and _is_table_like_token(tokens[idx]):
        idx += 1

    if idx >= 6 and idx < len(tokens):
        candidate = ' '.join(tokens[idx:]).strip()
        if candidate:
            return candidate

    # Additional handling for inline rating/target blocks
    post_markers = {'post', 'bottom', 'highlights'}
    post_idx = None
    for i, tok in enumerate(tokens):
        clean = re.sub(r'[^A-Za-z]', '', tok).lower()
        if clean in post_markers:
            post_idx = i
            break

    if post_idx is not None and post_idx > 0:
        start_idx = None
        for i in range(post_idx):
            tok = tokens[i]
            clean_lower = re.sub(r'[^A-Za-z]', '', tok).lower()
            if i <= 4:
                continue
            if (
                _is_table_like_token(tok)
                or clean_lower in TABLE_INLINE_WORDS
                or '$' in tok
                or ':' in tok
            ):
                start_idx = i
                break
        if start_idx is not None and post_idx - start_idx >= 2:
            trimmed = tokens[:start_idx] + tokens[post_idx:]
            return ' '.join(trimmed).strip()

    return sentence.strip()


def normalize_inline_artifacts(sentence):
    sentence = NON_ASCII_RE.sub(' ', sentence)
    sentence = re.sub(r'(?<=[a-z])(?=[A-Z][a-z])', ' ', sentence)
    sentence = re.sub(r'(?<=[0-9%$])(?=[A-Za-z]{2,})', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence.strip()


def looks_like_table_sentence(sentence):
    tokens = sentence.split()
    if len(tokens) < 6:
        return False
    tableish = 0
    meaningful = 0
    stopwords_found = 0
    for token in tokens:
        clean = re.sub(r'[^A-Za-z0-9]', '', token)
        if not clean:
            continue
        meaningful += 1
        if clean.lower() in STOPWORDS:
            stopwords_found += 1
            continue
        if any(ch.isdigit() for ch in clean) or clean.isupper():
            tableish += 1
    if meaningful == 0:
        return True
    if any(keyword in sentence for keyword in ['Senior Associate', 'Capital Markets', 'Source:', 'Analyst', 'Exhibit']):
        return tableish >= 4
    if stopwords_found >= max(1, meaningful // 6):
        return False
    return tableish >= max(5, int(0.7 * meaningful))


def merge_sentence_fragments(sentences):
    """Stitch fragments that were mistakenly split into separate sentences."""
    if not sentences:
        return []

    merged = [sentences[0]]

    for sentence in sentences[1:]:
        previous = merged[-1]
        if (
            previous
            and not previous.endswith(('.', '!', '?'))
            and sentence
            and sentence[0].islower()
        ):
            merged[-1] = f"{previous} {sentence}".strip()
        else:
            merged.append(sentence)

    return merged


def extract_sentences(text):
    """Extract all meaningful sentences from the text."""
    cleaned_text = clean_and_reconstruct_text(text)

    try:
        sentences = sent_tokenize(cleaned_text)
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)

    parsed_sentences = []

    for sentence in sentences:
        cleaned_sentence = re.sub(r'\s+', ' ', sentence).strip()
        if not cleaned_sentence:
            continue
        cleaned_sentence = normalize_inline_artifacts(cleaned_sentence)
        if is_numeric_sentence(cleaned_sentence):
            continue
        cleaned_sentence = strip_leading_table_tokens(cleaned_sentence)
        if not cleaned_sentence:
            continue
        if looks_like_table_sentence(cleaned_sentence):
            continue
        parsed_sentences.append(cleaned_sentence)

    merged = merge_sentence_fragments(parsed_sentences)

    deduped = []
    seen = set()
    for sentence in merged:
        if sentence not in seen:
            deduped.append(sentence)
            seen.add(sentence)

    return deduped


def extract_metadata(pdf_path, text):
    """Extract metadata from filename and content."""
    filename = Path(pdf_path).name
    
    # Extract date
    date_patterns = [
        r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})'
    ]
    
    date = "Unknown"
    for pattern in date_patterns:
        match = re.search(pattern, filename + " " + text[:200])
        if match:
            date = match.group()
            break
    
    # Extract bank from directory structure (e.g., 2025_Q3/EXAMPLE/BMO -> BMO)
    path_parts = Path(pdf_path).parts
    if len(path_parts) >= 3:
        bank = path_parts[-3] if 'Original_PDF' in str(pdf_path) or 'Parsed_CSV' in str(pdf_path) else path_parts[-2]
    else:
        bank = Path(pdf_path).parent.name.replace('_', ' ')
    
    # Extract quarter/year from path
    quarter = "Unknown"
    year = "Unknown"
    for part in path_parts:
        if re.match(r'20\d{2}_Q[1-4]', part):
            year_quarter = part.split('_')
            year = year_quarter[0]
            quarter = year_quarter[1]
            break
    
    return {
        'bank': bank,
        'date': date,
        'filename': Path(pdf_path).stem,
        'year': year,
        'quarter': quarter
    }
