"""
modules/nlp.py
Clinical Natural Language Processing for OPD Triage.
Supports English, Hinglish, Hindi transliterated, and Telugu-English complaints.
Extracts:
  - Present symptoms and clinical red flags
  - Negated symptoms (ensures "denies fever", "no chest pain", "chhati dard nahi" are not flagged)
  - Duration expressions (normalized to hours)
  - Severity qualifiers (mild, moderate, severe)
Provides explainable rule-based structured extraction and TF-IDF vectorization.
"""

import re
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Punctuation / delimiters to split clauses
CLAUSE_DELIMITERS = re.compile(r"[,;:\.\n]|\b(?:and|aur|mariyu|lekin|par|kani|but)\b", re.IGNORECASE)

# Negation triggers
PRE_NEGATIONS = [
    r"\bno\b", r"\bnot\b", r"\bdenies\b", r"\bdenied\b", r"\bwithout\b",
    r"\bnegative\s+for\b", r"\bfree\s+of\b", r"\bno\s+history\s+of\b",
    r"\bnahi\b", r"\bna\b", r"\bledu\b", r"\bleka\b"
]
POST_NEGATIONS = [
    r"\bnahi\s+hai\b", r"\bnahi\b", r"\bna\b", r"\bledu\b", r"\babsent\b",
    r"\bnot\s+present\b", r"\bnegative\b"
]

PRE_NEG_REGEX = re.compile(r"|".join(PRE_NEGATIONS), re.IGNORECASE)
POST_NEG_REGEX = re.compile(r"|".join(POST_NEGATIONS), re.IGNORECASE)

# Clinical Concept Patterns: (Concept Key, Regex Pattern, Is_Red_Flag)
CLINICAL_PATTERNS = [
    # Cardiac / ACS
    ("chest_pain", r"(?:chest|chhati|seene|chati|gundelo)(?:\s+(?:mein|me|lo))?(?:\s+\w+){0,3}\s+(?:pain|pressure|heaviness|tightness|burning|dard|noppi|manta)|chest\s*pain|chhati\s*dard|seene\s*dard|gundelo\s*noppi", True),
    ("radiation_to_arm", r"radiat(?:ing|ion)\s*to\s*(?:left\s*)?(?:arm|shoulder|jaw)|haath\s*(?:tak|me)?\s*dard|cheyi\s*laag(?:tundi|adam)", True),
    ("diaphoresis", r"sweat(?:ing|s)?|diaphoresis|profuse\s*sweat|paseena|paseene|chemata|chematalu", False),
    
    # Respiratory
    ("severe_dyspnea", r"severe\s*(?:shortness\s*of\s*breath|breathlessness|dyspnea)|cannot\s*speak\s*full\s*sentences|gasping|struggling\s*to\s*breathe|dam\s*ghut\s*raha|swasa\s*teeskoleka|swasa\s*aadatam\s*ledu", True),
    ("dyspnea", r"shortness\s*of\s*breath|breathless|breathlessness|difficulty\s*breathing|dyspnea|saans\s*(?:phool|chadh|lene\s*me\s*takleef)|aayasam|swasa\s*kashtam", False),
    ("wheeze", r"wheez(?:e|ing)|ghurr\s*ghurr|seeti\s*jaisi\s*awaaz", False),
    ("cyanosis", r"bluish|cyanosis|neela\s*pad\s*gaya|neeli\s*padipoyindi", True),
    ("stridor", r"stridor|high\s*pitched\s*sound\s*breathing", True),

    # Neurological / Stroke
    ("facial_droop", r"facial\s*droop|face\s*droop|muh\s*tedha|mooh\s*tedha|moothi\s*vankara", True),
    ("slurred_speech", r"slurred\s*speech|cannot\s*speak|difficulty\s*speaking|awaaz\s*ladkhada|bol\s*nahi\s*pa|maata\s*thadabad", True),
    ("unilateral_weakness", r"(?:one\s*sided|unilateral)\s*weakness|hemiparesis|arm\s*and\s*leg\s*weakness|ek\s*taraf\s*(?:kamzori|sunn)|oka\s*vaipu\s*(?:cheyyi|kaalu)\s*padipoy", True),
    ("seizure", r"seizure|convulsion|convulsions|fits|daura|daure|murcha|murcha\s*poyadu", True),
    ("altered_mentation", r"altered\s*mental|confused|confusion|unresponsive|drowsy|disoriented|behosh|hosh\s*me\s*nahi|sust|spruha\s*ledu", True),
    ("syncope", r"faint(?:ed|ing)?|syncope|loss\s*of\s*consciousness|behosh\s*ho\s*gaya|kallu\s*thirigi\s*padipoy", True),

    # Bleeding / GI / Trauma
    ("hematemesis", r"vomit(?:ing)?\s*blood|blood\s*in\s*vomit|hematemesis|khoon\s*ki\s*ulti|raktham\s*(?:tho\s*)?vaanthi", True),
    ("melena", r"black\s*stool|black\s*tarry|melena|kaala\s*latrine|nalla\s*motion", True),
    ("major_bleeding", r"heavy\s*bleed(?:ing)?|bleeding\s*profusely|uncontrolled\s*bleed|khoon\s*(?:ruk\s*nahi|bahut\s*beh)|raktham\s*karuthondi", True),
    ("head_injury", r"head\s*injury|hit\s*on\s*head|head\s*trauma|sar\s*pe\s*chot|sar\s*phat|thala\s*debba", False),

    # Anaphylaxis / Poisoning
    ("anaphylaxis", r"allergic\s*reaction|swelling\s*of\s*(?:lips|tongue|throat)|hives\s*and\s*breathless|dadduru|allargy", True),
    ("poisoning", r"poison|poisoning|consumed\s*pesticide|swallowed\s*chemical|overdose|zeher|keeda\s*maarne|purugula\s*mandhu|mandhu\s*thaag", True),

    # Pediatric Specific
    ("poor_feeding", r"not\s*(?:drinking|eating|feeding)|poor\s*feeding|doodh\s*nahi\s*pi|kuch\s*kha\s*pi\s*nahi|tindi\s*tinatledu", False),
    ("lethargic_child", r"lethargic|unusually\s*drowsy|bachcha\s*sust|pilladu\s*sustu", False),

    # General / Organ Specific
    ("fever", r"fever|febrile|high\s*temp|bukhar|tapam|jwaram", False),
    ("cough", r"cough|khansi|daggu", False),
    ("headache", r"(?:head|sir|sar|thala)(?:\s+(?:mein|me|lo))?(?:\s+\w+){0,2}\s+(?:pain|ache|dard|noppi)|headache|sir\s*dard|sar\s*dard|thala\s*noppi", False),
    ("vomiting", r"vomit(?:ing)?|ulti|vaanthi", False),
    ("loose_motions", r"loose\s*motion|diarrhea|dast|motions|virechanalu", False),
    ("abdominal_pain", r"(?:abdominal|stomach|pet|kadupu)(?:\s+(?:mein|me|lo))?(?:\s+\w+){0,3}\s+(?:pain|cramp|burning|dard|noppi|mantha)|abdominal\s*pain|stomach\s*pain|pet\s*dard|kadupu\s*noppi", False),
    ("flank_pain", r"flank\s*pain|kidney\s*pain|side\s*pain|kamar\s*me\s*dard", False),
    ("fracture_suspected", r"broken\s*bone|fracture|deformity|haddi\s*toot|terhi\s*ho\s*gayi|irigindi", False),
    ("rash", r"rash|itching|skin\s*allergy|khujli|dadduru", False),
    ("eye_problem", r"eye\s*(?:pain|redness|discharge)|aankh\s*me|kallu", False),
    ("ear_problem", r"ear\s*(?:pain|discharge)|kaan\s*me|chevi", False),
    ("toothache", r"tooth\s*pain|toothache|daant\s*me\s*dard|panti\s*noppi", False),
    ("pregnancy_complaint", r"pregnancy|pregnant|garbhawati|carrying|nela\s*tappindi", False),
]

# Severity Words
SEVERITY_MILD = re.compile(r"\b(?:mild|slight|little|halka|halki|chinna|konchem)\b", re.IGNORECASE)
SEVERITY_SEVERE = re.compile(
    r"\b(?:severe|acute|unbearable|extreme|worst|bahut\s*zyada|tez|chala\s*ekkuva|bardasht\s*nahi|sahan\s*nahi|marne\s*jaisa|chanipotunna)\b",
    re.IGNORECASE
)

# Duration Regex Patterns
DURATION_PATTERN_HOURS = re.compile(r"(\d+)\s*(?:hours?|hrs?|ghant[ae]|gantal[ua])\b", re.IGNORECASE)
DURATION_PATTERN_DAYS = re.compile(r"(\d+)\s*(?:days?|din|rojul[ua]|di,n)\b", re.IGNORECASE)
DURATION_PATTERN_WEEKS = re.compile(r"(\d+)\s*(?:weeks?|hafte|vaaraal[ua])\b", re.IGNORECASE)
DURATION_PATTERN_MONTHS = re.compile(r"(\d+)\s*(?:months?|mahine|nelal[ua])\b", re.IGNORECASE)
SUDDEN_ONSET_PATTERN = re.compile(r"\b(?:sudden|suddenly|acute|achanak|ventane|just\s*now)\b", re.IGNORECASE)


def extract_duration_hours(text: str) -> Tuple[Optional[float], bool]:
    """
    Extracts duration from text and converts to approximate hours.
    Returns (duration_hours, is_acute).
    """
    if not text or pd.isna(text):
        return None, False

    t = str(text).lower()

    if SUDDEN_ONSET_PATTERN.search(t):
        return 0.5, True

    m_h = DURATION_PATTERN_HOURS.search(t)
    if m_h:
        hours = float(m_h.group(1))
        return hours, hours <= 24.0

    m_d = DURATION_PATTERN_DAYS.search(t)
    if m_d:
        days = float(m_d.group(1))
        hours = days * 24.0
        return hours, hours <= 24.0

    m_w = DURATION_PATTERN_WEEKS.search(t)
    if m_w:
        weeks = float(m_w.group(1))
        return weeks * 168.0, False

    m_m = DURATION_PATTERN_MONTHS.search(t)
    if m_m:
        months = float(m_m.group(1))
        return months * 720.0, False

    return None, False


def extract_severity(text: str) -> str:
    """
    Categorizes symptom severity into 'mild', 'moderate', or 'severe'.
    """
    if not text or pd.isna(text):
        return "moderate"
    t = str(text)
    if SEVERITY_SEVERE.search(t):
        return "severe"
    if SEVERITY_MILD.search(t):
        return "mild"
    return "moderate"


def is_negated_in_clause(clause: str, match_start: int, match_end: int) -> bool:
    """
    Determines if a matched symptom in a clause is negated.
    Checks pre-negation (before the match) and post-negation (after the match).
    """
    pre_text = clause[:match_start]
    post_text = clause[match_end:]

    # Check words immediately preceding (within ~4 words)
    pre_words = pre_text.strip().split()[-4:] if pre_text.strip() else []
    pre_segment = " ".join(pre_words)
    if PRE_NEG_REGEX.search(pre_segment):
        return True

    # Check words immediately following (within ~4 words)
    post_words = post_text.strip().split()[:4] if post_text.strip() else []
    post_segment = " ".join(post_words)
    if POST_NEG_REGEX.search(post_segment):
        return True

    return False


def parse_clinical_text(text: str) -> Dict[str, Any]:
    """
    Analyzes free-text clinical complaint across English/Hinglish/Telugu-English.
    Returns structured clinical findings:
      - present_symptoms: set of confirmed symptoms
      - negated_symptoms: set of negated symptoms
      - detected_red_flags: list of high-risk red flag keys
      - severity: 'mild', 'moderate', 'severe'
      - duration_hours: float or None
      - is_acute: bool
    """
    if not text or pd.isna(text):
        return {
            "present_symptoms": [],
            "negated_symptoms": [],
            "detected_red_flags": [],
            "severity": "moderate",
            "duration_hours": None,
            "is_acute": False,
        }

    raw = str(text)
    severity = extract_severity(raw)
    duration_hours, is_acute = extract_duration_hours(raw)

    present = set()
    negated = set()
    red_flags = set()

    # Split into clauses
    clauses = CLAUSE_DELIMITERS.split(raw)

    for clause in clauses:
        clause_str = clause.strip()
        if not clause_str:
            continue

        for concept, pattern, is_rf in CLINICAL_PATTERNS:
            regex = re.compile(pattern, re.IGNORECASE)
            for m in regex.finditer(clause_str):
                start, end = m.span()
                if is_negated_in_clause(clause_str, start, end):
                    negated.add(concept)
                else:
                    present.add(concept)
                    if is_rf:
                        red_flags.add(concept)

    # Clean present of any strictly negated terms
    confirmed_present = sorted(list(present - negated))
    confirmed_negated = sorted(list(negated))
    confirmed_red_flags = sorted([rf for rf in red_flags if rf in confirmed_present])

    # Compound Red Flag Rules:
    # 1. ACS: chest_pain + (radiation_to_arm OR diaphoresis)
    if "chest_pain" in confirmed_present:
        if "radiation_to_arm" in confirmed_present or "diaphoresis" in confirmed_present:
            confirmed_red_flags.append("acs_pattern")

    # 2. Sepsis shock: altered_mentation + fever
    if "altered_mentation" in confirmed_present and "fever" in confirmed_present:
        confirmed_red_flags.append("sepsis_pattern")

    # 3. Severe respiratory: severe_dyspnea OR (dyspnea + cyanosis)
    if "cyanosis" in confirmed_present and "dyspnea" in confirmed_present:
        confirmed_red_flags.append("severe_respiratory_pattern")

    # 4. Stroke triad
    if ("facial_droop" in confirmed_present or "slurred_speech" in confirmed_present or "unilateral_weakness" in confirmed_present):
        confirmed_red_flags.append("stroke_pattern")

    return {
        "present_symptoms": confirmed_present,
        "negated_symptoms": confirmed_negated,
        "detected_red_flags": sorted(list(set(confirmed_red_flags))),
        "severity": severity,
        "duration_hours": duration_hours,
        "is_acute": is_acute,
    }


def batch_extract_nlp_features(df: pd.DataFrame, text_column: str = "chief_complaint") -> pd.DataFrame:
    """
    Applies clinical NLP extraction across a DataFrame.
    Adds indicator columns for key symptoms, red flags, duration, and severity.
    """
    records = df[text_column].fillna("").apply(parse_clinical_text)

    res = df.copy()
    res["nlp_severity"] = [r["severity"] for r in records]
    res["nlp_duration_hours"] = [r["duration_hours"] for r in records]
    res["nlp_is_acute"] = [int(r["is_acute"]) for r in records]
    res["nlp_red_flags"] = [",".join(r["detected_red_flags"]) for r in records]
    res["nlp_n_red_flags"] = [len(r["detected_red_flags"]) for r in records]

    # Add binary indicators for top concepts
    tracked_concepts = [
        "chest_pain", "severe_dyspnea", "dyspnea", "wheeze", "cyanosis",
        "facial_droop", "slurred_speech", "unilateral_weakness", "seizure",
        "altered_mentation", "syncope", "hematemesis", "melena", "major_bleeding",
        "head_injury", "poisoning", "anaphylaxis", "fever", "abdominal_pain",
        "fracture_suspected", "poor_feeding"
    ]
    for c in tracked_concepts:
        res[f"symptom_{c}"] = [int(c in r["present_symptoms"]) for r in records]

    return res


class ClinicalTextVectorizer:
    """
    TF-IDF Vectorizer wrapper for complaints.
    """
    def __init__(self, max_features: int = 500, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b",
        )

    def fit(self, texts: List[str]):
        clean_texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.vectorizer.fit(clean_texts)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        clean_texts = [str(t) if pd.notna(t) else "" for t in texts]
        return self.vectorizer.transform(clean_texts).toarray()

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)
