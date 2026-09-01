"""
pii_masking.py

Regex-based PII masking layer for SupportIQ.
Runs BEFORE any ticket enters the agent pipeline — ensures sensitive
data never reaches Groq API calls, DeepEval judge calls, or SQLite storage.

Masking order matters: context-specific patterns (CVV, OTP) are masked
FIRST because they rely on nearby keywords. Generic digit-count patterns
(card numbers, Aadhaar, phone) are masked LAST — by that point, any CVV/OTP
digits are already replaced with placeholder text, so they can't be
accidentally swept into a broader digit-matching pattern.
"""

import re


def mask_pii(text: str) -> str:
    """
    Masks personally identifiable information in text.
    Applied to incoming tickets before they enter the agent pipeline.

    Masking order (most specific → least specific):
        1. CVV (keyword + 3-4 digits)
        2. OTP (keyword + 4-6 digits)
        3. Email addresses
        4. PAN numbers (letter-digit-letter pattern, very specific)
        5. Aadhaar numbers (12 digits, often grouped)
        6. Card numbers (13-19 digits, most generic — masked last)
        7. Phone numbers (10 digits, Indian mobile pattern)

    Returns masked text — original digits/data replaced with [TYPE MASKED].
    """
    if not text:
        return text

    # 1. CVV — keyword-anchored, safest to mask first
    text = re.sub(
        r'\bCVV\s*(?:is|:)?\s*\d{3,4}\b',
        '[CVV MASKED]',
        text,
        flags=re.IGNORECASE
    )

    # 2. OTP — keyword-anchored
    text = re.sub(
        r'\bOTP\s*(?:is|:)?\s*\d{4,6}\b',
        '[OTP MASKED]',
        text,
        flags=re.IGNORECASE
    )

    # 3. Email addresses
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        '[EMAIL MASKED]',
        text
    )

    # 4. PAN numbers — 5 letters + 4 digits + 1 letter (very specific format)
    text = re.sub(
        r'\b[A-Z]{5}\d{4}[A-Z]\b',
        '[PAN MASKED]',
        text
    )

    # 5. Aadhaar numbers — 12 digits, often grouped in sets of 4
    text = re.sub(
        r'\b\d{4}\s?\d{4}\s?\d{4}\b',
        '[AADHAAR MASKED]',
        text
    )

    # 6. Card numbers — 13-19 digits, most generic pattern, masked last
    text = re.sub(
        r'\b(?:\d[ -]?){13,19}\b',
        '[CARD MASKED] ',
        text
    )

    # 7. Phone numbers — 10 digit Indian mobile (after longer patterns handled)
    text = re.sub(
        r'\b[6-9]\d{9}\b',
        '[PHONE MASKED]',
        text
    )

    return text


if __name__ == "__main__":
    test_cases = [
        "My card 4532123456789012 was charged twice, CVV is 123",
        "My Aadhaar number is 1234 5678 9012, please verify KYC",
        "OTP is 456789 but transaction failed",
        "My PAN is ABCDE1234F for tax purposes",
        "Contact me at rahul.sharma@gmail.com or 9876543210",
        "What is the interest rate on savings account?",
    ]

    print("Testing PII masking:\n")
    for original in test_cases:
        masked = mask_pii(original)
        print(f"Original: {original}")
        print(f"Masked:   {masked}")
        print("-" * 60)