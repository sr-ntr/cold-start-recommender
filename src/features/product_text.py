import numpy as np


FIELD_TOKEN_BUDGETS = {
    "title": 64,
    "categories": 32,
    "features": 90,
    "description": 54,
}


def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, np.ndarray):
        parts = [
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        ]
        return " ".join(parts)

    text = str(value).strip()
    return " ".join(text.split())


def truncate_field(text, tokenizer, max_tokens):
    if not text:
        return ""

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )["input_ids"]

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    )


def build_truncated_product_text(row, tokenizer):
    fields = {
        "title": clean_text(row["title"]),
        "categories": " > ".join(
            clean_text(category)
            for category in row["categories"]
            if clean_text(category)
        ),
        "features": clean_text(row["features"]),
        "description": clean_text(row["description"]),
    }

    sections = []

    for field_name in [
        "title",
        "categories",
        "features",
        "description",
    ]:
        text = truncate_field(
            fields[field_name],
            tokenizer,
            FIELD_TOKEN_BUDGETS[field_name],
        )

        if text:
            sections.append(f"{field_name.capitalize()}: {text}")

    return "\n".join(sections)
