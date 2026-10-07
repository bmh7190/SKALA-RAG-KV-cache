"""Generated cover text and a content-derived fallback for older checkpoints."""

import re

from pydantic import BaseModel, Field, field_validator


class CoverOutput(BaseModel):
    title: str = Field(min_length=1, max_length=80, pattern=r"^[^\[\]\r\n]+$")
    subtitle: str = Field(max_length=120, pattern=r"^[^\[\]\r\n]*$")
    scope: str = Field(max_length=150, pattern=r"^[^\[\]\r\n]*$")

    @field_validator("title", "subtitle", "scope", mode="before")
    @classmethod
    def strip_text(cls, value):
        return value.strip() if isinstance(value, str) else value


def cover_from_state(state):
    """Legacy writer adapters derive the subject from the actual run, not constants."""
    return CoverOutput(
        title=" · ".join(state["selected_technologies"]) + " 기술 평가",
        subtitle=state["domain_and_criteria"]["domain"],
        scope=" / ".join(state["domain_and_criteria"]["criteria"]),
    )


def report_cover(report):
    """New drafts carry generated text; old drafts use the first body sentence."""
    if report.get("cover") is not None:
        return CoverOutput.model_validate(report["cover"])
    text = next(
        (
            text.strip()
            for title, text in report["sections"]
            if title != "REFERENCE" and text.strip()
        ),
        "",
    )
    text = re.sub(r"\[[^\[\]]+\]", "", text)
    title = re.split(r"[\r\n]|(?<=[.!?])\s", text, maxsplit=1)[0].strip()[:80]
    return CoverOutput(title=title, subtitle="", scope="")
