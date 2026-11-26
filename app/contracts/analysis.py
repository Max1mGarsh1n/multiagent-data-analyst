from typing import Literal

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    summary: str
    insights: list[str] = Field(default_factory=list)
    chart_type: Literal["bar", "line", "table", "none"] = "none"
    plotly_json: str | None = None
    warnings: list[str] = Field(default_factory=list)
