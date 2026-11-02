"""Read-only API that serves aggregate report artifacts only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from fairness_audit.constants import REPORTS_DIR


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    result = []
    for record in clean.to_dict(orient="records"):
        result.append(
            {
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in record.items()
            }
        )
    return result


def create_app(reports_dir: Path = REPORTS_DIR) -> FastAPI:
    app = FastAPI(
        title="Adult Income Fairness Audit API",
        description="Read-only endpoints for aggregate benchmark audit results.",
        version="0.1.0",
    )
    app.state.reports_dir = reports_dir

    @app.get("/health")
    async def health() -> dict[str, object]:
        ready = (reports_dir / "summary.json").is_file()
        return {"status": "ok" if ready else "reports_missing", "reports_ready": ready}

    @app.get("/summary")
    async def summary() -> dict[str, Any]:
        path = reports_dir / "summary.json"
        if not path.is_file():
            raise HTTPException(status_code=503, detail="Aggregate report is not built")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/groups")
    async def groups(
        model: str | None = Query(default=None),
        policy: str | None = Query(default=None),
        group_type: str | None = Query(default=None),
    ) -> list[dict[str, Any]]:
        path = reports_dir / "group_metrics.csv"
        if not path.is_file():
            raise HTTPException(status_code=503, detail="Aggregate report is not built")
        frame = pd.read_csv(path)
        for column, value in (
            ("model", model),
            ("policy", policy),
            ("group_type", group_type),
        ):
            if value is not None:
                if value not in set(frame[column].astype(str)):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Unknown {column}: {value}",
                    )
                frame = frame[frame[column].astype(str) == value]
        return _records(frame)

    return app


app = create_app()
