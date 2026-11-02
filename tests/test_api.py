import asyncio
import json

import httpx
import pandas as pd

from fairness_audit.api import create_app


def test_api_exposes_aggregate_rows_only(tmp_path):
    (tmp_path / "summary.json").write_text(
        json.dumps({"split": {"locked_official_test_rows": 5}}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "model": "primary",
                "policy": "global",
                "group_type": "race",
                "group_value": "White",
                "support": 5,
                "selection_rate": 0.4,
            }
        ]
    ).to_csv(tmp_path / "group_metrics.csv", index=False)

    async def request_reports():
        transport = httpx.ASGITransport(app=create_app(tmp_path))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            summary = await client.get("/summary")
            groups = await client.get(
                "/groups",
                params={
                    "model": "primary",
                    "policy": "global",
                    "group_type": "race",
                },
            )
            health = await client.get("/health")
        return summary, groups, health

    summary, groups, health = asyncio.run(request_reports())

    assert summary.status_code == 200
    assert summary.json()["split"]["locked_official_test_rows"] == 5
    assert groups.status_code == 200
    assert groups.json()[0]["group_value"] == "White"
    assert "age" not in groups.json()[0]
    assert health.json()["reports_ready"] is True
