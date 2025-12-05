from typing import Any, List, Dict

import json
import urllib.parse

from utils.logger import get_logger

logger = get_logger(__name__)


def generate_chart(rows: List[Dict[str, Any]]) -> str:
    """
    使用 QuickChart 產生圖表 URL。

    - 目前假設 rows 至少有一個數值欄位與一個分類欄位
    - 實務上可依 BI 報表需求動態產圖
    - 此版本只回傳組好的 URL，不主動發 HTTP 請求
    """
    logger.info("generate_chart called with %d rows", len(rows))
    if not rows:
        # 回傳一個簡單的靜態示意圖
        return "https://quickchart.io/chart?c={type:'bar',data:{labels:['No Data'],datasets:[{data:[0]}]}}"

    # 假設第一個欄位為分類、第二個為數值
    first_row = rows[0]
    if len(first_row) < 2:
        return "https://quickchart.io/chart?c={type:'bar',data:{labels:['Invalid'],datasets:[{data:[0]}]}}"

    keys = list(first_row.keys())
    label_key = keys[0]
    value_key = keys[1]

    labels = [str(r.get(label_key, "")) for r in rows]
    values = [float(r.get(value_key, 0) or 0) for r in rows]

    chart_config = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": value_key, "data": values}
            ],
        },
    }

    encoded = urllib.parse.quote(json.dumps(chart_config))
    url = f"https://quickchart.io/chart?c={encoded}"
    return url
