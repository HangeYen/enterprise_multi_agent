# agents/db_agent.py
from typing import Dict, Any, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseAgent:
    """
    Database / BI Agent（示範版）

    - question: 使用者原始問題
    - slots: Supervisor 抽出的語意欄位（例如 metrics / period）

    回傳格式：
    {
        "ok": True,
        "query": {...},       # 實際查詢條件（示範）
        "rows": [...],        # 查詢結果（示範）
        "message": "可直接給使用者看的說明文字"
    }
    """

    def run(self, question: str, slots: Dict[str, Any] | None = None) -> Dict[str, Any]:
        logger.info("DatabaseAgent received question=%s, slots=%s", question, slots)

        slots = slots or {}
        metrics: List[str] = slots.get("metrics") or ["sales"]
        period: Optional[str] = slots.get("period") or "last_month"

        # 這裡實務上應該組出 SQL / ORM 查詢條件
        # 目前先用示範資料
        query_conditions = {
            "metrics": metrics,
            "period": period,
        }

        # 假資料 rows
        rows: List[Dict[str, Any]] = []

        message = (
            f"已根據條件 metrics={metrics}, period={period} 查詢銷售報表。"
            f"（目前為示範用假資料，實務上請連接資料庫並格式化輸出。）"
        )

        return {
            "ok": True,
            "query": query_conditions,
            "rows": rows,
            "message": message,
        }
