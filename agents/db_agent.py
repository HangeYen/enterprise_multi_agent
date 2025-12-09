# agents/db_agent.py
"""
DatabaseAgent 強化版
- 自動從 question / slots 解析：
    - metrics（例如：sales / revenue / profit / headcount）
    - period（例如：this_month / last_month / this_year / last_year）
- 真實情境下應改為組出 SQL 並查詢 BI / DWH，此處為示範用假資料。
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseAgent:
    """
    企業內部 BI / SQL 專用 Agent（示範版）
    """

    # 關鍵字對應指標類型（metrics）
    METRIC_KEYWORDS = {
        "sales": ["銷售", "營收", "營業額", "sales", "revenue"],
        "profit": ["獲利", "毛利", "淨利", "profit"],
        "headcount": ["人數", "員工數", "人力", "headcount"],
        "orders": ["訂單", "單量", "orders"],
    }

    # 關鍵字對應期間（period）
    PERIOD_KEYWORDS = {
        "this_month": ["本月", "這個月", "當月"],
        "last_month": ["上個月", "上月", "前一個月"],
        "this_year": ["今年"],
        "last_year": ["去年", "去年度"],
        "this_quarter": ["本季", "這一季"],
        "last_quarter": ["上季", "上一季"],
    }

    def __init__(self) -> None:
        pass

    # =============================================================
    # metrics / period 自動解析
    # =============================================================
    def _infer_metrics(self, text: str, slots: Dict[str, Any]) -> List[str]:
        """
        根據 question / slots 自動推論 metrics。
        - 若 slots 中已有 metrics，會優先使用，否則根據關鍵字補齊。
        """
        metrics = slots.get("metrics") or []
        if not isinstance(metrics, list):
            metrics = []

        lowered = text.lower()

        # 若已經有 metrics，就不再覆蓋，只補充
        for metric_name, keywords in self.METRIC_KEYWORDS.items():
            if metric_name in metrics:
                continue
            if any(kw.lower() in lowered for kw in keywords):
                metrics.append(metric_name)

        # 如果仍然空，且題目有「報表」之類字樣，預設給 sales
        if not metrics and ("報表" in text or "報告" in text or "統計" in text):
            metrics.append("sales")

        return metrics

    def _infer_period(self, text: str, slots: Dict[str, Any]) -> str:
        """
        根據 question / slots 自動推論 period。
        - 若 slots 已有 period，優先使用。
        - 否則依關鍵字推論，最後 fallback 到 'recent'。
        """
        period = slots.get("period")
        if isinstance(period, str) and period:
            return period

        lowered = text.lower()

        for period_name, keywords in self.PERIOD_KEYWORDS.items():
            if any(kw.lower() in lowered for kw in keywords):
                return period_name

        # 若提到「去年Q1」等可延伸規則，可在此擴充
        # 簡化處理：若完全沒有資訊，就用 last_month 當示範預設
        if "去年" in text:
            return "last_year"
        if "今年" in text:
            return "this_year"

        return "last_month"

    # =============================================================
    # SQL（示範）生成
    # =============================================================
    def _build_demo_sql(self, metrics: List[str], period: str, slots: Dict[str, Any]) -> str:
        """
        示範用 SQL 生成邏輯：
        - 真實環境下應依據 data model 組合 SELECT / WHERE / GROUP BY
        """
        metric_cols = ", ".join(metrics) if metrics else "*"
        where_parts: List[str] = []

        # 根據 period 做簡單 where（示範）
        period_map = {
            "this_month": "order_month = THIS_MONTH()",
            "last_month": "order_month = LAST_MONTH()",
            "this_year": "order_year = THIS_YEAR()",
            "last_year": "order_year = LAST_YEAR()",
            "this_quarter": "order_quarter = THIS_QUARTER()",
            "last_quarter": "order_quarter = LAST_QUARTER()",
        }
        if period in period_map:
            where_parts.append(period_map[period])

        # 若 slots 有 date，可另外附加
        date_str = slots.get("date")
        if isinstance(date_str, str) and date_str:
            where_parts.append(f"order_date = '{date_str}'")

        where_clause = " AND ".join(where_parts) if where_parts else "1=1"

        sql = f"SELECT {metric_cols} FROM sales_summary WHERE {where_clause};"
        return sql

    # =============================================================
    # 對外主函式
    # =============================================================
    def run(self, question: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        DatabaseAgent 主入口：
        - 自動決定 metrics / period
        - 組出示範用 SQL
        - 回傳 summary 文字與 SQL 字串
        """
        logger.info("DatabaseAgent received question=%s, slots=%s", question, slots)

        metrics = self._infer_metrics(question, slots)
        period = self._infer_period(question, slots)

        sql = self._build_demo_sql(metrics, period, slots)

        summary = (
            f"已根據條件 metrics={metrics}, period={period} 查詢銷售 / BI 資料。\n"
            f"（目前為示範用假資料，實務上請連接資料庫並執行 SQL。）"
        )

        logger.info("DatabaseAgent 推論結果：metrics=%s, period=%s, sql=%s", metrics, period, sql)

        return {
            "answer": summary,
            "metrics": metrics,
            "period": period,
            "sql": sql,
            # 此處暫未直接呼叫 LLM，Token 設為 0（若之後加上 RAG 可再調整）
            "usage": {
                "total_tokens": 0
            },
        }
