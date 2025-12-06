# agents/doc_agent.py
from typing import Dict, Any, Optional, List
from utils.logger import get_logger
from tools.vector_tools import vector_query

logger = get_logger(__name__)


class DocumentAgent:
    """
    文件 / 管理辦法 / SOP / 規範 查詢 Agent（slot-aware）

    支援欄位：
    - metrics: 若使用者明確提到查詢 "啥內容"
    - period: 查詢年度、版本（如：2024, 最近一年）
    - date: 查詢某日期版本的制度
    """

    def run(self, question: str, slots: Dict[str, Any] | None = None) -> Dict[str, Any]:
        logger.info("DocumentAgent received question=%s, slots=%s", question, slots)

        slots = slots or {}

        # ----------- 從 slots 萃取參數 -----------
        metrics = slots.get("metrics") or []
        period = slots.get("period")
        date = slots.get("date")

        # 組合查詢關鍵字（加入使用者原始 question）
        query_terms: List[str] = [question]
        query_terms.extend(metrics)

        if period:
            query_terms.append(str(period))
        if date:
            query_terms.append(str(date))

        final_query = " ".join(query_terms)
        logger.info("DocumentAgent vector query terms=%s", final_query)

        # ----------- Vector Search 查詢文件 -----------
        try:
            vec_result = vector_query(final_query)
        except Exception as e:
            logger.error("呼叫向量查詢失敗：%s", e)
            return {
                "ok": False,
                "answer": "文件查詢系統無法使用，請稍後再試。",
                "query": final_query,
                "snippets": [],
            }

        # 格式化向量回傳內容（擷取最相關片段）
        snippets = [hit.get("content", "") for hit in vec_result][:3]

        if not snippets:
            return {
                "ok": False,
                "answer": "目前查不到相關管理辦法或制度。",
                "query": final_query,
                "snippets": [],
            }

        # ----------- 最終回答：取前三個片段組合 -----------
        answer = self._format_answer(snippets)

        return {
            "ok": True,
            "answer": answer,
            "query": final_query,
            "snippets": snippets,
            "slots": slots,
        }

    # ------------------------------------------------------
    # Answer Formatter（可依企業需求調整）
    # ------------------------------------------------------
    def _format_answer(self, snippets: List[str]) -> str:
        lines = ["以下為文件管理辦法相關內容摘要：\n"]
        for i, s in enumerate(snippets, 1):
            lines.append(f"{i}. {s}\n")
        return "\n".join(lines)
