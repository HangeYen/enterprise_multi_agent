# agents/doc_agent.py
"""
DocumentAgent 強化版
- 企業制度 / 管理辦法向量檢索優化：
    - 依 question + slots 判斷主題類別（出差 / 請假 / 遲到 / 資遣 ...）
    - 自動擴充 query term：加入同義詞、關鍵詞
    - 呼叫 tools.vector_tools.vector_query 做 RAG 檢索
- 回傳：
    - answer：整理後的條列摘要
    - raw_docs：向量檢索回傳的原始文件內容
"""

from __future__ import annotations

from typing import Dict, Any, List

from utils.logger import get_logger
from tools.vector_tools import vector_query

logger = get_logger(__name__)


class DocumentAgent:
    """
    企業內部制度 / SOP / 管理辦法向量檢索 Agent。
    """

    def __init__(self, top_k: int = 3) -> None:
        """
        :param top_k: 希望取回的相關文件數量（視 vector_query 實作而定，這裡主要用來裁切輸出）
        """
        self.top_k = top_k

    # =============================================================
    # 主題分類（企業制度領域）
    # =============================================================
    def _detect_policy_topic(self, text: str) -> str:
        """
        簡單依關鍵字分類主題：
        - 出差 / 差旅管理
        - 請假 / 遲到 / 出勤
        - 資遣 / 離職 / 裁員
        - 其他通用人事制度
        """
        if any(k in text for k in ["出差", "差旅", "旅費"]):
            return "出差管理辦法"
        if any(k in text for k in ["請假", "病假", "事假", "特休", "加班", "補休", "遲到", "早退", "出勤"]):
            return "出勤與請假管理辦法"
        if any(k in text for k in ["資遣", "裁員", "遣散", "離職"]):
            return "離職與資遣辦法"
        if any(k in text for k in ["獎懲", "懲戒", "獎金", "獎勵"]):
            return "員工獎懲辦法"
        return "一般人事與行政管理辦法"

    def _build_query(self, question: str, slots: Dict[str, Any]) -> str:
        """
        依 question + slots 建立加強版向量檢索 query：
        - 加入推論出的主題標籤（topic）
        - 若有日期，可一併放入（方便檢索最新修訂版）
        """
        base = question.strip()
        topic = self._detect_policy_topic(base)

        parts: List[str] = [base, topic]

        date_str = slots.get("date")
        if isinstance(date_str, str) and date_str:
            parts.append(date_str)

        # 若有 time / people 通常對制度檢索影響不大，可視需要加入
        query = " ".join(p for p in parts if p)

        return query

    # =============================================================
    # 文件摘要整理
    # =============================================================
    def _summarize_docs(self, docs: List[str]) -> str:
        """
        將向量檢索回傳的多筆內容，整理成簡要條列。
        這裡不再呼叫 LLM，而是直接做簡單 rule-based 摘要。
        """
        if not docs:
            return (
                "目前無法從知識庫中找到明確對應的管理辦法條文。\n"
                "請確認關鍵字是否正確，或洽詢人資 / 行政單位。"
            )

        lines: List[str] = ["以下為文件管理辦法相關內容摘要：", ""]
        # 僅取前 top_k 筆
        for idx, d in enumerate(docs[: self.top_k], start=1):
            # 避免內容過長，可視情況做截斷
            text = str(d).strip()
            if len(text) > 200:
                text = text[:200] + "..."
            lines.append(f"{idx}. {text}")

        return "\n".join(lines)

    # =============================================================
    # 對外主函式
    # =============================================================
    def run(self, question: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        DocumentAgent 主入口：
        - 建立增強版 query term
        - 呼叫 vector_query(text=...)
        - 將結果整理為 answer（條列摘要）
        """
        logger.info("DocumentAgent received question=%s, slots=%s", question, slots)

        query_text = self._build_query(question, slots)
        logger.info("DocumentAgent vector query terms=%s", query_text)

        # 呼叫向量檢索工具
        try:
            result = vector_query(text=query_text)
        except TypeError:
            # 若舊版 vector_query 僅接受一個參數，退回舊呼叫方式
            result = vector_query(query_text)

        # 將 result 規整成 List[str]
        docs: List[str] = []
        if isinstance(result, dict):
            # 常見格式：{"docs": [...]} / {"documents": [...]}
            maybe_docs = result.get("docs") or result.get("documents") or []
            if isinstance(maybe_docs, list):
                docs = [str(d) for d in maybe_docs]
            else:
                docs = [str(maybe_docs)]
        elif isinstance(result, list):
            docs = [str(d) for d in result]
        elif result is not None:
            docs = [str(result)]

        answer = self._summarize_docs(docs)

        return {
            "answer": answer,
            "raw_docs": docs,
            # 此處未直接再呼叫 LLM，因此 Token 設為 0
            "usage": {
                "total_tokens": 0
            },
        }
