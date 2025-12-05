from typing import Dict, Any

from utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseAgent:
    """
    Database Agent

    負責：
    - 接收自然語言問題
    - 轉換成 SQL（未來可接 LangChain Text-to-SQL）
    - 呼叫 MCP Server 中的 SQL 工具與 Chart 工具
    - 回傳表格與圖表 URL 給上游 Supervisor

    目前仍使用假資料，以方便先跑通 Multi-Agent 流程。
    """

    def __init__(self) -> None:
        # 未來可注入 MCP client 或 LangChain Chain
        pass

    def run(self, question: str) -> Dict[str, Any]:
        """
        執行 Database Agent 主流程。

        目前簡化為：
        - log 問題
        - 回傳假 rows + 假圖表 URL
        """
        logger.info("DatabaseAgent received question: %s", question)

        # TODO：
        # 1) 使用 LLM 產生 SQL
        # 2) 經由 MCPServer.sql_query(sql) 執行查詢
        # 3) 經由 MCPServer.chart_generate(rows) 產生圖表
        dummy_rows = [
            {"product": "A", "sales": 120},
            {"product": "B", "sales": 200},
            {"product": "C", "sales": 80},
        ]
        dummy_chart_url = "https://quickchart.io/chart?c={type:'bar',data:{labels:['A','B','C'],datasets:[{data:[120,200,80]}]}}"

        return {
            "type": "db_result",
            "rows": dummy_rows,
            "chart_url": dummy_chart_url,
        }
