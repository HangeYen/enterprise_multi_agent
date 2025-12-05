from typing import Any, List, Dict, Tuple
from datetime import datetime

from tools.sql_tools import run_sql
from tools.chart_tools import generate_chart
from tools.vector_tools import vector_query
from tools.booking_tools import check_rooms, book_room


class MCPServer:
    """
    MCPServer 提供統一介面給 Agents 呼叫工具。

    - 在此實作版本中採用「同進程函式呼叫」
    - 未來可以將此類呼叫包裝成 HTTP / stdio MCP Server 對外暴露
    """

    def sql_query(self, sql: str) -> List[Dict[str, Any]]:
        return run_sql(sql)

    def chart_generate(self, rows: List[Dict[str, Any]]) -> str:
        return generate_chart(rows)

    def vector_search(self, text: str) -> List[Dict[str, Any]]:
        return vector_query(text)

    def check_rooms(self, time_range: Tuple[datetime, datetime]) -> List[Dict[str, Any]]:
        return check_rooms(time_range)

    def book_room(self, time_range: Tuple[datetime, datetime], room_id: str) -> Dict[str, Any]:
        return book_room(time_range, room_id)


if __name__ == "__main__":
    server = MCPServer()
    print("MCP Server stub is ready.")
    print("SQL test:", server.sql_query("SELECT 1"))
