# enterprise_multi_agent

企業內部多 Agent AI 應用範例專案  
包含：

- Supervisor Agent（基於 LangGraph 1.0）
- Database Agent（BI / Text-to-SQL）
- Document Agent（RAG 文件查詢）
- Booking Agent（會議室預約）
- MCP 工具層（SQL / Vector / Chart / Booking）
- RWD Portal + Login（Flask + MongoDB）
- 向量資料庫建置流程
- LangSmith 追蹤與觀測

## 技術堆疊

- Python 3.10+
- LangChain 1.0+
- LangGraph 1.0+
- LangSmith
- Flask
- MongoDB（登入用）
- SQL Server / Azure SQL（BI 資料庫）
- 向量資料庫（Chroma / Qdrant 任選）
- QuickChart（圖表服務，可替換）

## 專案結構

```text
enterprise_multi_agent/
├── requirements.txt       # 套件需求列表
├── config.ini             # 全系統設定
├── main.py                # Supervisor Agent + LangGraph 1.0 工作流
│
├── auth/
│   └── auth.py            # Login / Token 機制（MongoDB）
│
├── portal/
│   ├── app.py             # Flask RWD Portal（前端 + 後端 API）
│   └── templates/         # HTML 模板
│
├── agents/
│   ├── db_agent.py        # Database Agent (Text-to-SQL + 圖表)
│   ├── doc_agent.py       # Document Agent（向量查詢 + RAG）
│   └── booking_agent.py   # Booking Agent（會議室預約）
│
├── tools/
│   ├── sql_tools.py       # SQL 查詢工具
│   ├── chart_tools.py     # 圖表工具（QuickChart）
│   ├── vector_tools.py    # 向量資料庫查詢工具
│   └── booking_tools.py   # 會議室預約工具
│
├── vector_db/
│   ├── build_vector_db.py # 建立向量庫（企業文件）
│   └── docs/              # 原始 PDF / Word 文件
│
├── mcp_server/
│   └── mcp_server.py      # MCP Server：統一封裝 SQL/Vector/Calendar 等工具
│
└── utils/
    ├── logger.py          # 日誌工具
    └── helper.py          # 共用函式
