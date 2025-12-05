from typing import Any, List, Dict
from configparser import ConfigParser

import pyodbc

from utils.logger import get_logger

logger = get_logger(__name__)


def _get_sql_connection(config_path: str = "config.ini"):
    """
    根據 config.ini 建立 SQL Server 連線。

    - 使用 pyodbc
    - 實務上建議搭配連線池管理
    """
    cfg = ConfigParser()
    cfg.read(config_path, encoding="utf-8")

    server = cfg.get("SQL", "server")
    database = cfg.get("SQL", "database")
    username = cfg.get("SQL", "username")
    password = cfg.get("SQL", "password")
    driver = cfg.get("SQL", "driver", fallback="ODBC Driver 18 for SQL Server")
    port = cfg.get("SQL", "port", fallback="1433")

    conn_str = (
        f"DRIVER={{{{ {driver} }}}};"
        f"SERVER={server},{port};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    )

    return pyodbc.connect(conn_str)


def run_sql(sql: str, config_path: str = "config.ini") -> List[Dict[str, Any]]:
    """
    執行 SQL 查詢並回傳資料列 List[Dict]。

    - 防呆：若連線或查詢失敗，會記錄 log 並回傳空結果
    """
    logger.info("run_sql called with sql: %s", sql)
    try:
        conn = _get_sql_connection(config_path)
    except Exception as e:
        logger.error("Failed to connect SQL Server: %s", e)
        return []

    rows: List[Dict[str, Any]] = []
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            columns = [col[0] for col in cursor.description]
            for r in cursor.fetchall():
                row_dict = {col: val for col, val in zip(columns, r)}
                rows.append(row_dict)
    except Exception as e:
        logger.error("SQL execution error: %s", e)
    finally:
        conn.close()

    return rows
