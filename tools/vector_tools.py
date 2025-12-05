from typing import List, Dict, Any

from utils.logger import get_logger

logger = get_logger(__name__)


def vector_query(text: str) -> List[Dict[str, Any]]:
    """
    查詢向量資料庫，回傳相關的文件片段。

    實務實作（之後可替換）：
    - 使用 Chroma / Qdrant client
    - 對 text 做 embedding 後進行相似度搜尋
    - 回傳 top_k 筆結果，每筆包含 content、score、metadata

    目前先回傳假資料，確保 DocumentAgent 可以運行：
    """
    logger.info("vector_query called with text: %s", text)
    return [
        {
            "content": "出差申請需於出發前三日完成線上申請，並經直屬主管核准。",
            "score": 0.91,
        },
        {
            "content": "出差旅費報支需檢附交通票據與住宿收據，依公司規範報帳。",
            "score": 0.87,
        },
    ]
