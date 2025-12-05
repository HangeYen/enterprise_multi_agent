import os

from utils.logger import get_logger

logger = get_logger(__name__)

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")


def build_vector_db() -> None:
    """
    建立向量資料庫的腳本骨架。

    後續實作建議流程：
    1. 掃描 docs/ 目錄中的 PDF / DOCX
    2. 萃取純文字
    3. 使用固定規則或 LangChain Text Splitter 切 chunk
    4. 使用 embeddings 模型產生向量
    5. 寫入 Chroma / Qdrant 等向量庫

    目前僅印出提示，作為流程定位點。
    """
    logger.info("Start building vector database from docs: %s", DOCS_DIR)

    if not os.path.isdir(DOCS_DIR):
        logger.warning("Docs directory not found: %s", DOCS_DIR)
        return

    files = [f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))]
    logger.info("Found %d files for vectorization (stub).", len(files))

    # TODO: 實作完整向量化流程

    logger.info("Vector DB build skeleton completed.")


if __name__ == "__main__":
    build_vector_db()
