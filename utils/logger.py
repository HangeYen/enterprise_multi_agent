import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    建立並回傳統一格式的 logger。

    - 所有模組都應該透過本函式取得 logger
    - 方便集中調整 log 格式、等級與輸出位置
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # 避免重複加入 handler
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
