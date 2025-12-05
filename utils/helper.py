from datetime import datetime, timedelta
from typing import Tuple, Optional


def parse_datetime_from_text(text: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    簡易範例：從文字中解析起訖時間。

    實務上你可以改成：
    - 交給 LLM 指令產生結構化 JSON
    - 或使用 dateparser / Duckling 等套件

    目前為了讓 BookingAgent 可運行：
    - 一律回傳「明天 14:00 ~ 15:00」作為假資料。
    """
    now = datetime.now()
    start = (now + timedelta(days=1)).replace(hour=14, minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=1)
    return start, end
