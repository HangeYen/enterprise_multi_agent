from typing import Any, Dict, List, Tuple
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)


def check_rooms(time_range: Tuple[datetime, datetime]) -> List[Dict[str, Any]]:
    """
    查詢指定時間區間內可用會議室。

    實務上應：
    - 呼叫企業 OA/Exchange/Google Calendar 之 API
    - 回傳可用會議室列表及其容量、設備資訊等

    本實作版本先使用假資料。
    """
    logger.info("check_rooms called with time_range: %s", time_range)
    return [
        {"room_id": "R301", "name": "會議室 301", "capacity": 8},
        {"room_id": "R302", "name": "會議室 302", "capacity": 10},
    ]


def book_room(time_range: Tuple[datetime, datetime], room_id: str) -> Dict[str, Any]:
    """
    預約特定會議室。

    實務上應呼叫 OA 系統寫入預約紀錄並回傳預約編號等。
    目前回傳假預約結果。
    """
    logger.info("book_room called with room=%s, time_range=%s", room_id, time_range)
    return {
        "room_id": room_id,
        "time_range": f"{time_range[0].isoformat()} ~ {time_range[1].isoformat()}",
        "status": "booked",
        "booking_id": "BK-EXAMPLE-001",
    }
