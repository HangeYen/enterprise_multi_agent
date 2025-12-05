from typing import Dict, Any

from utils.logger import get_logger
from utils.helper import parse_datetime_from_text

logger = get_logger(__name__)


class BookingAgent:
    """
    Booking Agent

    負責：
    - 解析使用者文字中的會議室預約需求（時間、場地）
    - 透過 MCP 工具層（booking_tools.py）查詢與預約會議室

    目前先使用 parse_datetime_from_text 回傳預設時間，
    並產生假預約結果，確保整體流程可運行。
    """

    def __init__(self) -> None:
        # 未來可注入 MCP client
        pass

    def run(self, question: str) -> Dict[str, Any]:
        logger.info("BookingAgent received question: %s", question)

        start, end = parse_datetime_from_text(question)
        # TODO: 實務上呼叫 MCP.booking_tools.check_rooms / book_room

        if not start or not end:
            message = "暫時無法解析您要預約的時間，請再具體說明日期與時段。"
        else:
            message = f"已為您預約 {start.strftime('%Y-%m-%d %H:%M')} ~ {end.strftime('%H:%M')} 的會議室（示範用假資料）。"

        return {
            "type": "booking_result",
            "message": message,
        }
