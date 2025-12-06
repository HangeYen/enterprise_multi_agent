# agents/booking_agent.py

from typing import Dict, Any, Optional
import datetime as dt

from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================
# 時間與日期解析工具
# =============================================================

CHINESE_NUM = {
    "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}


def _parse_chinese_hour(token: str) -> Optional[int]:
    """
    專門處理像「三點」「三點半」「下午三點」這類中文格式，回傳 0~23 的小時數。
    - 若有「下午」或 token 外部語境是下午，可進一步做 12 小時 → 24 小時轉換
    目前簡化：沒有 AM/PM 資訊時，一律採用 24 小時制的「工作時段」：
    - 1~11 點 → 13~23？（此處我們選擇：1→13, 2→14, 3→15, 以此類推）
    你可以依實際需求調整。
    """
    # 先簡單判斷是否有「下午」字樣
    is_pm = "下午" in token or "pm" in token.lower()

    # 去掉「下午」「早上」「上午」等字樣
    clean = token.replace("下午", "").replace("早上", "").replace("上午", "")
    clean = clean.replace("點", "").replace("時", "")

    if not clean:
        return None

    # 如果是純數字（例如「3」「15」）
    if clean.isdigit():
        hour = int(clean)
    else:
        # 若是中文數字，例如「三」「兩」
        c = clean[0]
        if c in CHINESE_NUM:
            hour = CHINESE_NUM[c]
        else:
            return None

    # 套用 PM 規則或預設工作時段
    if is_pm:
        if hour < 12:
            hour += 12
    else:
        # 若未標明 AM/PM，這裡可以依企業邏輯決定
        # 這裡假設 1~11 點 → 13~23；12 點 → 12
        if 1 <= hour <= 11:
            hour += 12

    # 保護邊界
    hour = max(0, min(23, hour))
    return hour


def _resolve_date(date_str: Optional[str]) -> dt.date:
    """
    將語意日期轉成實際日期：
    - None → 今天
    - '明天' / '明日' / 'tomorrow' → 明天
    - '後天' / '後日' / 'day after tomorrow' → 後天
    - '2025-12-07' → 直接 parse
    其他 parse 失敗 → 今天
    """
    today = dt.date.today()
    if not date_str:
        return today

    s = str(date_str).strip()

    if s in ["明天", "明日", "tomorrow"]:
        return today + dt.timedelta(days=1)
    if s in ["後天", "後日", "day after tomorrow"]:
        return today + dt.timedelta(days=2)

    try:
        return dt.date.fromisoformat(s)
    except Exception:
        return today


def _resolve_time_range(time_str: Optional[str]) -> tuple[dt.time, dt.time]:
    """
    將時間字串轉成 (start_time, end_time)

    支援格式：
    - None → 預設 14:00 ~ 15:00
    - '14:00' → 14:00 ~ 15:00
    - '15:00-16:00'
    - '三點到四點'
    - '下午兩點'
    - '三點'（視為一小時區間）
    """
    default_start = dt.time(14, 0)
    default_end = dt.time(15, 0)

    if not time_str:
        return default_start, default_end

    s = str(time_str).strip()

    # 情況 1：包含 "-" 或 "到" → 視為區間
    if "-" in s or "到" in s:
        if "到" in s:
            parts = s.split("到")
        else:
            parts = s.split("-")

        if len(parts) >= 2:
            start_token = parts[0].strip()
            end_token = parts[1].strip()

            start_time = _parse_time_token(start_token) or default_start
            end_time = _parse_time_token(end_token) or (
                (dt.datetime.combine(dt.date.today(), start_time) + dt.timedelta(hours=1)).time()
            )
            return start_time, end_time

    # 情況 2：單一時間點（例如「下午兩點」「三點」「14:30」）
    start_time = _parse_time_token(s) or default_start
    end_time = (dt.datetime.combine(dt.date.today(), start_time) + dt.timedelta(hours=1)).time()
    return start_time, end_time


def _parse_time_token(token: str) -> Optional[dt.time]:
    """
    將單一時間標記轉成 time：
    - '14:00' / '14:30'
    - '三點' / '下午三點'
    - '3pm'
    """
    t = token.strip()

    # 1) 先試 24 小時制 'HH:MM'
    if ":" in t:
        try:
            hour, minute = t.split(":", 1)
            return dt.time(int(hour), int(minute))
        except Exception:
            pass

    # 2) 若含 am/pm
    lower = t.lower()
    if "am" in lower or "pm" in lower:
        is_pm = "pm" in lower
        digits = "".join(ch for ch in lower if ch.isdigit())
        if digits:
            hour = int(digits)
            if is_pm and hour < 12:
                hour += 12
            return dt.time(hour, 0)

    # 3) 中文格式
    hour = _parse_chinese_hour(t)
    if hour is not None:
        return dt.time(hour, 0)

    return None

class BookingAgent:
    """
    會議室預約 Agent（slot-aware + memory-aware friendly）

    - question: 使用者原始文字（自然語言）
    - slots: 由 Supervisor Router 提供的語意欄位：
        - date: '明天' / '2025-12-07' / None
        - time: '下午兩點' / '三點到四點' / '15:00-16:00' / None
        - people: int / None

    回傳：
    {
        "ok": True,
        "date": 原始 slot date,
        "time": 原始 slot time,
        "people": 人數,
        "start": ISO datetime string,
        "end": ISO datetime string,
        "message": "可直接給 User 的中文回覆"
    }
    """

    def run(self, question: str, slots: Dict[str, Any] | None = None) -> Dict[str, Any]:
        logger.info("BookingAgent received question=%s, slots=%s", question, slots)

        slots = slots or {}

        slot_date = slots.get("date")
        slot_time = slots.get("time")
        slot_people = slots.get("people")

        date_obj = _resolve_date(slot_date)
        start_t, end_t = _resolve_time_range(slot_time)

        people = slot_people if isinstance(slot_people, int) and slot_people > 0 else 4

        start_dt = dt.datetime.combine(date_obj, start_t)
        end_dt = dt.datetime.combine(date_obj, end_t)

        message = (
            f"已為您預約 {date_obj.isoformat()} "
            f"{start_t.strftime('%H:%M')} ~ {end_t.strftime('%H:%M')} 的會議室，"
            f"預計 {people} 人使用。（示範用假資料）"
        )

        return {
            "ok": True,
            "date": slot_date,
            "time": slot_time,
            "people": people,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "message": message,
        }
