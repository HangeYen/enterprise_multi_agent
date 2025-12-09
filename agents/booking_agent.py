# agents/booking_agent.py
"""
BookingAgent v3
- 依據 slots.date / slots.time / slots.people 產生實際會議時間區間
- 強化時間解析：
    - 14:00、10:30、9:00-10:30、10:00~11:00、10:00 至 11:00
    - 早上 10:00、上午 9:30、下午兩點、晚上 7:00
    - 三點到四點、15:00-16:00 等範圍
- 若只有開始時間 → 預設會議時長 60 分鐘
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from utils.logger import get_logger

logger = get_logger(__name__)


class BookingAgent:
    """
    會議室預約 OA Agent（示範用）
    - 真實情境下，此處應改為呼叫會議室 API / DB 查詢閒置狀況
    """

    def __init__(self, default_duration_minutes: int = 60) -> None:
        """
        :param default_duration_minutes: 若只給開始時間，會議預設持續時間（分鐘）
        """
        self.default_duration_minutes = default_duration_minutes

    # =============================================================
    # 核心時間解析
    # =============================================================
    def _parse_chinese_hour(self, s: str) -> Optional[int]:
        """
        將「一二三四五六七八九十」與「兩」等中文數字解析為小時（0~23 不含日期）。
        僅處理 1~12 點的情境，例如：
        - '三點' → 3
        - '兩點' → 2
        """
        mapping = {
            "零": 0, "〇": 0,
            "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
            "十一": 11, "十二": 12,
        }

        s = s.replace("時", "").replace("點", "")
        if not s:
            return None

        # 直接對照完整數字
        if s in mapping:
            return mapping[s]

        # 簡單處理「十X」→ 10 + X
        if s.startswith("十") and len(s) == 2:
            base = 10
            tail = mapping.get(s[1])
            if tail is not None:
                return base + tail

        # 「X十」→ X * 10（此處幾乎不會出現在小時解析，保留以防萬一）
        if s.endswith("十") and len(s) == 2:
            head = mapping.get(s[0])
            if head is not None:
                return head * 10

        return None

    def _normalize_time_token(self, raw: str) -> str:
        """
        將時間字串做基本清理：
        - 去掉多餘空白
        - 全形「：」轉半形「:」
        """
        if not raw:
            return ""
        t = raw.strip()
        t = t.replace("：", ":")
        return t

    def _parse_single_time(self, token: str) -> Optional[str]:
        """
        將單一時間片段解析成 'HH:MM' 格式。
        支援：
        - 14:00、9:30、09:00
        - 早上 10:00、上午9:30、下午兩點、晚上7點
        - 三點、五點半（半小時可以簡單處理為 :30）
        """
        if not token:
            return None

        t = self._normalize_time_token(token)
        # 去除空白
        t = t.replace(" ", "")

        # 先處理包含數字的情況，例如 10:00、9:30、9點、9時
        # 移除中文前綴（早上/上午/下午/晚上 等），但保留是否為下午/晚上，用於 12 小時制轉換
        is_pm = any(x in t for x in ["下午", "晚上", "傍晚"])
        is_am = any(x in t for x in ["早上", "上午", "清晨", "早晨"])

        for prefix in ["早上", "上午", "下午", "晚上", "傍晚", "清晨", "早晨"]:
            t = t.replace(prefix, "")

        # Case 1: 含冒號 → 直接視為 HH:MM 或 H:MM
        if ":" in t:
            try:
                parts = t.split(":")
                if len(parts) != 2:
                    return None
                hour = int(parts[0])
                minute = int(parts[1]) if parts[1] else 0

                # 處理 AM/PM
                if is_pm and hour < 12:
                    hour += 12
                if is_am and hour == 12:
                    hour = 0

                return f"{hour:02d}:{minute:02d}"
            except ValueError:
                return None

        # Case 2: 不含冒號，例如「三點」、「兩點」、「七點半」
        # 先判斷是否有「半」
        has_half = "半" in t
        t = t.replace("半", "")

        # 去掉「點」、「時」
        t_clean = t.replace("點", "").replace("時", "")
        hour = None

        # 嘗試用阿拉伯數字解析
        if t_clean.isdigit():
            hour = int(t_clean)
        else:
            # 嘗試中文數字解析
            hour = self._parse_chinese_hour(t_clean)

        if hour is None:
            return None

        if is_pm and hour < 12:
            hour += 12
        if is_am and hour == 12:
            hour = 0

        minute = 30 if has_half else 0
        return f"{hour:02d}:{minute:02d}"

    def parse_time_range(self, raw_time: Optional[str]) -> Tuple[str, str]:
        """
        將 slots.time 解析成 (start_time, end_time)。

        支援：
        - 單一時間：'14:00'、'早上 10:00'、'下午兩點' → 預設 +default_duration_minutes
        - 範圍時間：'10:00-11:30'、'10:00~11:00'、'三點到四點' 等
        """
        logger.info("BookingAgent.parse_time 收到原始時間字串：%s", raw_time)

        # 預設使用「現在時間 + 1 小時」只是最後一道防線，正常情況應由 slots 提供
        now = datetime.now()
        default_start = now.replace(minute=0, second=0, microsecond=0)
        default_end = default_start + timedelta(minutes=self.default_duration_minutes)

        if not raw_time:
            start_str = default_start.strftime("%H:%M")
            end_str = default_end.strftime("%H:%M")
            logger.info("BookingAgent.parse_time 無時間 → 使用預設 %s ~ %s", start_str, end_str)
            return start_str, end_str

        text = self._normalize_time_token(raw_time)

        # 嘗試拆成範圍（到、至、-、~ 等）
        separators = ["到", "至", "-", "－", "~", "～"]
        for sep in separators:
            if sep in text:
                left, right = text.split(sep, 1)
                start_str = self._parse_single_time(left)
                end_str = self._parse_single_time(right)
                if start_str and end_str:
                    logger.info("BookingAgent.parse_time 範圍解析結果：%s -> %s ~ %s", raw_time, start_str, end_str)
                    return start_str, end_str

        # 否則視為單一時間
        start_str = self._parse_single_time(text)
        if start_str:
            # 單一時間 → 預設持續 default_duration_minutes
            try:
                dt = datetime.strptime(start_str, "%H:%M")
                end_dt = dt + timedelta(minutes=self.default_duration_minutes)
                end_str = end_dt.strftime("%H:%M")
            except Exception:
                # 若解析失敗 fallback 使用預設
                logger.warning("BookingAgent.parse_time 單一時間計算結束時間失敗，改用預設 60 分鐘。")
                end_str = (now + timedelta(minutes=self.default_duration_minutes)).strftime("%H:%M")

            logger.info("BookingAgent.parse_time 解析結果：%s -> %s ~ %s", raw_time, start_str, end_str)
            return start_str, end_str

        # 最後防線：完全解析失敗 → 使用預設
        start_str = default_start.strftime("%H:%M")
        end_str = default_end.strftime("%H:%M")
        logger.warning("BookingAgent.parse_time 解析失敗，使用預設時間區間：%s ~ %s", start_str, end_str)
        return start_str, end_str

    # =============================================================
    # 對外主函式
    # =============================================================
    def run(self, question: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        BookingAgent 主入口。

        :param question: 使用者原始問題（例如：幫我預約12/30下午兩點會議室...）
        :param slots: Supervisor 傳入之 slots（含 date / time / people）
        :return: dict，至少包含：
                 - message: 給使用者看的自然語言訊息
                 - date: 預約日期（YYYY-MM-DD 或原樣）
                 - start_time / end_time / people
                 - usage: {"total_tokens": 0}（示範）
        """
        logger.info("BookingAgent received question=%s, slots=%s", question, slots)

        # 1. 日期
        date_str = slots.get("date")
        if not date_str:
            # 若 Supervisor 沒有給 date，就簡單用今天當預設（示範用）
            date_str = datetime.now().strftime("%Y-%m-%d")

        # 2. 人數
        people = slots.get("people")
        if not isinstance(people, int) or people <= 0:
            people = 4  # 預設 4 人（示範）

        # 3. 時間解析：依 slots.time 產生開始、結束時間
        raw_time = slots.get("time")
        start_time, end_time = self.parse_time_range(raw_time)

        # 4. 組合回覆訊息（示範用假資料）
        message = (
            f"已為您預約 {date_str} {start_time} ~ {end_time} 的會議室，"
            f"預計 {people} 人使用。（示範用假資料）"
        )

        logger.info(
            "BookingAgent 產生回覆：date=%s start=%s end=%s people=%s",
            date_str,
            start_time,
            end_time,
            people,
        )

        # 5. 回傳結構化結果
        return {
            "message": message,
            "date": date_str,
            "start_time": start_time,
            "end_time": end_time,
            "people": people,
            # 此處未直接呼叫 LLM，因此 token usage 設為 0
            "usage": {
                "total_tokens": 0
            },
        }
