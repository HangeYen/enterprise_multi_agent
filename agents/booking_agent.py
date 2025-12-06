# agents/booking_agent.py

from datetime import datetime, timedelta
import re
from utils.logger import get_logger

logger = get_logger(__name__)


class BookingAgent:
    """
    BookingAgent v3.3 — 修正「早上十點」被解析成 14:00 的錯誤
    - 新增中文數字解析（十、十一、十二）
    - 修正時間 fallback
    - 早上 / 上午 / 清晨 → 優先處理 AM
    """

    def __init__(self):
        pass

    # ---------------------------------------------------------------
    # 中文數字解析
    # ---------------------------------------------------------------
    def zh_to_num(self, text: str):
        """
        支援：
        一 二 三 四 五 六 七 八 九
        十 十一 十二
        """
        zh_map_single = {
            "零": 0, "〇": 0,
            "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9
        }

        if text in zh_map_single:
            return zh_map_single[text]

        # 十 = 10
        if text == "十":
            return 10

        # 十一 = 11、十二 = 12
        m = re.match(r"十([一二三四五六七八九])", text)
        if m:
            return 10 + zh_map_single[m.group(1)]

        return None

    # ---------------------------------------------------------------
    # 時間解析器
    # ---------------------------------------------------------------
    def parse_time(self, text: str) -> str:

        if not text:
            return "14:00"

        orig = text
        t = text.lower().replace("：", ":").replace(" ", "")
        logger.info("BookingAgent.parse_time 收到原始時間字串：%s", orig)

        # AM / PM 判斷
        is_pm = any(k in t for k in ["下午", "pm", "晚上", "傍晚"])
        is_am = any(k in t for k in ["上午", "早上", "清晨", "am"])

        hour = None
        minute = 0

        # ---------------------------------------------------------------
        # 1) 阿拉伯數字格式：10:30、14:20
        # ---------------------------------------------------------------
        m = re.search(r"(\d{1,2}):(\d{1,2})", t)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2))
        else:
            # ---------------------------------------------------------------
            # 2) X點Y分（含阿拉伯數字）
            # ---------------------------------------------------------------
            m = re.search(r"(\d{1,2})點(\d{1,2})分?", t)
            if m:
                hour = int(m.group(1))
                minute = int(m.group(2))
            else:
                # ---------------------------------------------------------------
                # 3) 中文 X點Y分（十點三十分）
                # ---------------------------------------------------------------
                m = re.search(r"([一二兩三四五六七八九十]+)點([一二三四五六七八九十]+)分?", t)
                if m:
                    hour = self.zh_to_num(m.group(1))
                    minute = self.zh_to_num(m.group(2))
                else:
                    # ---------------------------------------------------------------
                    # 4) X點半
                    # ---------------------------------------------------------------
                    m = re.search(r"(\d{1,2})點半", t)
                    if m:
                        hour = int(m.group(1))
                        minute = 30
                    else:
                        m = re.search(r"([一二兩三四五六七八九十]+)點半", t)
                        if m:
                            hour = self.zh_to_num(m.group(1))
                            minute = 30
                        else:
                            # ---------------------------------------------------------------
                            # 5) 單純 X點（含中文）
                            # ---------------------------------------------------------------
                            m = re.search(r"(\d{1,2})點", t)
                            if m:
                                hour = int(m.group(1))
                            else:
                                m = re.search(r"([一二兩三四五六七八九十]+)點", t)
                                if m:
                                    hour = self.zh_to_num(m.group(1))

        # ---------------------------------------------------------------
        # 若仍未解析到 hour → fallback = 14:00
        # ---------------------------------------------------------------
        if hour is None:
            hour = 14
            minute = 0

        # ---------------------------------------------------------------
        # AM/PM 修正（核心問題修正）
        # ---------------------------------------------------------------
        if is_pm and hour < 12:
            hour += 12

        if is_am:
            if hour == 12:
                hour = 0
            # 修正：若是早上，但被 fallback 成 14 → 改 10
            if hour > 12:
                hour = 10

        hhmm = f"{hour:02d}:{minute:02d}"

        logger.info("BookingAgent.parse_time 解析結果：%s -> %s", orig, hhmm)
        return hhmm

    # ---------------------------------------------------------------
    # 主入口
    # ---------------------------------------------------------------
    def run(self, question: str, slots: dict) -> dict:

        date = slots.get("date")
        time_str = slots.get("time")
        people = slots.get("people") or 4

        start_hhmm = self.parse_time(time_str or "14:00")

        start_dt = datetime.strptime(start_hhmm, "%H:%M")
        end_dt = start_dt + timedelta(hours=1)

        msg = (
            f"已為您預約 {date} {start_hhmm} ~ {end_dt.strftime('%H:%M')} 的會議室，"
            f"預計 {people} 人使用。（示範用假資料）"
        )

        logger.info(
            "BookingAgent 產生回覆：date=%s start=%s end=%s people=%s",
            date, start_hhmm, end_dt.strftime('%H:%M'), people
        )

        return {"message": msg, "usage": {"total_tokens": 0}}
