"""
main.py — Enterprise Multi-Agent Supervisor
功能重點：
- LangGraph 1.0 + MemorySaver：使用 LangGraph state 做多輪記憶
- 完全移除自訂 session dict，全靠 thread_id + checkpoint 維持 state
- Semantic Router（LLM + PydanticOutputParser）+ robust JSON cleaner
- Hybrid Router（LLM route + keyword fallback + multi-route）
- Slot-aware Supervisor（date/time/people/metrics/period）
- 會議室 OA slots（date/time/people）跨輪延續與更新
    - 整合 BookingAgent v2：真正依據 slots.time / slots.date / slots.people 產生會議時間區間
- Multi-Agent Orchestrator（async + concurrency；booking / db / doc multi tasks）
- Responder 統一彙整多 Agent 結果
- Token Logging：Router + DB + Doc + Booking（OpenAI SDK for Azure Token + LangGraph + Multi-Agent)
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, Literal, Dict, Any, List, Optional

from configparser import ConfigParser

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# LangChain core（只用 message / prompt / parser）
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field

# OpenAI 官方 SDK（支援 Azure 模式）
from openai import OpenAI

# Internal Agents
from utils.logger import get_logger
from agents.booking_agent import BookingAgent
from agents.db_agent import DatabaseAgent
from agents.doc_agent import DocumentAgent

logger = get_logger(__name__)


# ================================================================
# OpenAI SDK (Azure 模式) 初始化 — v9.6
# ================================================================
def init_azure_openai_client():
    """
    使用 OpenAI SDK 1.x 的 Azure 模式。

    client = OpenAI(
        api_key=AZURE_KEY,
        base_url=f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}",
        default_query={"api-version": AZURE_API_VERSION},
    )

    之後呼叫：
    client.chat.completions.create(model=AZURE_DEPLOYMENT, messages=[...])
    """
    cfg = ConfigParser()
    cfg.read("config.ini", encoding="utf-8")

    endpoint = cfg.get("AZURE_OPENAI", "endpoint").rstrip("/")
    key = cfg.get("AZURE_OPENAI", "key")
    deployment = cfg.get("AZURE_OPENAI", "deployment")
    api_version = cfg.get("AZURE_OPENAI", "api_version")

    client = OpenAI(
        api_key=key,
        base_url=f"{endpoint}/openai/deployments/{deployment}",
        default_query={"api-version": api_version},
    )

    logger.info(
        "Azure OpenAI SDK 初始化完成：endpoint=%s deployment=%s api_version=%s",
        endpoint,
        deployment,
        api_version,
    )

    return client, deployment


AZURE_CLIENT, AZURE_DEPLOYMENT = init_azure_openai_client()


# ================================================================
# 日期語意化：今天 / 明天 / 下星期一～下星期日 → YYYY-MM-DD
# ================================================================
WEEKDAY_MAP = {
    "星期一": 0, "禮拜一": 0, "週一": 0, "周一": 0,
    "星期二": 1, "禮拜二": 1, "週二": 1, "周二": 1,
    "星期三": 2, "禮拜三": 2, "週三": 2, "周三": 2,
    "星期四": 3, "禮拜四": 3, "週四": 3, "周四": 3,
    "星期五": 4, "禮拜五": 4, "週五": 4, "周五": 4,
    "星期六": 5, "禮拜六": 5, "週六": 5, "周六": 5,
    "星期日": 6, "禮拜天": 6, "禮拜日": 6, "週日": 6, "周日": 6,
}


def _next_weekday_after_now(weekday: int) -> datetime:
    """
    取得下一次指定星期幾（不包含今天）。
    例如今天星期六，weekday=1(週一) → 下週一。
    """
    now = datetime.now()
    diff = (weekday - now.weekday() + 7) % 7
    return now + timedelta(days=(diff or 7))


def normalize_date(text: Optional[str]) -> Optional[str]:
    """
    將自然語意日期轉成 YYYY-MM-DD：
    - 今天 / 明天 / 後天
    - 下星期一 ~ 下星期日 / 下週三 / 下禮拜五
    - 單獨「星期三」之類視為找下一次的星期三
    - 若已是 YYYY-MM-DD，直接回傳
    無法解析 → None
    """
    if not text:
        return None

    now = datetime.now()

    if "今天" in text:
        return now.strftime("%Y-%m-%d")
    if "明天" in text:
        return (now + timedelta(days=1)).strftime("%Y-%m-%d")
    if "後天" in text:
        return (now + timedelta(days=2)).strftime("%Y-%m-%d")

    if "下星期" in text or "下週" in text or "下周" in text or "下禮拜" in text:
        for k, wd in WEEKDAY_MAP.items():
            if k in text:
                dt = _next_weekday_after_now(wd)
                return dt.strftime("%Y-%m-%d")

    for k, wd in WEEKDAY_MAP.items():
        if k in text:
            dt = _next_weekday_after_now(wd)
            return dt.strftime("%Y-%m-%d")

    try:
        datetime.strptime(text, "%Y-%m-%d")
        return text
    except ValueError:
        return None


DATE_KEYWORDS = [
    "今天", "明天", "後天",
    "下星期", "下週", "下周", "下禮拜",
    "星期", "週", "周", "禮拜",
    "月", "日", "號",
    "202", "203",
]


def user_mentioned_date(text: str) -> bool:
    return any(k in text for k in DATE_KEYWORDS)


# ================================================================
# Router Schema
# ================================================================
class SlotModel(BaseModel):
    date: Optional[str] = None
    time: Optional[str] = None
    people: Optional[int] = None
    metrics: Optional[List[str]] = None
    period: Optional[str] = None


class RouterOutput(BaseModel):
    route: Literal["booking", "db", "doc", "none"]
    confidence: float
    slots: Optional[SlotModel] = None


router_parser = PydanticOutputParser(pydantic_object=RouterOutput)


# ================================================================
# Router Prompt
# ================================================================
router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是企業級意圖分類器，請嚴格輸出 JSON。

{format_instructions}

規則：
- date：只能輸出語意日期（今天、明天、後天、下星期一 等），不要輸出 YYYY-MM-DD。
- time：自然語意或 HH:MM。
- people：整數或 null。
- route：booking / db / doc / none。
- 不要加入多餘說明。
"""
        ),
        ("human", "使用者輸入：{input}")
    ]
).partial(
    format_instructions=router_parser.get_format_instructions()
)


# ================================================================
# JSON 清理
# ================================================================
def clean_llm_output(text: str) -> str:
    t = text.strip()

    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*", "", t)
        t = t.replace("```", "").strip()

    if t.lower().startswith("json"):
        t = t[4:].strip()

    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        return m.group(0).strip()

    return t


# ================================================================
# Semantic Router（Azure ChatCompletion + Token）
# ================================================================
def semantic_route(user_text: str) -> RouterOutput:
    lc_msgs = router_prompt.format_messages(input=user_text)

    api_msgs: List[Dict[str, str]] = []
    for m in lc_msgs:
        if isinstance(m, HumanMessage):
            api_msgs.append({"role": "user", "content": m.content})
        else:
            api_msgs.append({"role": "system", "content": m.content})

    resp = AZURE_CLIENT.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=api_msgs,
        temperature=0,
    )

    content = resp.choices[0].message.content
    usage = resp.usage

    semantic_route.last_usage = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    logger.info("[Token] Router usage=%s", semantic_route.last_usage)

    cleaned = clean_llm_output(content)
    result = router_parser.parse(cleaned)

    if result.slots is None:
        result.slots = SlotModel()

    logger.info(
        "Semantic Router 結果：route=%s, conf=%.2f, slots=%s",
        result.route,
        result.confidence,
        result.slots.model_dump(),
    )

    return result


# ================================================================
# Keyword Router（fallback 用）
# ================================================================
BOOKING_KW = ["會議室", "預約", "開會"]
DB_KW = ["銷售", "報表", "統計", "營收", "kpi", "分析"]
DOC_KW = ["出差", "辦法", "制度", "流程", "規範", "報帳", "SOP", "請假",
    "出勤", "合約", "合約書", "文件", "手冊", "指南", "政策",
    "規定", "準則", "條款", "條例", "法規",
    # HR / 勞動法務關鍵字(新增),
    "資遣", "解僱", "終止契約", "遣散", "裁員", "離職"]


def keyword_router(text: str) -> str:
    t = text.lower()
    if any(k in t for k in BOOKING_KW):
        return "booking"
    if any(k in t for k in DB_KW):
        return "db"
    if any(k in t for k in DOC_KW):
        return "doc"
    return "none"


# ================================================================
# Multi-route 判斷
# ================================================================
def detect_all_routes(text: str, primary: str) -> List[str]:
    routes: List[str] = []
    t = text.lower()

    if primary != "none":
        routes.append(primary)

    if any(k in t for k in DB_KW) and "db" not in routes:
        routes.append("db")
    if any(k in t for k in DOC_KW) and "doc" not in routes:
        routes.append("doc")
    if any(k in t for k in BOOKING_KW) and "booking" not in routes:
        routes.append("booking")

    return routes or ["none"]


# ================================================================
# Hybrid Router（slots 永遠來自 semantic router）
# ================================================================
def classify_intent(text: str):
    try:
        sem = semantic_route(text)
        slots = sem.slots
        if sem.confidence >= 0.5 and sem.route != "none":
            primary = sem.route
        else:
            primary = keyword_router(text)
    except Exception as e:
        logger.warning("Semantic Router 出錯，fallback keyword。錯誤：%s", e)
        primary = keyword_router(text)
        slots = SlotModel()

    routes = detect_all_routes(text, primary)
    return primary, slots, routes


# ================================================================
# LangGraph State 定義
# ================================================================
class EnterpriseState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    route: str
    routes: List[str]
    slots: Dict[str, Any]
    oa_slots: Dict[str, Any]
    payload: Dict[str, Any]
    token_usage: Dict[str, int]


# ================================================================
# Agents 實例
# ================================================================
booking_agent = BookingAgent()
db_agent = DatabaseAgent()
doc_agent = DocumentAgent()


# ================================================================
# Supervisor Node（路由 + slot merge）
# ================================================================
def supervisor_node(state: EnterpriseState) -> EnterpriseState:
    state.setdefault("payload", {})
    state.setdefault("slots", {})
    state.setdefault("oa_slots", {})
    state.setdefault("token_usage", {})
    state["payload"] = {}

    last = state["messages"][-1]
    if not isinstance(last, HumanMessage):
        state["route"] = "none"
        state["routes"] = ["none"]
        return state

    text = last.content
    logger.info("Supervisor 收到輸入：%s", text)

    primary, slot_obj, routes = classify_intent(text)

    usage = getattr(semantic_route, "last_usage", {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })
    state["token_usage"]["router"] = usage.get("total_tokens", 0)

    date_explicit = user_mentioned_date(text)
    prev_oa = state["oa_slots"] or {}

    slots_dict = slot_obj.model_dump()

    has_new_booking_info = (
        (slot_obj.people is not None)
        or (slot_obj.time is not None)
        or (slot_obj.date is not None and date_explicit)
    )

    if primary == "none" and prev_oa and has_new_booking_info:
        primary = "booking"
        routes = ["booking"]

    if slot_obj.date and date_explicit:
        normalized_date = normalize_date(slot_obj.date)
    else:
        normalized_date = None

    merged_oa = {
        "date": normalized_date or prev_oa.get("date"),
        "time": slot_obj.time or prev_oa.get("time"),
        "people": slot_obj.people if slot_obj.people is not None else prev_oa.get("people"),
    }

    state["oa_slots"] = merged_oa

    slots_dict["date"] = merged_oa.get("date")
    state["slots"] = slots_dict

    state["route"] = primary
    state["routes"] = routes

    logger.info(
        "Supervisor 判斷：primary=%s routes=%s slots=%s oa_slots=%s",
        primary,
        routes,
        slots_dict,
        merged_oa,
    )
    return state


# ================================================================
# Async Agent 子任務
# ================================================================
async def _run_booking(state: EnterpriseState):
    last_text = state["messages"][-1].content
    slots = state["oa_slots"]

    def _call():
        return booking_agent.run(last_text, slots)

    result = await asyncio.to_thread(_call)
    msg = result.get("message", "已處理會議室相關需求。（示範）")

    usage_tokens = 0
    if isinstance(result, dict):
        usage_tokens = result.get("usage", {}).get("total_tokens", 0)

    return {"booking": msg}, [AIMessage(content=msg)], usage_tokens


async def _run_db(state: EnterpriseState):
    last_text = state["messages"][-1].content
    slots = state["slots"]

    def _call():
        return db_agent.run(last_text, slots)

    result = await asyncio.to_thread(_call)
    msg = result.get("answer") or result.get("message") or "已完成資料庫查詢（示範）。"

    usage_tokens = 0
    if isinstance(result, dict):
        usage_tokens = result.get("usage", {}).get("total_tokens", 0)

    return {"db": msg}, [AIMessage(content=msg)], usage_tokens


async def _run_doc(state: EnterpriseState):
    last_text = state["messages"][-1].content
    slots = state["slots"]

    def _call():
        return doc_agent.run(last_text, slots)

    result = await asyncio.to_thread(_call)
    msg = result.get("answer") or result.get("message") or "已完成文件查詢（示範）。"

    usage_tokens = 0
    if isinstance(result, dict):
        usage_tokens = result.get("usage", {}).get("total_tokens", 0)

    return {"doc": msg}, [AIMessage(content=msg)], usage_tokens


# ================================================================
# Orchestrator Node（多 Agent 並行）
# ================================================================
async def agent_orchestrator_node(state: EnterpriseState) -> EnterpriseState:
    routes = state.get("routes", ["none"])
    logger.info("Orchestrator 收到 routes=%s", routes)

    tasks: List[asyncio.Task] = []

    if "booking" in routes:
        tasks.append(_run_booking(state))
    if "db" in routes:
        tasks.append(_run_db(state))
    if "doc" in routes:
        tasks.append(_run_doc(state))

    if not tasks:
        return state

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, Exception):
            logger.error("Agent 執行錯誤：%s", res)
            continue

        payload_delta, msgs_delta, usage_tokens = res
        key = list(payload_delta.keys())[0]

        state["payload"].update(payload_delta)
        state["messages"].extend(msgs_delta)
        state["token_usage"][key] = usage_tokens

    return state


# ================================================================
# Responder Node（任務整合 + fallback 強化版）
# ================================================================
def responder_node(state: EnterpriseState) -> EnterpriseState:
    payload = state.get("payload", {})
    usage = state.get("token_usage", {})

    # -------------------------------
    # 組合 Token 用量塊
    # -------------------------------
    total = sum(usage.values())
    token_lines = [
        f"Router = {usage.get('router', 0)}",
        f"Booking = {usage.get('booking', 0)}",
        f"DB = {usage.get('db', 0)}",
        f"Doc = {usage.get('doc', 0)}",
        f"Total = {total}",
    ]
    token_block = "【Token 用量】\n" + "\n".join(token_lines)

    # -------------------------------
    # 是否有任務可彙整？
    # -------------------------------
    has_task = any(k in payload for k in ("booking", "db", "doc"))

    if not has_task:
        final_msg = (
            "目前沒有偵測到可處理的任務。\n"
            "若需要查詢會議室 / 文件 / 資料庫，請提出具體需求。\n\n"
            + token_block
        )
    else:
        parts = []
        if "booking" in payload:
            parts.append(f"【會議室預約】\n{payload['booking']}")
        if "db" in payload:
            parts.append(f"【資料庫查詢】\n{payload['db']}")
        if "doc" in payload:
            parts.append(f"【文件查詢】\n{payload['doc']}")
        parts.append(token_block)
        final_msg = "\n\n".join(parts)

    # -------------------------------
    # 覆蓋最後一則 AI 回覆（避免累加）
    # -------------------------------
    # 先把舊的 AI 回覆刪除，只保留 HumanMessage
    state["messages"] = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    state["messages"].append(AIMessage(content=final_msg))

    logger.info("Responder 完成，輸出單一回覆。")
    return state


# ================================================================
# LangGraph Workflow
# ================================================================
def build_graph() -> StateGraph:
    g = StateGraph(EnterpriseState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("agents", agent_orchestrator_node)
    g.add_node("responder", responder_node)

    g.add_edge(START, "supervisor")
    g.add_edge("supervisor", "agents")
    g.add_edge("agents", "responder")
    g.add_edge("responder", END)

    return g


memory = MemorySaver()
app = build_graph().compile(checkpointer=memory)


# ================================================================
# 對外 async 入口
# ================================================================
async def arun_supervisor(question: str, thread_id: str = "cli") -> str:
    try:
        final_state = await app.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        msgs = final_state["messages"]
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        if not ai_msgs:
            return "目前沒有可用回覆。"
        return ai_msgs[-1].content
    except Exception as e:
        logger.error("arun_supervisor() 失敗：%s", e)
        return "系統暫時無法提供服務。"


# ================================================================
# CLI 測試
# ================================================================
async def cli_main():
    sid = "cli-session"
    while True:
        text = input("User> ").strip()
        if not text:
            break
        reply = await arun_supervisor(text, thread_id=sid)
        print("Agent>", reply)


if __name__ == "__main__":
    asyncio.run(cli_main())
