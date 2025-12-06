"""
main.py — Enterprise Multi-Agent Supervisor v2
整合：
- LangChain Semantic Router（PydanticOutputParser）
- Slot-aware Supervisor
- Memory-aware Booking slot merge
- DB / Doc / Booking Agent pipeline
"""

from typing import (
    TypedDict, Annotated, Literal, Dict, Any, List, Optional, Tuple
)
from typing_extensions import assert_never

from configparser import ConfigParser

# LangGraph 1.0
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# LangChain
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Internal
from utils.logger import get_logger
from agents.db_agent import DatabaseAgent
from agents.doc_agent import DocumentAgent
from agents.booking_agent import BookingAgent


logger = get_logger(__name__)


# ================================================================
# Load LLM
# ================================================================
def load_llm():
    cfg = ConfigParser()
    cfg.read("config.ini", encoding="utf-8")

    provider = cfg.get("LLM", "provider", fallback="azure").lower()

    try:
        if provider == "azure":
            logger.info("初始化 AzureChatOpenAI 作為 Supervisor LLM")
            return AzureChatOpenAI(
                azure_endpoint=cfg.get("AZURE_OPENAI", "endpoint"),
                api_key=cfg.get("AZURE_OPENAI", "key"),
                azure_deployment=cfg.get("AZURE_OPENAI", "deployment"),
                api_version=cfg.get("AZURE_OPENAI", "api_version"),
                temperature=0,
            )
        else:
            logger.info("初始化 OpenAI ChatOpenAI 作為 Supervisor LLM")
            return ChatOpenAI(
                api_key=cfg.get("OPENAI", "api_key"),
                model=cfg.get("OPENAI", "model"),
                temperature=0,
            )

    except Exception as e:
        logger.error("Supervisor LLM 初始化失敗 → fallback keyword router：%s", e)
        return None


LLM = load_llm()

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
    route: Literal["db", "doc", "booking", "none"]
    confidence: float = Field(..., ge=0, le=1)
    slots: Optional[SlotModel] = None


router_parser = PydanticOutputParser(pydantic_object=RouterOutput)


router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是企業內部 AI 系統的 Supervisor Router。
請依照以下 JSON Schema 回覆，不要加入任何註解：

{format_instructions}

route 說明：
- db: 銷售、報表、統計、KPI、資料庫查詢
- doc: 辦法、制度、SOP、規範查詢
- booking: 會議室 / 時間 / 日期 / 人數
- none: 無法分類
"""
        ),
        ("human", "使用者輸入：{input}")
    ]
)

router_prompt = router_prompt.partial(
    format_instructions=router_parser.get_format_instructions()
)


# ================================================================
# Semantic Router
# ================================================================
def semantic_route(user_text: str) -> Optional[RouterOutput]:
    if LLM is None:
        return None

    try:
        chain = router_prompt | LLM | router_parser
        result: RouterOutput = chain.invoke({"input": user_text})

        if result.slots is None:
            result.slots = SlotModel()

        logger.info(
            "Semantic Router 命中：route=%s, conf=%.2f, slots=%s",
            result.route,
            result.confidence,
            result.slots.model_dump(),
        )
        return result

    except Exception as e:
        logger.warning("Semantic Router 執行失敗 → fallback keyword router：%s", e)
        return None


# ================================================================
# Keyword Router
# ================================================================
def keyword_router(text: str) -> str:
    t = text.lower()

    if any(k in t for k in ["銷售", "報表", "統計", "營收", "kpi", "sql"]):
        return "db"
    if any(k in t for k in ["辦法", "制度", "流程", "規範", "守則", "出差"]):
        return "doc"
    if any(k in t for k in ["會議室", "預約", "空房", "借用"]):
        return "booking"
    return "none"


# ================================================================
# Hybrid Router
# ================================================================
def classify_intent(text: str) -> Tuple[str, SlotModel]:
    s = semantic_route(text)
    if s and s.route != "none" and s.confidence >= 0.5:
        return s.route, s.slots
    return keyword_router(text), SlotModel()


# ================================================================
# State 定義
# ================================================================
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    route: str
    payload: Dict[str, Any]
    slots: Dict[str, Any]


SESSION_MESSAGES: Dict[str, List[AnyMessage]] = {}

# ================================================================
# Agents
# ================================================================
db_agent = DatabaseAgent()
doc_agent = DocumentAgent()
booking_agent = BookingAgent()


# ================================================================
# Supervisor Node
# ================================================================
def supervisor_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    if not isinstance(last, HumanMessage):
        state["route"] = "none"
        state["slots"] = {}
        return state

    text = last.content
    logger.info("Supervisor 收到輸入：%s", text)

    route, slot_obj = classify_intent(text)
    slots = slot_obj.model_dump()

    state["route"] = route
    state["slots"] = slots

    logger.info("Supervisor 判斷 route=%s, slots=%s", route, slots)
    return state


# ================================================================
# Booking Slot Merge（關鍵）
# ================================================================
def merge_booking_slots(new_slots: Dict[str, Any], prev_slots: Dict[str, Any]):
    merged = {
        "date": new_slots.get("date") or prev_slots.get("date"),
        "time": new_slots.get("time") or prev_slots.get("time"),
        "people": new_slots.get("people") or prev_slots.get("people"),
    }
    return merged


# ================================================================
# Agent Nodes
# ================================================================
def booking_agent_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    new_slots = state.get("slots", {})

    prev = state["payload"].get("booking", {})
    merged_slots = merge_booking_slots(new_slots, prev)

    logger.info(f"BookingAgent merged_slots={merged_slots}")

    try:
        result = booking_agent.run(last.content, slots=merged_slots)

        state["payload"]["booking"] = {
            "date": merged_slots["date"],
            "time": merged_slots["time"],
            "people": merged_slots["people"],
        }

        message = result.get("message")
        state["messages"].append(AIMessage(content=message))

    except Exception as e:
        logger.error("Booking Agent 失敗：%s", e)
        state["messages"].append(AIMessage(content="會議室系統目前無法使用。"))

    return state



def db_agent_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    slots = state["slots"]

    result = db_agent.run(last.content, slots=slots)
    state["payload"]["db"] = result

    msg = result.get("answer") or result.get("message")
    state["messages"].append(AIMessage(content=msg))
    return state


def doc_agent_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    slots = state["slots"]

    result = doc_agent.run(last.content, slots=slots)
    state["payload"]["doc"] = result

    state["messages"].append(AIMessage(content=result.get("answer")))
    return state


# ================================================================
# Routing Edge
# ================================================================
def route_from_supervisor(state: AgentState) -> str:
    r = state["route"]
    if r == "db":
        return "db_agent_node"
    if r == "doc":
        return "doc_agent_node"
    if r == "booking":
        return "booking_agent_node"
    if r == "none":
        return "responder"
    assert_never(r)


def responder_node(state: AgentState) -> AgentState:
    logger.info("Responder 收到 payload keys=%s", list(state["payload"].keys()))
    return state


# ================================================================
# Build Graph
# ================================================================
def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("db_agent_node", db_agent_node)
    g.add_node("doc_agent_node", doc_agent_node)
    g.add_node("booking_agent_node", booking_agent_node)
    g.add_node("responder", responder_node)

    g.add_edge(START, "supervisor")

    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "db_agent_node": "db_agent_node",
            "doc_agent_node": "doc_agent_node",
            "booking_agent_node": "booking_agent_node",
            "responder": "responder",
        },
    )

    g.add_edge("db_agent_node", "responder")
    g.add_edge("doc_agent_node", "responder")
    g.add_edge("booking_agent_node", "responder")
    g.add_edge("responder", END)

    return g


_app = build_graph().compile(checkpointer=None)

# ================================================================
# run_supervisor（含對話記憶）
# ================================================================
def run_supervisor(question: str, session_id: str = "default") -> str:
    try:
        history = SESSION_MESSAGES.get(session_id, [])
        new_messages = history + [HumanMessage(content=question)]

        init_state: AgentState = {
            "messages": new_messages,
            "route": "none",
            "payload": {},
            "slots": {},
        }

        final_state = _app.invoke(init_state)

        SESSION_MESSAGES[session_id] = final_state["messages"]

        ai_msgs = [m for m in final_state["messages"] if isinstance(m, AIMessage)]
        if not ai_msgs:
            return "目前沒有可用回覆。"

        return ai_msgs[-1].content

    except Exception as e:
        logger.error("run_supervisor() 失敗：%s", e)
        return "系統暫時無法提供服務。"


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    sid = "cli"
    while True:
        text = input("User> ").strip()
        if not text:
            break
        reply = run_supervisor(text, session_id=sid)
        print("Agent>", reply)

