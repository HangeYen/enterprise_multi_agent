"""
main.py

- 建立 LangGraph 1.0 多 Agent 工作流
- 節點：
  - supervisor
  - db_agent_node
  - doc_agent_node
  - booking_agent_node
  - responder
- 提供 run_supervisor(question: str) 入口給 Portal / CLI 使用
"""

from typing import TypedDict, Annotated, Literal, Dict, Any, List

from typing_extensions import assert_never

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

from utils.logger import get_logger
from agents.db_agent import DatabaseAgent
from agents.doc_agent import DocumentAgent
from agents.booking_agent import BookingAgent

logger = get_logger(__name__)


class AgentState(TypedDict):
    """
    LangGraph 狀態結構。

    messages:
        - 對話歷史，使用 add_messages 自動累積。
    route:
        - supervisor 決定的下一個 agent 名稱：
          "db" / "doc" / "booking" / "none"
    payload:
        - 各 Agent 回傳的結構化資料。
    """
    messages: Annotated[List[AnyMessage], add_messages]
    route: Literal["db", "doc", "booking", "none"]
    payload: Dict[str, Any]


db_agent = DatabaseAgent()
doc_agent = DocumentAgent()
booking_agent = BookingAgent()


def supervisor_node(state: AgentState) -> AgentState:
    """
    Supervisor 節點

    - 查看最後一則 HumanMessage
    - 目前用關鍵字作 Intent Routing（之後可換成 LLM 分類）
    """
    messages = state["messages"]
    last = messages[-1]
    if not isinstance(last, HumanMessage):
        logger.info("Last message is not HumanMessage, route=none")
        state["route"] = "none"
        return state

    text = last.content.lower()
    logger.info("Supervisor received text: %s", text)

    if "銷售" in text or "報表" in text or "sql" in text:
        route: Literal["db"] = "db"
    elif "出差" in text or "辦法" in text or "規定" in text:
        route = "doc"  # type: ignore[assignment]
    elif "會議室" in text or "預約" in text:
        route = "booking"  # type: ignore[assignment]
    else:
        route = "doc"  # type: ignore[assignment]

    state["route"] = route  # type: ignore[assignment]
    logger.info("Supervisor decided route=%s", route)
    return state


def db_agent_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    assert isinstance(last, HumanMessage)
    result = db_agent.run(last.content)

    state["payload"]["db"] = result
    state["messages"].append(
        AIMessage(content="以下是依據資料庫查詢得到的分析結果（目前為示範骨架）。")
    )
    return state


def doc_agent_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    assert isinstance(last, HumanMessage)
    result = doc_agent.run(last.content)

    state["payload"]["doc"] = result
    state["messages"].append(
        AIMessage(content=result["answer"])
    )
    return state


def booking_agent_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    assert isinstance(last, HumanMessage)
    result = booking_agent.run(last.content)

    state["payload"]["booking"] = result
    state["messages"].append(
        AIMessage(content=result["message"])
    )
    return state


def route_from_supervisor(state: AgentState) -> str:
    """
    Conditional Edge 的 routing 函式。
    """
    route = state["route"]
    if route == "db":
        return "db_agent_node"
    if route == "doc":
        return "doc_agent_node"
    if route == "booking":
        return "booking_agent_node"
    if route == "none":
        return "responder"
    assert_never(route)


def responder_node(state: AgentState) -> AgentState:
    """
    Responder 節點

    - 目前只負責結束流程，可在此統整 payload→最終答案
    - 若未來要組合多 Agent 結果，可在這裡做統一格式化
    """
    logger.info("Responder reached. Payload keys: %s", list(state["payload"].keys()))
    return state


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("db_agent_node", db_agent_node)
    graph.add_node("doc_agent_node", doc_agent_node)
    graph.add_node("booking_agent_node", booking_agent_node)
    graph.add_node("responder", responder_node)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "db_agent_node": "db_agent_node",
            "doc_agent_node": "doc_agent_node",
            "booking_agent_node": "booking_agent_node",
            "responder": "responder",
        },
    )

    graph.add_edge("db_agent_node", "responder")
    graph.add_edge("doc_agent_node", "responder")
    graph.add_edge("booking_agent_node", "responder")

    graph.add_edge("responder", END)

    return graph


_app = build_graph().compile()


def run_supervisor(question: str) -> str:
    """
    對外提供的簡化入口。

    - Portal 或 CLI 只需呼叫此函式
    - 回傳最後一則 AIMessage 的內容
    """
    initial_state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "route": "none",
        "payload": {},
    }

    final_state = _app.invoke(initial_state)
    messages = final_state["messages"]

    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    if not ai_messages:
        return "目前沒有可用的回覆。"

    return ai_messages[-1].content


if __name__ == "__main__":
    # 簡單 CLI 測試入口
    while True:
        text = input("User> ")
        if not text:
            break
        answer = run_supervisor(text)
        print("Agent>", answer)
