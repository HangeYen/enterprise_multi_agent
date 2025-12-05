from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from configparser import ConfigParser

from utils.logger import get_logger
from tools.vector_tools import vector_query

logger = get_logger(__name__)


def _load_llm_from_config(config_path: str = "config.ini"):
    """
    根據 config.ini 讀取 [LLM] + 對應 Provider 設定，建立 LLM 物件。

    - provider = azure → 使用 AzureChatOpenAI
    - provider = openai → 使用 ChatOpenAI
    - provider = gemini → 可以留作未來擴充
    """
    cfg = ConfigParser()
    cfg.read(config_path, encoding="utf-8")

    provider = cfg.get("LLM", "provider", fallback="azure").lower()
    temperature = cfg.getfloat("LLM", "temperature", fallback=0.2)

    if provider == "azure":
        endpoint = cfg.get("AZURE_OPENAI", "endpoint")
        api_key = cfg.get("AZURE_OPENAI", "key")
        deployment = cfg.get("AZURE_OPENAI", "deployment")
        api_version = cfg.get("AZURE_OPENAI", "api_version")

        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            azure_deployment=deployment,
            api_version=api_version,
            temperature=temperature,
        )
        return llm

    if provider == "openai":
        api_key = cfg.get("OPENAI", "api_key")
        model = cfg.get("OPENAI", "model", fallback="gpt-4o-mini")
        llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
        return llm

    # 預設使用 azure，避免無法運行
    endpoint = cfg.get("AZURE_OPENAI", "endpoint")
    api_key = cfg.get("AZURE_OPENAI", "key")
    deployment = cfg.get("AZURE_OPENAI", "deployment")
    api_version = cfg.get("AZURE_OPENAI", "api_version")

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        azure_deployment=deployment,
        api_version=api_version,
        temperature=temperature,
    )


class DocumentAgent:
    """
    Document Agent

    負責：
    - 呼叫向量資料庫查詢相關文件片段（tools.vector_tools.vector_query）
    - 使用 LangChain 建立簡單的 RAG Chain（context + question → answer）
    """

    def __init__(self, config_path: str = "config.ini") -> None:
        self.config_path = config_path
        self.llm = _load_llm_from_config(config_path)

        # RAG Prompt 範例，可依企業風格修改
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一位企業內部知識庫助理，請依據提供的文件內容回答問題。"
                    "若文件中沒有明確提到，請老實說不知道，不要捏造。",
                ),
                (
                    "human",
                    "以下是與問題相關的文件片段：\n\n{context}\n\n"
                    "問題：{question}\n\n"
                    "請以條列式輸出重點說明。",
                ),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _build_context(self, docs: List[Dict[str, Any]]) -> str:
        """
        將 vector_query 回傳的多筆 file chunk 組成一個 context 字串。
        """
        contents = [d.get("content", "") for d in docs]
        return "\n\n".join(contents)

    def run(self, question: str) -> Dict[str, Any]:
        logger.info("DocumentAgent received question: %s", question)

        # 1) 向向量庫查詢相關文件 chunk
        docs = vector_query(question)
        context = self._build_context(docs)

        if not context.strip():
            dummy_answer = "目前文件庫中找不到與此問題相關的內容。"
            return {
                "type": "doc_result",
                "answer": dummy_answer,
                "source_count": 0,
            }

        # 2) LangChain RAG Chain
        answer = self.chain.invoke({"context": context, "question": question})

        return {
            "type": "doc_result",
            "answer": answer,
            "source_count": len(docs),
        }
