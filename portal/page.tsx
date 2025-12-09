"use client";

import Header from "@/components/Header";
import TemplateButtons from "@/components/TemplateButtons";
import AIMessage from "@/components/AIMessage";
import { useState, useRef } from "react";
import { Send } from "lucide-react";

export default function ChatPage() {
  const initialInputRef = useRef<HTMLTextAreaElement | null>(null);
  const chatInputRef = useRef<HTMLTextAreaElement | null>(null);

  const [showTemplates, setShowTemplates] = useState(true);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<any[]>([]);

  function handleSendMessage() {
    if (!input.trim()) return;

    // 切換為聊天模式
    if (showTemplates) setShowTemplates(false);

    const userMsg = input;
    setInput("");

    // 重置高度
    if (initialInputRef.current) initialInputRef.current.style.height = "auto";
    if (chatInputRef.current) chatInputRef.current.style.height = "auto";

    setMessages((prev) => [
      ...prev,
      { type: "user", content: userMsg },
    ]);

    // 假 AI 回覆
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          type: "ai",
          summary: "這是摘要標題",
          content: "這裡是 AI 回覆內容，之後會接你的多 Agent 回覆結果。",
          insights: ["Insight 1", "Insight 2"],
        },
      ]);
    }, 500);
  }

  return (
    <div className="flex flex-col h-screen bg-white text-gray-900">
      <Header />

      {/* === 初始畫面（置中輸入框） === */}
      {showTemplates && messages.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center px-6">

          <h2 className="text-2xl font-semibold text-gray-900 mb-2">
            Charlie&apos;s Agents
          </h2>
          <p className="text-gray-500 mb-6">請問今天想查詢什麼？</p>

          <TemplateButtons onSelect={setInput} />

          {/* 置中輸入框 */}
          <div className="w-full max-w-2xl flex items-center gap-3 border border-gray-300 rounded-full px-5 py-4 mt-6">
            <textarea
              ref={initialInputRef}
              value={input}
              placeholder="提出任何問題…"
              onChange={(e) => {
                setInput(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${e.target.scrollHeight}px`;
              }}
              className="flex-1 bg-transparent focus:outline-none resize-none"
              rows={1}
            />

            <button
              onClick={handleSendMessage}
              className="p-3 bg-blue-600 text-white rounded-full"
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      )}

      {/* === Chat 訊息列表 === */}
      {!showTemplates && (
        <div className="flex-1 overflow-y-auto p-6 w-full max-w-3xl mx-auto">
          {messages.map((msg, idx) =>
            msg.type === "ai" ? (
              <AIMessage
                key={idx}
                summary={msg.summary}
                content={msg.content}
                insights={msg.insights}
              />
            ) : (
              <div key={idx} className="mb-4 flex justify-end">
                <div
                    className="
                    bg-blue-100 text-gray-900 
                    px-4 py-2 rounded-xl 
                    max-w-[70%] whitespace-pre-wrap text-left
                    "
                >
                    {msg.content}
                </div>
              </div>
            )
          )}
        </div>
      )}

      {/* === 底部聊天輸入框（只有聊天模式顯示） === */}
      {!showTemplates && (
        <div className="w-full border-t border-gray-200 p-4 flex justify-center bg-white">
          <div className="w-full max-w-3xl flex items-end gap-3">

            <textarea
              ref={chatInputRef}
              value={input}
              placeholder="提出任何問題…"
              onChange={(e) => {
                setInput(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${e.target.scrollHeight}px`;
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              className="flex-1 p-3 border border-gray-300 rounded-xl resize-none
                         max-h-48 overflow-y-auto bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              rows={1}
            />

            <button
              onClick={handleSendMessage}
              className="p-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700"
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
