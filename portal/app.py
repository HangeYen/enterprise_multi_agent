from flask import Flask, request, render_template, redirect, url_for

from auth.auth import AuthService
from utils.logger import get_logger
from main import run_supervisor  # 從 main.py 匯入入口

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)


logger = get_logger(__name__)

app = Flask(__name__)
auth_service = AuthService()


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    簡易登入頁：
    - GET: 顯示登入表單
    - POST: 驗證帳密，成功後導向 /chat
    """
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        token = auth_service.login(username, password)

        if token is None:
            return "Login failed", 401

        # 簡化作法：token 放在 query string
        return redirect(url_for("chat", token=token))

    return render_template("login.html")


@app.route("/chat")
def chat():
    token = request.args.get("token")
    if not token:
        return redirect(url_for("login"))
    return render_template("chat.html", token=token)


@app.route("/ask", methods=["POST"])
def ask():
    """
    前端呼叫此 API，將問題送給 Supervisor。

    回傳 JSON：
    {
        "answer": "<最後一則 AI 回覆內容>"
    }
    """
    data = request.get_json() or {}
    question = data.get("question", "")
    logger.info("Portal received question: %s", question)

    answer = run_supervisor(question)
    return {"answer": answer}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
