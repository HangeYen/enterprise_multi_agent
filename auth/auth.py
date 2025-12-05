from typing import Optional
from dataclasses import dataclass
from configparser import ConfigParser

from pymongo import MongoClient

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class User:
    """簡單的使用者資料結構，可依需求擴充欄位。"""
    username: str


class AuthService:
    """
    登入與 Token 驗證服務層。

    - 使用 MongoDB 儲存使用者帳號資料
    - 目前僅示範檢查使用者是否存在，不含密碼雜湊
    - 後續可改用 JWT / password hashing
    """

    def __init__(self, config_path: str = "config.ini") -> None:
        cfg = ConfigParser()
        cfg.read(config_path, encoding="utf-8")

        conn_str = cfg.get("MONGODB", "connection_string", fallback="mongodb://localhost:27017")
        db_name = cfg.get("MONGODB", "database", fallback="enterprise_ai")
        coll_name = cfg.get("MONGODB", "user_collection", fallback="users")

        self.client = MongoClient(conn_str)
        self.db = self.client[db_name]
        self.collection = self.db[coll_name]

    def login(self, username: str, password: str) -> Optional[str]:
        """
        驗證帳號密碼。

        - 目前只檢查 user 是否存在，不驗證密碼（示範用）
        - 回傳模擬 token 字串，未使用 JWT
        """
        logger.info("Login attempt for user=%s", username)
        user = self.collection.find_one({"username": username})

        if not user:
            logger.info("User not found")
            return None

        # TODO: 實務上請改為檢查密碼雜湊
        token = f"dummy-token-{username}"
        return token

    def verify_token(self, token: str) -> Optional[User]:
        """
        驗證 token 合法性。

        - 範例中使用簡單字串前綴判斷
        - 實務上請改用 JWT 驗證
        """
        if not token.startswith("dummy-token-"):
            return None
        username = token.replace("dummy-token-", "", 1)
        return User(username=username)
