import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

class Settings:
    #Database Settings
    DATABASE_URL: str = os.getenv("MYSQL_URL", "").replace(
        "mysql://", "mysql+pymysql://"
    )
    # DB_USER: str = os.getenv("DB_USER", "")
    # DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    # DB_HOST: str = os.getenv("DB_HOST", "")
    # DB_PORT: str = os.getenv("DB_PORT", "")
    # DB_NAME: str = os.getenv("DB_NAME", "")

    # DATABASE_URL: str = (
    #     f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASSWORD)}"
    #     f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    # )

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     if not self.DATABASE_URL:
    #         self.DATABASE_URL = (
    #             f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}"
    #             f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    #         )

    class Config:
        env_file = ".env"


    # JWT
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "")
    JWT_EXPIRATION_MINUTES: int = int(os.getenv("JWT_EXPIRATION_MINUTES", "60"))

    # Roboflow
    ROBOFLOW_API_KEY: str = os.getenv("ROBOFLOW_API_KEY", "")
    ROBOFLOW_MODEL_ID: str = os.getenv("ROBOFLOW_MODEL_ID", "")
    ROBOFLOW_API_URL: str = os.getenv("ROBOFLOW_API_URL", "")
    FRACTURE_MODEL_ID: str = os.getenv("FRACTURE_MODEL_ID", "")

    # Model
    RESNET_MODEL_PATH: str = os.getenv("RESNET_MODEL_PATH", "")

    # File storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "")


settings = Settings()

print("mysql_url:", settings.DATABASE_URL)