import os
from datetime import datetime
from sqlalchemy import Column, BigInteger, String, DateTime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.future import select

# --- CONFIG ---
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")
POSTGRES_DB = os.getenv("POSTGRES_DB", "epistemic_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, index=True) # Telegram ID
    username = Column(String, nullable=True)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)

class DB:
    def __init__(self, url=DATABASE_URL):
        self.engine = create_async_engine(url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def register_or_update_user(self, telegram_id: int, username: str, full_name: str):
        async with self.async_session() as session:
            result = await session.execute(select(User).where(User.id == telegram_id))
            user = result.scalar_one_or_none()

            if user:
                user.last_active = datetime.utcnow()
                user.username = username
                user.full_name = full_name
            else:
                user = User(
                    id=telegram_id,
                    username=username,
                    full_name=full_name
                )
                session.add(user)

            await session.commit()
            return user

db = DB()
