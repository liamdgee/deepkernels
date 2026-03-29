
from sqlalchemy.ext.asyncio import create_async_engine, AsyncConnection
from sqlalchemy import event

DATABASE_URL = "sqlite+aiosqlite:///./deepkernels_metrics.db"

engine = create_async_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()

async def get_db():
    """
    Yields a safe, async database connection to FastAPI routes.
    (Optimized for read-only telemetry)
    """
    async with engine.connect() as conn:
        yield conn