"""
Database Connection Pooling and Management for Kortix

Provides connection pooling, context managers, and transaction support
for SQLite databases used throughout the framework.
"""
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Any, Dict, Generator, List, Optional, Tuple
import time

from .config import get_db_config


class ConnectionPool:
    """
    Thread-safe SQLite connection pool.

    Manages a pool of database connections that can be reused
    across multiple operations, reducing connection overhead.
    """

    def __init__(
        self,
        database: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: float = 30.0,
        check_same_thread: bool = False,
    ):
        """
        Initialize connection pool.

        Args:
            database: Path to SQLite database file
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum additional connections when pool exhausted
            pool_timeout: Seconds to wait for available connection
            check_same_thread: SQLite check_same_thread parameter
        """
        self.database = database
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.check_same_thread = check_same_thread

        self._pool: Queue = Queue(maxsize=pool_size)
        self._overflow_count = 0
        self._lock = threading.Lock()
        self._initialized = False

        # Pre-populate pool
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Pre-populate the connection pool."""
        if self._initialized:
            return

        for _ in range(self.pool_size):
            conn = self._create_connection()
            self._pool.put(conn)

        self._initialized = True

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(
            self.database,
            check_same_thread=self.check_same_thread,
            timeout=self.pool_timeout,
        )
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def acquire(self) -> sqlite3.Connection:
        """
        Acquire a connection from the pool.

        Returns:
            sqlite3.Connection: Database connection

        Raises:
            TimeoutError: If no connection available within timeout
        """
        try:
            conn = self._pool.get(timeout=self.pool_timeout)
            # Test connection is still valid
            try:
                conn.execute("SELECT 1")
                return conn
            except sqlite3.Error:
                # Connection is stale, create new one
                return self._create_connection()
        except Empty:
            # Pool exhausted, try overflow
            with self._lock:
                if self._overflow_count < self.max_overflow:
                    self._overflow_count += 1
                    return self._create_connection()

            raise TimeoutError(
                f"Connection pool exhausted. Pool size: {self.pool_size}, "
                f"overflow: {self._overflow_count}/{self.max_overflow}"
            )

    def release(self, conn: sqlite3.Connection) -> None:
        """
        Release a connection back to the pool.

        Args:
            conn: Connection to release
        """
        try:
            # Rollback any uncommitted transaction
            conn.rollback()

            if self._pool.full():
                # Pool is full, close overflow connection
                with self._lock:
                    if self._overflow_count > 0:
                        self._overflow_count -= 1
                conn.close()
            else:
                self._pool.put_nowait(conn)
        except Exception:
            # If anything goes wrong, just close the connection
            try:
                conn.close()
            except Exception:
                pass

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for acquiring and releasing connections.

        Usage:
            with pool.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        """
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for transactions with automatic commit/rollback.

        Usage:
            with pool.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO table VALUES (?)", (value,))
                # Commits automatically on success
                # Rolls back automatically on exception
        """
        conn = self.acquire()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.release(conn)

    def close_all(self) -> None:
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break

    def stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            "pool_size": self.pool_size,
            "available": self._pool.qsize(),
            "overflow": self._overflow_count,
            "max_overflow": self.max_overflow,
        }


# Global pool instances
_pools: Dict[str, ConnectionPool] = {}
_pools_lock = threading.Lock()


def get_pool(database: Optional[str] = None) -> ConnectionPool:
    """
    Get or create a connection pool for a database.

    Args:
        database: Path to database file (uses config default if None)

    Returns:
        ConnectionPool instance
    """
    if database is None:
        database = get_db_config().logs_db_path

    with _pools_lock:
        if database not in _pools:
            config = get_db_config()
            _pools[database] = ConnectionPool(
                database=database,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow,
                pool_timeout=config.pool_timeout,
                check_same_thread=config.check_same_thread,
            )
        return _pools[database]


def close_all_pools() -> None:
    """Close all connection pools (call during shutdown)."""
    with _pools_lock:
        for pool in _pools.values():
            pool.close_all()
        _pools.clear()


@dataclass
class DatabaseManager:
    """
    High-level database operations manager.

    Provides convenient methods for common database operations
    with automatic connection management.
    """

    pool: ConnectionPool

    def execute(
        self,
        query: str,
        params: Tuple = (),
        commit: bool = False,
    ) -> sqlite3.Cursor:
        """Execute a query and return cursor."""
        with self.pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if commit:
                conn.commit()
            return cursor

    def execute_many(
        self,
        query: str,
        params_list: List[Tuple],
        commit: bool = True,
    ) -> int:
        """Execute a query with multiple parameter sets."""
        with self.pool.transaction() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            return cursor.rowcount

    def fetch_one(
        self,
        query: str,
        params: Tuple = (),
    ) -> Optional[Tuple]:
        """Fetch a single row."""
        with self.pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()

    def fetch_all(
        self,
        query: str,
        params: Tuple = (),
    ) -> List[Tuple]:
        """Fetch all rows."""
        with self.pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def insert(
        self,
        table: str,
        data: Dict[str, Any],
    ) -> int:
        """Insert a row and return lastrowid."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        with self.pool.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(data.values()))
            return cursor.lastrowid

    def update(
        self,
        table: str,
        data: Dict[str, Any],
        where: str,
        where_params: Tuple = (),
    ) -> int:
        """Update rows and return affected count."""
        set_clause = ", ".join(f"{k} = ?" for k in data.keys())
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"
        params = tuple(data.values()) + where_params

        with self.pool.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount

    def delete(
        self,
        table: str,
        where: str,
        where_params: Tuple = (),
    ) -> int:
        """Delete rows and return affected count."""
        query = f"DELETE FROM {table} WHERE {where}"

        with self.pool.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, where_params)
            return cursor.rowcount


def get_database_manager(database: Optional[str] = None) -> DatabaseManager:
    """Get a database manager instance."""
    return DatabaseManager(pool=get_pool(database))


# Migration support
@dataclass
class Migration:
    """Database migration definition."""
    version: int
    description: str
    up_sql: str
    down_sql: Optional[str] = None


class MigrationManager:
    """Manages database schema migrations."""

    def __init__(self, pool: ConnectionPool):
        self.pool = pool
        self._ensure_migrations_table()

    def _ensure_migrations_table(self) -> None:
        """Create migrations tracking table if not exists."""
        with self.pool.transaction() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _migrations (
                    version INTEGER PRIMARY KEY,
                    description TEXT,
                    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def get_current_version(self) -> int:
        """Get the current migration version."""
        with self.pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(version) FROM _migrations")
            result = cursor.fetchone()
            return result[0] if result[0] is not None else 0

    def apply(self, migration: Migration) -> bool:
        """Apply a single migration."""
        current = self.get_current_version()
        if migration.version <= current:
            return False

        with self.pool.transaction() as conn:
            conn.executescript(migration.up_sql)
            conn.execute(
                "INSERT INTO _migrations (version, description) VALUES (?, ?)",
                (migration.version, migration.description)
            )
        return True

    def apply_all(self, migrations: List[Migration]) -> int:
        """Apply all pending migrations."""
        applied = 0
        # Sort by version
        sorted_migrations = sorted(migrations, key=lambda m: m.version)
        for migration in sorted_migrations:
            if self.apply(migration):
                applied += 1
        return applied


# Schema definitions for Kortix tables
LOGS_SCHEMA = Migration(
    version=1,
    description="Create logs and sessions tables",
    up_sql="""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_id TEXT UNIQUE NOT NULL,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT,
            unit_name TEXT,
            parent_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_logs_session ON logs(session_id);
        CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level);

        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            status TEXT DEFAULT 'active',
            metadata TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
    """,
)

WORKING_MEMORY_SCHEMA = Migration(
    version=1,
    description="Create working memory tables",
    up_sql="""
        CREATE TABLE IF NOT EXISTS MemoryModules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            module_name TEXT UNIQUE NOT NULL,
            data TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_memory_module ON MemoryModules(module_name);

        CREATE TRIGGER IF NOT EXISTS update_memory_timestamp
        AFTER UPDATE ON MemoryModules
        BEGIN
            UPDATE MemoryModules SET updated_at = CURRENT_TIMESTAMP
            WHERE id = NEW.id;
        END;
    """,
)


def initialize_databases() -> None:
    """Initialize all databases with their schemas."""
    config = get_db_config()

    # Initialize logs database
    logs_pool = get_pool(config.logs_db_path)
    logs_migrator = MigrationManager(logs_pool)
    logs_migrator.apply(LOGS_SCHEMA)

    # Initialize working memory database
    wm_pool = get_pool(config.working_memory_db_path)
    wm_migrator = MigrationManager(wm_pool)
    wm_migrator.apply(WORKING_MEMORY_SCHEMA)
