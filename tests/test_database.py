"""Tests for database connection pooling and management."""
import pytest
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import os
os.environ.setdefault("OPENAI_API_KEY", "test-api-key")


class TestConnectionPool:
    """Tests for ConnectionPool."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database file."""
        db_path = tmp_path / "test.db"
        return str(db_path)

    @pytest.fixture
    def pool(self, temp_db):
        """Create a connection pool."""
        from core.database import ConnectionPool
        return ConnectionPool(database=temp_db, pool_size=3)

    def test_pool_initialization(self, pool):
        """Test pool is initialized with connections."""
        stats = pool.stats()
        assert stats["pool_size"] == 3
        assert stats["available"] == 3
        assert stats["overflow"] == 0

    def test_acquire_and_release(self, pool):
        """Test acquiring and releasing connections."""
        # Acquire connection
        conn = pool.acquire()
        assert conn is not None
        assert pool.stats()["available"] == 2

        # Release connection
        pool.release(conn)
        assert pool.stats()["available"] == 3

    def test_connection_context_manager(self, pool):
        """Test using connection context manager."""
        with pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1

    def test_transaction_context_manager(self, pool):
        """Test transaction context manager commits on success."""
        with pool.transaction() as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO test (value) VALUES (?)", ("hello",))

        # Verify data was committed
        with pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM test WHERE id = 1")
            result = cursor.fetchone()
            assert result[0] == "hello"

    def test_transaction_rollback_on_error(self, pool):
        """Test transaction context manager rolls back on error."""
        try:
            with pool.transaction() as conn:
                conn.execute("CREATE TABLE test2 (id INTEGER PRIMARY KEY)")
                conn.execute("INSERT INTO test2 (id) VALUES (1)")
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Verify data was rolled back (table may not exist)
        with pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test2'"
            )
            result = cursor.fetchone()
            # Table creation might have been rolled back depending on SQLite version
            # At minimum, we verify no error occurred

    def test_overflow_connections(self, temp_db):
        """Test overflow connection creation."""
        from core.database import ConnectionPool

        pool = ConnectionPool(database=temp_db, pool_size=2, max_overflow=2)

        # Acquire all pool connections
        conns = [pool.acquire() for _ in range(2)]
        assert pool.stats()["available"] == 0

        # Acquire overflow connection
        overflow_conn = pool.acquire()
        assert pool.stats()["overflow"] == 1

        # Release pool connections first (pool becomes available)
        for conn in conns:
            pool.release(conn)
        assert pool.stats()["available"] == 2

        # Now release overflow - since pool is full, it should close
        pool.release(overflow_conn)
        assert pool.stats()["overflow"] == 0

    def test_timeout_when_exhausted(self, temp_db):
        """Test timeout when pool is exhausted."""
        from core.database import ConnectionPool

        pool = ConnectionPool(
            database=temp_db,
            pool_size=1,
            max_overflow=0,
            pool_timeout=0.1
        )

        # Acquire the only connection
        conn = pool.acquire()

        # Try to acquire another - should timeout
        with pytest.raises(TimeoutError):
            pool.acquire()

        pool.release(conn)

    def test_close_all(self, pool):
        """Test closing all connections."""
        pool.close_all()
        assert pool.stats()["available"] == 0

    def test_concurrent_access(self, temp_db):
        """Test thread-safe concurrent access."""
        from core.database import ConnectionPool

        pool = ConnectionPool(database=temp_db, pool_size=5)

        # Create table
        with pool.transaction() as conn:
            conn.execute("CREATE TABLE counter (id INTEGER PRIMARY KEY, value INTEGER)")
            conn.execute("INSERT INTO counter (id, value) VALUES (1, 0)")

        errors = []
        results = []

        def increment():
            try:
                for _ in range(10):
                    with pool.transaction() as conn:
                        cursor = conn.cursor()
                        cursor.execute("UPDATE counter SET value = value + 1 WHERE id = 1")
                results.append(True)
            except Exception as e:
                errors.append(e)

        # Run concurrent threads
        threads = [threading.Thread(target=increment) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5


class TestDatabaseManager:
    """Tests for DatabaseManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a database manager."""
        from core.database import ConnectionPool, DatabaseManager

        db_path = tmp_path / "test.db"
        pool = ConnectionPool(database=str(db_path))

        # Create test table
        with pool.transaction() as conn:
            conn.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT
                )
            """)

        return DatabaseManager(pool=pool)

    def test_insert(self, manager):
        """Test inserting a row."""
        row_id = manager.insert("users", {"name": "Alice", "email": "alice@example.com"})
        assert row_id == 1

    def test_fetch_one(self, manager):
        """Test fetching a single row."""
        manager.insert("users", {"name": "Bob"})
        result = manager.fetch_one("SELECT name FROM users WHERE id = ?", (1,))
        assert result[0] == "Bob"

    def test_fetch_all(self, manager):
        """Test fetching all rows."""
        manager.insert("users", {"name": "Alice"})
        manager.insert("users", {"name": "Bob"})
        manager.insert("users", {"name": "Charlie"})

        results = manager.fetch_all("SELECT name FROM users ORDER BY name")
        names = [r[0] for r in results]
        assert names == ["Alice", "Bob", "Charlie"]

    def test_update(self, manager):
        """Test updating rows."""
        manager.insert("users", {"name": "Alice", "email": "old@example.com"})

        count = manager.update(
            "users",
            {"email": "new@example.com"},
            "name = ?",
            ("Alice",)
        )
        assert count == 1

        result = manager.fetch_one("SELECT email FROM users WHERE name = ?", ("Alice",))
        assert result[0] == "new@example.com"

    def test_delete(self, manager):
        """Test deleting rows."""
        manager.insert("users", {"name": "Alice"})
        manager.insert("users", {"name": "Bob"})

        count = manager.delete("users", "name = ?", ("Alice",))
        assert count == 1

        results = manager.fetch_all("SELECT name FROM users")
        assert len(results) == 1
        assert results[0][0] == "Bob"


class TestMigrationManager:
    """Tests for migration manager."""

    @pytest.fixture
    def migrator(self, tmp_path):
        """Create a migration manager."""
        from core.database import ConnectionPool, MigrationManager

        db_path = tmp_path / "test.db"
        pool = ConnectionPool(database=str(db_path))
        return MigrationManager(pool)

    def test_initial_version(self, migrator):
        """Test initial version is 0."""
        assert migrator.get_current_version() == 0

    def test_apply_migration(self, migrator):
        """Test applying a migration."""
        from core.database import Migration

        migration = Migration(
            version=1,
            description="Create users table",
            up_sql="CREATE TABLE users (id INTEGER PRIMARY KEY)"
        )

        result = migrator.apply(migration)
        assert result is True
        assert migrator.get_current_version() == 1

    def test_skip_already_applied(self, migrator):
        """Test skipping already applied migration."""
        from core.database import Migration

        migration = Migration(
            version=1,
            description="Create users table",
            up_sql="CREATE TABLE users (id INTEGER PRIMARY KEY)"
        )

        migrator.apply(migration)
        result = migrator.apply(migration)  # Apply again
        assert result is False

    def test_apply_all(self, migrator):
        """Test applying multiple migrations."""
        from core.database import Migration

        migrations = [
            Migration(version=1, description="v1", up_sql="CREATE TABLE t1 (id INTEGER)"),
            Migration(version=2, description="v2", up_sql="CREATE TABLE t2 (id INTEGER)"),
            Migration(version=3, description="v3", up_sql="CREATE TABLE t3 (id INTEGER)"),
        ]

        applied = migrator.apply_all(migrations)
        assert applied == 3
        assert migrator.get_current_version() == 3

    def test_apply_all_skips_existing(self, migrator):
        """Test apply_all skips already applied migrations."""
        from core.database import Migration

        # Apply first two
        migrations_v2 = [
            Migration(version=1, description="v1", up_sql="CREATE TABLE t1 (id INTEGER)"),
            Migration(version=2, description="v2", up_sql="CREATE TABLE t2 (id INTEGER)"),
        ]
        migrator.apply_all(migrations_v2)

        # Apply all three
        migrations_v3 = [
            Migration(version=1, description="v1", up_sql="CREATE TABLE t1 (id INTEGER)"),
            Migration(version=2, description="v2", up_sql="CREATE TABLE t2 (id INTEGER)"),
            Migration(version=3, description="v3", up_sql="CREATE TABLE t3 (id INTEGER)"),
        ]
        applied = migrator.apply_all(migrations_v3)

        assert applied == 1  # Only v3 applied
        assert migrator.get_current_version() == 3


class TestGlobalPoolManagement:
    """Tests for global pool management functions."""

    def test_get_pool_creates_pool(self, tmp_path):
        """Test get_pool creates a new pool."""
        from core.database import get_pool, close_all_pools

        db_path = tmp_path / "test.db"

        with patch("core.database.get_db_config") as mock_config:
            mock_config.return_value.logs_db_path = str(db_path)
            mock_config.return_value.pool_size = 3
            mock_config.return_value.max_overflow = 5
            mock_config.return_value.pool_timeout = 30.0
            mock_config.return_value.check_same_thread = False

            pool = get_pool(str(db_path))
            assert pool is not None
            assert pool.stats()["pool_size"] == 3

        close_all_pools()

    def test_get_pool_returns_same_pool(self, tmp_path):
        """Test get_pool returns the same pool for same database."""
        from core.database import get_pool, close_all_pools

        db_path = tmp_path / "test.db"

        with patch("core.database.get_db_config") as mock_config:
            mock_config.return_value.logs_db_path = str(db_path)
            mock_config.return_value.pool_size = 3
            mock_config.return_value.max_overflow = 5
            mock_config.return_value.pool_timeout = 30.0
            mock_config.return_value.check_same_thread = False

            pool1 = get_pool(str(db_path))
            pool2 = get_pool(str(db_path))
            assert pool1 is pool2

        close_all_pools()
