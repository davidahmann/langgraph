# type: ignore

import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event
from time import monotonic, sleep
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    EXCLUDED_METADATA_KEYS,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.serde.types import TASKS
from psycopg import Connection, IsolationLevel
from psycopg.errors import SerializationFailure
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langgraph.checkpoint.postgres import PostgresSaver, ShallowPostgresSaver
from tests.conftest import DEFAULT_POSTGRES_URI


def _exclude_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in EXCLUDED_METADATA_KEYS}


@contextmanager
def _pool_saver():
    """Fixture for pool mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        with ConnectionPool(
            DEFAULT_POSTGRES_URI + database,
            max_size=10,
            kwargs={"autocommit": True, "row_factory": dict_row},
        ) as pool:
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


@contextmanager
def _pipe_saver():
    """Fixture for pipeline mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        with Connection.connect(
            DEFAULT_POSTGRES_URI + database,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()
            with conn.pipeline() as pipe:
                checkpointer = PostgresSaver(conn, pipe=pipe)
                yield checkpointer
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


@contextmanager
def _base_saver():
    """Fixture for regular connection mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        with Connection.connect(
            DEFAULT_POSTGRES_URI + database,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


@contextmanager
def _shallow_saver():
    """Fixture for regular connection mode testing with a shallow checkpointer."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        with Connection.connect(
            DEFAULT_POSTGRES_URI + database,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            checkpointer = ShallowPostgresSaver(conn)
            checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


@contextmanager
def _saver(name: str):
    if name in ("base", "fallback"):
        with _base_saver() as saver:
            if name == "fallback":
                saver.supports_pipeline = False
            yield saver
    elif name == "shallow":
        with _shallow_saver() as saver:
            yield saver
    elif name == "pool":
        with _pool_saver() as saver:
            yield saver
    elif name == "pipe":
        with _pipe_saver() as saver:
            yield saver


@pytest.fixture
def test_data():
    """Fixture providing test data for checkpoint tests."""
    config_1: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_id": "1",
            "checkpoint_ns": "",
        }
    }
    config_2: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2",
            "checkpoint_ns": "",
        }
    }
    config_3: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2-inner",
            "checkpoint_ns": "inner",
        }
    }

    chkpnt_1: Checkpoint = empty_checkpoint()
    chkpnt_2: Checkpoint = create_checkpoint(chkpnt_1, {}, 1)
    chkpnt_3: Checkpoint = empty_checkpoint()

    metadata_1: CheckpointMetadata = {
        "source": "input",
        "step": 2,
        "score": 1,
    }
    metadata_2: CheckpointMetadata = {
        "source": "loop",
        "step": 1,
        "score": None,
    }
    metadata_3: CheckpointMetadata = {}

    return {
        "configs": [config_1, config_2, config_3],
        "checkpoints": [chkpnt_1, chkpnt_2, chkpnt_3],
        "metadata": [metadata_1, metadata_2, metadata_3],
    }


@pytest.mark.parametrize("saver_name", ["base", "pool", "pipe", "shallow"])
def test_combined_metadata(saver_name: str, test_data) -> None:
    with _saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_ns": "",
                "__super_private_key": "super_private_value",
            },
            "metadata": {"run_id": "my_run_id"},
        }
        chkpnt: Checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
        metadata: CheckpointMetadata = {
            "source": "loop",
            "step": 1,
            "score": None,
        }
        saver.put(config, chkpnt, metadata, {})
        checkpoint = saver.get_tuple(config)
        assert checkpoint.metadata == {
            **metadata,
            "run_id": "my_run_id",
        }


@pytest.mark.parametrize("saver_name", ["base", "pool", "pipe", "shallow"])
def test_search(saver_name: str, test_data) -> None:
    with _saver(saver_name) as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]

        saver.put(configs[0], checkpoints[0], metadata[0], {})
        saver.put(configs[1], checkpoints[1], metadata[1], {})
        saver.put(configs[2], checkpoints[2], metadata[2], {})

        # call method / assertions
        query_1 = {"source": "input"}  # search by 1 key
        query_2 = {
            "step": 1,
        }  # search by multiple keys
        query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
        query_4 = {"source": "update", "step": 1}  # no match

        search_results_1 = list(saver.list(None, filter=query_1))
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == {
            **_exclude_keys(configs[0]["configurable"]),
            **metadata[0],
        }

        search_results_2 = list(saver.list(None, filter=query_2))
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == {
            **_exclude_keys(configs[1]["configurable"]),
            **metadata[1],
        }

        search_results_3 = list(saver.list(None, filter=query_3))
        assert len(search_results_3) == 3

        search_results_4 = list(saver.list(None, filter=query_4))
        assert len(search_results_4) == 0

        # search by config (defaults to checkpoints across all namespaces)
        search_results_5 = list(saver.list({"configurable": {"thread_id": "thread-2"}}))
        assert len(search_results_5) == 2
        assert {
            search_results_5[0].config["configurable"]["checkpoint_ns"],
            search_results_5[1].config["configurable"]["checkpoint_ns"],
        } == {"", "inner"}


@pytest.mark.parametrize("saver_name", ["base", "pool", "pipe", "fallback"])
def test_delete_thread_ignores_late_writes(saver_name: str) -> None:
    with _saver(saver_name) as saver:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-delete",
                "checkpoint_ns": "",
            }
        }

        stored = saver.put(config, empty_checkpoint(), {}, {})
        saver.delete_thread("thread-delete")

        saver.put(stored, empty_checkpoint(), {"step": 99}, {})
        saver.put_writes(stored, [("ch", "late-write")], str(uuid4()))

        assert saver.get_tuple({"configurable": {"thread_id": "thread-delete"}}) is None
        assert list(saver.list({"configurable": {"thread_id": "thread-delete"}})) == []


@pytest.mark.parametrize("saver_name", ["base", "pool", "pipe", "fallback"])
@pytest.mark.parametrize("operation", ["put", "put_writes"])
def test_delete_thread_waits_for_inflight_writes(
    saver_name: str, operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _saver(saver_name) as saver:
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"ch": ["value"]}
        checkpoint["channel_versions"] = {"ch": "1"}
        for namespace in ("", "inner"):
            stored = saver.put(
                {
                    "configurable": {
                        "thread_id": "thread-delete",
                        "checkpoint_ns": namespace,
                    }
                },
                checkpoint,
                {},
                {"ch": "1"},
            )
            saver.put_writes(stored, [("ch", "initial")], "initial")

        with saver._cursor() as cur:
            conninfo = DEFAULT_POSTGRES_URI + cur.connection.info.dbname

        checked, resume = Event(), Event()
        writer_pid = []
        original_cursor = saver._cursor

        class PausedCursor:
            def __init__(self, cur):
                self.cur = cur

            def __getattr__(self, name):
                return getattr(self.cur, name)

            def fetchone(self):
                row = self.cur.fetchone()
                # Pause after the tombstone lookup, before any data is written.
                if not checked.is_set():
                    writer_pid.append(self.cur.connection.info.backend_pid)
                    checked.set()
                    assert resume.wait(10)
                return row

        @contextmanager
        def paused_cursor(*, pipeline=False):
            with original_cursor(pipeline=pipeline) as cur:
                yield PausedCursor(cur)

        monkeypatch.setattr(saver, "_cursor", paused_cursor)
        with (
            Connection.connect(
                conninfo, autocommit=True, row_factory=dict_row
            ) as delete_conn,
            Connection.connect(
                conninfo, autocommit=True, row_factory=dict_row
            ) as observer,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            delete_conn.execute("SET statement_timeout = '10s'")
            observer.execute("SET statement_timeout = '10s'")
            deleter = PostgresSaver(delete_conn)
            writer = executor.submit(
                saver.put if operation == "put" else saver.put_writes,
                *(
                    (stored, checkpoint, {}, {"ch": "1"})
                    if operation == "put"
                    else (stored, [("ch", "inflight")], "inflight")
                ),
            )
            try:
                assert checked.wait(10)
                deletion = executor.submit(deleter.delete_thread, "thread-delete")
                deadline = monotonic() + 10
                while not observer.execute(
                    "SELECT %s = ANY(pg_blocking_pids(%s)) AS blocked",
                    (writer_pid[0], delete_conn.info.backend_pid),
                ).fetchone()["blocked"]:
                    assert not deletion.done(), (
                        "Deletion did not wait for the in-flight write"
                    )
                    assert monotonic() < deadline, (
                        "Deletion did not acquire a database lock"
                    )
                    sleep(0.01)

                # A different thread must remain writable while deletion is waiting.
                other_saver = PostgresSaver(observer)
                other = other_saver.put(
                    {
                        "configurable": {
                            "thread_id": "thread-other",
                            "checkpoint_ns": "",
                        }
                    },
                    checkpoint,
                    {},
                    {"ch": "1"},
                )
                other_saver.put_writes(other, [("ch", "keep")], "keep")
            finally:
                resume.set()
            writer.result(timeout=10)
            deletion.result(timeout=10)
            monkeypatch.setattr(saver, "_cursor", original_cursor)

            # Tombstones must also suppress later writes from a different saver.
            deleter.put(stored, checkpoint, {}, {"ch": "1"})
            deleter.put_writes(stored, [("ch", "late")], "late")
            assert saver.get_tuple(stored) is None
            assert (
                list(saver.list({"configurable": {"thread_id": "thread-delete"}})) == []
            )
            assert other_saver.get_tuple(other).pending_writes == [
                ("keep", "ch", "keep")
            ]
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                assert (
                    observer.execute(
                        f"SELECT COUNT(*) AS count FROM {table} WHERE thread_id = %s",
                        ("thread-delete",),
                    ).fetchone()["count"]
                    == 0
                )
                assert (
                    observer.execute(
                        f"SELECT COUNT(*) AS count FROM {table} WHERE thread_id = %s",
                        ("thread-other",),
                    ).fetchone()["count"]
                    == 1
                )


@pytest.mark.parametrize(
    ("saver_name", "thread_row_exists"),
    [("base", True), ("base", False), ("pipe", True), ("fallback", True)],
)
@pytest.mark.parametrize(
    "isolation_level", [IsolationLevel.REPEATABLE_READ, IsolationLevel.SERIALIZABLE]
)
@pytest.mark.parametrize("operation", ["put", "put_writes", "delete_thread"])
def test_delete_thread_rejects_stale_snapshot(
    saver_name: str,
    thread_row_exists: bool,
    isolation_level: IsolationLevel,
    operation: str,
) -> None:
    with _saver(saver_name) as saver:
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"ch": ["initial"]}
        checkpoint["channel_versions"] = {"ch": "1"}
        stored = saver.put(
            {"configurable": {"thread_id": "thread-delete", "checkpoint_ns": ""}},
            checkpoint,
            {},
            {"ch": "1"},
        )
        saver.put_writes(stored, [("ch", "initial")], "initial")
        conn = saver.conn
        with Connection.connect(
            DEFAULT_POSTGRES_URI + conn.info.dbname,
            autocommit=True,
            row_factory=dict_row,
        ) as peer_conn:
            peer = PostgresSaver(peer_conn)
            if not thread_row_exists:
                # Existing data may predate the per-thread coordination table.
                peer_conn.execute(
                    "DELETE FROM checkpoint_threads WHERE thread_id = %s",
                    ("thread-delete",),
                )

            # New keys avoid unrelated conflicts on existing checkpoint data.
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {"ch": ["new"]}
            checkpoint["channel_versions"] = {"ch": "2"}

            def mutate():
                if operation == "put":
                    saver.put(stored, checkpoint, {}, {"ch": "2"})
                elif operation == "put_writes":
                    saver.put_writes(stored, [("ch", "new")], "new")
                else:
                    saver.delete_thread("thread-delete")

            conn.isolation_level = isolation_level
            with pytest.raises(SerializationFailure):
                with conn.transaction():
                    conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()
                    if operation == "delete_thread":
                        new = peer.put(stored, checkpoint, {}, {"ch": "2"})
                        peer.put_writes(new, [("ch", "new")], "new")
                    else:
                        peer.delete_thread("thread-delete")
                    mutate()

            # The caller retries the whole transaction with a fresh snapshot.
            with conn.transaction():
                mutate()

            assert saver.get_tuple(stored) is None
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                assert (
                    peer_conn.execute(
                        f"SELECT COUNT(*) AS count FROM {table} WHERE thread_id = %s",
                        ("thread-delete",),
                    ).fetchone()["count"]
                    == 0
                )


def test_thread_coordination_migration_preserves_data() -> None:
    with _saver("base") as saver:
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"ch": ["keep"]}
        checkpoint["channel_versions"] = {"ch": "1"}
        live = saver.put(
            {"configurable": {"thread_id": "thread-live", "checkpoint_ns": ""}},
            checkpoint,
            {},
            {"ch": "1"},
        )
        saver.put_writes(live, [("ch", "keep")], "keep")
        deleted = saver.put(
            {"configurable": {"thread_id": "thread-deleted", "checkpoint_ns": ""}},
            checkpoint,
            {},
            {"ch": "1"},
        )
        saver.delete_thread("thread-deleted")
        before = saver.get_tuple(live)

        # Recreate the prior schema without losing checkpoints or tombstones.
        saver.conn.execute("DROP TABLE checkpoint_threads")
        saver.conn.execute(
            "DELETE FROM checkpoint_migrations WHERE v = %s",
            (len(saver.MIGRATIONS) - 1,),
        )
        saver.setup()
        saver.setup()

        assert saver.get_tuple(live) == before
        saver.put(deleted, empty_checkpoint(), {}, {})
        saver.put_writes(deleted, [("ch", "late")], "late")
        assert saver.get_tuple(deleted) is None
        assert saver.conn.execute(
            "SELECT thread_id FROM checkpoint_deleted_threads"
        ).fetchall() == [{"thread_id": "thread-deleted"}]
        assert saver.conn.execute(
            "SELECT COUNT(*) AS count FROM checkpoint_migrations"
        ).fetchone()["count"] == len(saver.MIGRATIONS)


@pytest.mark.parametrize("saver_name", ["base", "pool", "pipe", "shallow"])
def test_null_chars(saver_name: str, test_data) -> None:
    with _saver(saver_name) as saver:
        config = saver.put(
            test_data["configs"][0],
            test_data["checkpoints"][0],
            {"my_key": "\x00abc"},
            {},
        )
        assert saver.get_tuple(config).metadata["my_key"] == "abc"  # type: ignore
        assert (
            list(saver.list(None, filter={"my_key": "abc"}))[0].metadata["my_key"]
            == "abc"
        )


def test_nonnull_migrations() -> None:
    _leading_comment_remover = re.compile(r"^/\*.*?\*/")
    for migration in PostgresSaver.MIGRATIONS:
        statement = _leading_comment_remover.sub("", migration).split()[0]
        assert statement.strip()


@pytest.mark.parametrize("saver_name", ["base", "pool", "pipe"])
def test_pending_sends_migration(saver_name: str) -> None:
    with _saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }

        # create the first checkpoint
        # and put some pending sends
        checkpoint_0 = empty_checkpoint()
        config = saver.put(config, checkpoint_0, {}, {})
        saver.put_writes(
            config, [(TASKS, "send-1"), (TASKS, "send-2")], task_id="task-1"
        )
        saver.put_writes(config, [(TASKS, "send-3")], task_id="task-2")

        # check that fetching checkpoint_0 doesn't attach pending sends
        # (they should be attached to the next checkpoint)
        tuple_0 = saver.get_tuple(config)
        assert tuple_0.checkpoint["channel_values"] == {}
        assert tuple_0.checkpoint["channel_versions"] == {}

        # create the second checkpoint
        checkpoint_1 = create_checkpoint(checkpoint_0, {}, 1)
        config = saver.put(config, checkpoint_1, {}, {})

        # check that pending sends are attached to checkpoint_1
        checkpoint_1 = saver.get_tuple(config)
        assert checkpoint_1.checkpoint["channel_values"] == {
            TASKS: ["send-1", "send-2", "send-3"]
        }
        assert TASKS in checkpoint_1.checkpoint["channel_versions"]

        # check that list also applies the migration
        search_results = [
            c for c in saver.list({"configurable": {"thread_id": "thread-1"}})
        ]
        assert len(search_results) == 2
        assert search_results[-1].checkpoint["channel_values"] == {}
        assert search_results[-1].checkpoint["channel_versions"] == {}
        assert search_results[0].checkpoint["channel_values"] == {
            TASKS: ["send-1", "send-2", "send-3"]
        }
        assert TASKS in search_results[0].checkpoint["channel_versions"]


@pytest.mark.parametrize("saver_name", ["base", "pool", "pipe"])
def test_get_checkpoint_no_channel_values(
    monkeypatch, saver_name: str, test_data
) -> None:
    """Backwards compatibility test that verifies a checkpoint with no channel_values key can be retrieved without throwing an error."""
    with _saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_ns": "",
                "__super_private_key": "super_private_value",
            },
        }
        chkpnt: Checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
        saver.put(config, chkpnt, {}, {})

        load_checkpoint_tuple = saver._load_checkpoint_tuple

        def patched_load_checkpoint_tuple(value):
            value["checkpoint"].pop("channel_values", None)
            return load_checkpoint_tuple(value)

        monkeypatch.setattr(
            saver, "_load_checkpoint_tuple", patched_load_checkpoint_tuple
        )

        checkpoint = saver.get_tuple(config)
        assert checkpoint.checkpoint["channel_values"] == {}
