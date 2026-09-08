# type: ignore

import asyncio
from contextlib import asynccontextmanager
from time import monotonic
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
from psycopg import AsyncConnection, IsolationLevel
from psycopg.errors import SerializationFailure
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver,
    AsyncShallowPostgresSaver,
)
from tests.conftest import DEFAULT_POSTGRES_URI


def _exclude_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in EXCLUDED_METADATA_KEYS}


@asynccontextmanager
async def _pool_saver():
    """Fixture for pool mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        async with AsyncConnectionPool(
            DEFAULT_POSTGRES_URI + database,
            max_size=10,
            kwargs={"autocommit": True, "row_factory": dict_row},
        ) as pool:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _pipe_saver():
    """Fixture for pipeline mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
    try:
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI + database,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            checkpointer = AsyncPostgresSaver(conn)
            await checkpointer.setup()
            async with conn.pipeline() as pipe:
                checkpointer = AsyncPostgresSaver(conn, pipe=pipe)
                yield checkpointer
    finally:
        # drop unique db
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _base_saver():
    """Fixture for regular connection mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
    try:
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI + database,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            checkpointer = AsyncPostgresSaver(conn)
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _shallow_saver():
    """Fixture for shallow connection mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
    try:
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI + database,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            checkpointer = AsyncShallowPostgresSaver(conn)
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _saver(name: str):
    if name in ("base", "fallback"):
        async with _base_saver() as saver:
            if name == "fallback":
                saver.supports_pipeline = False
            yield saver
    elif name == "shallow":
        async with _shallow_saver() as saver:
            yield saver
    elif name == "pool":
        async with _pool_saver() as saver:
            yield saver
    elif name == "pipe":
        async with _pipe_saver() as saver:
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
async def test_combined_metadata(saver_name: str, test_data) -> None:
    async with _saver(saver_name) as saver:
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
        await saver.aput(config, chkpnt, metadata, {})
        checkpoint = await saver.aget_tuple(config)
        assert checkpoint.metadata == {
            **metadata,
            "run_id": "my_run_id",
        }


@pytest.mark.parametrize("saver_name", ["base", "pool", "pipe", "shallow"])
async def test_asearch(saver_name: str, test_data) -> None:
    async with _saver(saver_name) as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]

        await saver.aput(configs[0], checkpoints[0], metadata[0], {})
        await saver.aput(configs[1], checkpoints[1], metadata[1], {})
        await saver.aput(configs[2], checkpoints[2], metadata[2], {})

        # call method / assertions
        query_1 = {"source": "input"}  # search by 1 key
        query_2 = {
            "step": 1,
        }  # search by multiple keys
        query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
        query_4 = {"source": "update", "step": 1}  # no match

        search_results_1 = [c async for c in saver.alist(None, filter=query_1)]
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == {
            **_exclude_keys(configs[0]["configurable"]),
            **metadata[0],
        }

        search_results_2 = [c async for c in saver.alist(None, filter=query_2)]
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == {
            **_exclude_keys(configs[1]["configurable"]),
            **metadata[1],
        }

        search_results_3 = [c async for c in saver.alist(None, filter=query_3)]
        assert len(search_results_3) == 3

        search_results_4 = [c async for c in saver.alist(None, filter=query_4)]
        assert len(search_results_4) == 0

        # search by config (defaults to checkpoints across all namespaces)
        search_results_5 = [
            c async for c in saver.alist({"configurable": {"thread_id": "thread-2"}})
        ]
        assert len(search_results_5) == 2
        assert {
            search_results_5[0].config["configurable"]["checkpoint_ns"],
            search_results_5[1].config["configurable"]["checkpoint_ns"],
        } == {"", "inner"}


@pytest.mark.parametrize("saver_name", ["base", "pool", "pipe", "fallback"])
async def test_delete_thread_ignores_late_writes(saver_name: str) -> None:
    async with _saver(saver_name) as saver:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-delete",
                "checkpoint_ns": "",
            }
        }

        stored = await saver.aput(config, empty_checkpoint(), {}, {})
        await saver.adelete_thread("thread-delete")

        await saver.aput(stored, empty_checkpoint(), {"step": 99}, {})
        await saver.aput_writes(stored, [("ch", "late-write")], str(uuid4()))

        assert (
            await saver.aget_tuple({"configurable": {"thread_id": "thread-delete"}})
            is None
        )
        assert [
            c
            async for c in saver.alist({"configurable": {"thread_id": "thread-delete"}})
        ] == []


@pytest.mark.parametrize("saver_name", ["base", "pool", "pipe", "fallback"])
@pytest.mark.parametrize("operation", ["put", "put_writes"])
async def test_delete_thread_waits_for_inflight_writes(
    saver_name: str, operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _saver(saver_name) as saver:
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"ch": ["value"]}
        checkpoint["channel_versions"] = {"ch": "1"}
        for namespace in ("", "inner"):
            stored = await saver.aput(
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
            await saver.aput_writes(stored, [("ch", "initial")], "initial")

        async with saver._cursor() as cur:
            conninfo = DEFAULT_POSTGRES_URI + cur.connection.info.dbname

        checked, resume = asyncio.Event(), asyncio.Event()
        writer_pid = []
        original_cursor = saver._cursor

        class PausedCursor:
            def __init__(self, cur):
                self.cur = cur

            def __getattr__(self, name):
                return getattr(self.cur, name)

            async def fetchone(self):
                row = await self.cur.fetchone()
                # Pause after the tombstone lookup, before any data is written.
                if not checked.is_set():
                    writer_pid.append(self.cur.connection.info.backend_pid)
                    checked.set()
                    await asyncio.wait_for(resume.wait(), 10)
                return row

        @asynccontextmanager
        async def paused_cursor(*, pipeline=False):
            async with original_cursor(pipeline=pipeline) as cur:
                yield PausedCursor(cur)

        monkeypatch.setattr(saver, "_cursor", paused_cursor)
        async with (
            await AsyncConnection.connect(
                conninfo, autocommit=True, row_factory=dict_row
            ) as delete_conn,
            await AsyncConnection.connect(
                conninfo, autocommit=True, row_factory=dict_row
            ) as observer,
        ):
            await delete_conn.execute("SET statement_timeout = '10s'")
            await observer.execute("SET statement_timeout = '10s'")
            deleter = AsyncPostgresSaver(delete_conn)
            writer = asyncio.create_task(
                saver.aput(stored, checkpoint, {}, {"ch": "1"})
                if operation == "put"
                else saver.aput_writes(stored, [("ch", "inflight")], "inflight")
            )
            tasks = [writer]
            try:
                await asyncio.wait_for(checked.wait(), 10)
                deletion = asyncio.create_task(deleter.adelete_thread("thread-delete"))
                tasks.append(deletion)
                deadline = monotonic() + 10
                while not (
                    await (
                        await observer.execute(
                            "SELECT %s = ANY(pg_blocking_pids(%s)) AS blocked",
                            (writer_pid[0], delete_conn.info.backend_pid),
                        )
                    ).fetchone()
                )["blocked"]:
                    assert not deletion.done(), (
                        "Deletion did not wait for the in-flight write"
                    )
                    assert monotonic() < deadline, (
                        "Deletion did not acquire a database lock"
                    )
                    await asyncio.sleep(0.01)

                # A different thread must remain writable while deletion is waiting.
                other_saver = AsyncPostgresSaver(observer)
                other = await other_saver.aput(
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
                await other_saver.aput_writes(other, [("ch", "keep")], "keep")
            finally:
                resume.set()
                await asyncio.wait_for(asyncio.gather(*tasks), 10)
            monkeypatch.setattr(saver, "_cursor", original_cursor)

            # Tombstones must also suppress later writes from a different saver.
            await deleter.aput(stored, checkpoint, {}, {"ch": "1"})
            await deleter.aput_writes(stored, [("ch", "late")], "late")
            assert await saver.aget_tuple(stored) is None
            assert [
                item
                async for item in saver.alist(
                    {"configurable": {"thread_id": "thread-delete"}}
                )
            ] == []
            assert (await other_saver.aget_tuple(other)).pending_writes == [
                ("keep", "ch", "keep")
            ]
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                assert (
                    await (
                        await observer.execute(
                            f"SELECT COUNT(*) AS count FROM {table} WHERE thread_id = %s",
                            ("thread-delete",),
                        )
                    ).fetchone()
                )["count"] == 0
                assert (
                    await (
                        await observer.execute(
                            f"SELECT COUNT(*) AS count FROM {table} WHERE thread_id = %s",
                            ("thread-other",),
                        )
                    ).fetchone()
                )["count"] == 1


@pytest.mark.parametrize(
    ("saver_name", "thread_row_exists"),
    [("base", True), ("base", False), ("pipe", True), ("fallback", True)],
)
@pytest.mark.parametrize(
    "isolation_level", [IsolationLevel.REPEATABLE_READ, IsolationLevel.SERIALIZABLE]
)
@pytest.mark.parametrize("operation", ["put", "put_writes", "delete_thread"])
async def test_delete_thread_rejects_stale_snapshot(
    saver_name: str,
    thread_row_exists: bool,
    isolation_level: IsolationLevel,
    operation: str,
) -> None:
    async with _saver(saver_name) as saver:
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"ch": ["initial"]}
        checkpoint["channel_versions"] = {"ch": "1"}
        stored = await saver.aput(
            {"configurable": {"thread_id": "thread-delete", "checkpoint_ns": ""}},
            checkpoint,
            {},
            {"ch": "1"},
        )
        await saver.aput_writes(stored, [("ch", "initial")], "initial")
        conn = saver.conn
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI + conn.info.dbname,
            autocommit=True,
            row_factory=dict_row,
        ) as peer_conn:
            peer = AsyncPostgresSaver(peer_conn)
            if not thread_row_exists:
                # Existing data may predate the per-thread coordination table.
                await peer_conn.execute(
                    "DELETE FROM checkpoint_threads WHERE thread_id = %s",
                    ("thread-delete",),
                )

            # New keys avoid unrelated conflicts on existing checkpoint data.
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {"ch": ["new"]}
            checkpoint["channel_versions"] = {"ch": "2"}

            async def mutate():
                if operation == "put":
                    await saver.aput(stored, checkpoint, {}, {"ch": "2"})
                elif operation == "put_writes":
                    await saver.aput_writes(stored, [("ch", "new")], "new")
                else:
                    await saver.adelete_thread("thread-delete")

            await conn.set_isolation_level(isolation_level)
            with pytest.raises(SerializationFailure):
                async with conn.transaction():
                    await (
                        await conn.execute("SELECT COUNT(*) FROM checkpoints")
                    ).fetchone()
                    if operation == "delete_thread":
                        new = await peer.aput(stored, checkpoint, {}, {"ch": "2"})
                        await peer.aput_writes(new, [("ch", "new")], "new")
                    else:
                        await peer.adelete_thread("thread-delete")
                    await mutate()

            # The caller retries the whole transaction with a fresh snapshot.
            async with conn.transaction():
                await mutate()

            assert await saver.aget_tuple(stored) is None
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                assert (
                    await (
                        await peer_conn.execute(
                            f"SELECT COUNT(*) AS count FROM {table} WHERE thread_id = %s",
                            ("thread-delete",),
                        )
                    ).fetchone()
                )["count"] == 0


@pytest.mark.parametrize("saver_name", ["base", "pool", "pipe", "shallow"])
async def test_null_chars(saver_name: str, test_data) -> None:
    async with _saver(saver_name) as saver:
        config = await saver.aput(
            test_data["configs"][0],
            test_data["checkpoints"][0],
            {"my_key": "\x00abc"},
            {},
        )
        assert (await saver.aget_tuple(config)).metadata["my_key"] == "abc"  # type: ignore
        assert [c async for c in saver.alist(None, filter={"my_key": "abc"})][
            0
        ].metadata["my_key"] == "abc"


@pytest.mark.parametrize("saver_name", ["base", "pool", "pipe"])
async def test_pending_sends_migration(saver_name: str) -> None:
    async with _saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }

        # create the first checkpoint
        # and put some pending sends
        checkpoint_0 = empty_checkpoint()
        config = await saver.aput(config, checkpoint_0, {}, {})
        await saver.aput_writes(
            config, [(TASKS, "send-1"), (TASKS, "send-2")], task_id="task-1"
        )
        await saver.aput_writes(config, [(TASKS, "send-3")], task_id="task-2")

        # check that fetching checkpoint_0 doesn't attach pending sends
        # (they should be attached to the next checkpoint)
        tuple_0 = await saver.aget_tuple(config)
        assert tuple_0.checkpoint["channel_values"] == {}
        assert tuple_0.checkpoint["channel_versions"] == {}

        # create the second checkpoint
        checkpoint_1 = create_checkpoint(checkpoint_0, {}, 1)
        config = await saver.aput(config, checkpoint_1, {}, {})

        # check that pending sends are attached to checkpoint_1
        tuple_1 = await saver.aget_tuple(config)
        assert tuple_1.checkpoint["channel_values"] == {
            TASKS: ["send-1", "send-2", "send-3"]
        }
        assert TASKS in tuple_1.checkpoint["channel_versions"]

        # check that list also applies the migration
        search_results = [
            c async for c in saver.alist({"configurable": {"thread_id": "thread-1"}})
        ]
        assert len(search_results) == 2
        assert search_results[-1].checkpoint["channel_values"] == {}
        assert search_results[-1].checkpoint["channel_versions"] == {}
        assert search_results[0].checkpoint["channel_values"] == {
            TASKS: ["send-1", "send-2", "send-3"]
        }
        assert TASKS in search_results[0].checkpoint["channel_versions"]


@pytest.mark.parametrize("saver_name", ["base", "pool", "pipe"])
async def test_get_checkpoint_no_channel_values(
    monkeypatch, saver_name: str, test_data
) -> None:
    """Backwards compatibility test that verifies a checkpoint with no channel_values key can be retrieved without throwing an error."""
    async with _saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_ns": "",
                "__super_private_key": "super_private_value",
            },
            "metadata": {"run_id": "my_run_id"},
        }
        chkpnt: Checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
        await saver.aput(config, chkpnt, {}, {})

        load_checkpoint_tuple = saver._load_checkpoint_tuple

        def patched_load_checkpoint_tuple(value):
            value["checkpoint"].pop("channel_values", None)
            return load_checkpoint_tuple(value)

        monkeypatch.setattr(
            saver, "_load_checkpoint_tuple", patched_load_checkpoint_tuple
        )

        checkpoint = await saver.aget_tuple(config)
        assert checkpoint.checkpoint["channel_values"] == {}
