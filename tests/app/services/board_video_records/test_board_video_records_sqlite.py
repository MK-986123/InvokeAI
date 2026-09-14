import pytest

from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
from invokeai.app.services.board_video_records.board_video_records_sqlite import SqliteBoardVideoRecordStorage
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.video_records.video_records_sqlite import SqliteVideoRecordStorage
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


@pytest.fixture
def db():
    config = InvokeAIAppConfig(use_memory_db=True)
    logger = InvokeAILogger.get_logger(config=config)
    return create_mock_sqlite_database(config, logger)


@pytest.fixture
def video_store(db):
    return SqliteVideoRecordStorage(db=db)


@pytest.fixture
def board_video_store(db):
    return SqliteBoardVideoRecordStorage(db=db)


@pytest.fixture
def board_store(db):
    return SqliteBoardRecordStorage(db=db)


def test_get_boards_for_videos_empty_list(board_video_store):
    assert board_video_store.get_boards_for_videos([]) == {}


def test_get_boards_for_videos(db, video_store, board_video_store, board_store):
    board1 = board_store.save("Board 1", "user1")
    board2 = board_store.save("Board 2", "user1")

    video_store.save(
        video_name="video1.mp4",
        video_origin=ResourceOrigin.INTERNAL,
        video_category=ImageCategory.GENERAL,
        width=64,
        height=64,
        duration=1.0,
        fps=8.0,
        has_workflow=False,
        user_id="user1",
    )
    video_store.save(
        video_name="video2.mp4",
        video_origin=ResourceOrigin.INTERNAL,
        video_category=ImageCategory.GENERAL,
        width=64,
        height=64,
        duration=1.0,
        fps=8.0,
        has_workflow=False,
        user_id="user1",
    )
    video_store.save(
        video_name="video3.mp4",
        video_origin=ResourceOrigin.INTERNAL,
        video_category=ImageCategory.GENERAL,
        width=64,
        height=64,
        duration=1.0,
        fps=8.0,
        has_workflow=False,
        user_id="user1",
    )

    board_video_store.add_video_to_board(board1.board_id, "video1.mp4")
    board_video_store.add_video_to_board(board2.board_id, "video2.mp4")

    boards = board_video_store.get_boards_for_videos(["video1.mp4", "video2.mp4", "video3.mp4", "nonexistent.mp4"])
    assert boards == {
        "video1.mp4": board1.board_id,
        "video2.mp4": board2.board_id,
    }


def test_get_boards_for_videos_chunking(db, video_store, board_video_store, board_store):
    board = board_store.save("Large Board", "user1")
    video_names = [f"video_{i}.mp4" for i in range(1000)]
    for name in video_names:
        video_store.save(
            video_name=name,
            video_origin=ResourceOrigin.INTERNAL,
            video_category=ImageCategory.GENERAL,
            width=64,
            height=64,
            duration=1.0,
            fps=8.0,
            has_workflow=False,
            user_id="user1",
        )
        board_video_store.add_video_to_board(board.board_id, name)

    boards = board_video_store.get_boards_for_videos(video_names)
    assert len(boards) == 1000
    assert all(b == board.board_id for b in boards.values())
