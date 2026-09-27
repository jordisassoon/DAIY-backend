from src.tools.select_files import select_files


def test_empty_data_dir(tmp_path):
    data_path = tmp_path / "empty_data"
    data_path.mkdir()
    save_path = tmp_path / "empty_save"
    save_path.mkdir()
    result = select_files(
        data_dir=data_path,
        save_dir=save_path,
        source_format=".mp4",
        target_format=".npy",
    )
    assert len(result) == 0


def test_empty_save_dir(tmp_path):
    data_path = "./tests/data/"
    save_path = tmp_path / "empty_save"
    save_path.mkdir()
    result = select_files(
        data_dir=data_path,
        save_dir=save_path,
        source_format=".mp4",
        target_format=".npy",
    )
    assert len(result) == 1


def test_full_save_dir():
    data_path = "./tests/data/"
    save_path = "./tests/out/"
    result = select_files(
        data_dir=data_path,
        save_dir=save_path,
        source_format=".mp4",
        target_format=".npy",
    )
    assert len(result) == 0


def test_subset_json(tmp_path):
    data_path = "./tests/data/"
    save_path = tmp_path / "empty_save"
    subset_json = "./tests/data/subset.json"
    result = select_files(
        data_dir=data_path,
        save_dir=save_path,
        source_format=".mp4",
        target_format=".npy",
        subset_path=subset_json,
    )
    assert len(result) == 1
