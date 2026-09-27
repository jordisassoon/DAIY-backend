from src.tools.fetch_size import find_size


def test_shape():
    assert find_size("./tests/out/test_video.npy")[1] == 768
