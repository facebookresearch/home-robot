import pytest

try:
    import habitat
except ImportError:
    print("Warning: habitat not installed, skipping habitat tests")
    pytest.skip(allow_module_level=True)


def test_objectnav_env():
    pass
