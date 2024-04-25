import pytest

from medkit.core.store import GlobalStore


@pytest.fixture(autouse=True)
def _clear_store():
    yield
    GlobalStore.del_store()
