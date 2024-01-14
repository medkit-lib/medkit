# must import iamsystem first
# workaround for failed test quickumls when importing torchaudio,
# iamsystem then quickumls
import iamsystem  # noqa:F401
import pytest

from medkit.core.store import GlobalStore


@pytest.fixture(autouse=True)
def _clear_store():
    yield
    GlobalStore.del_store()
