import pytest

from medkit.core.store import GlobalStore, _DictStore


class SubStore(_DictStore):
    name = "substore"


def test_global_store_init_store():
    store = SubStore()
    GlobalStore.init_store(store)
    assert isinstance(store, SubStore)
    assert store == GlobalStore.get_store()


def test_global_store_init_store_error():
    store = GlobalStore.get_store()
    assert isinstance(store, _DictStore)
    with pytest.raises(RuntimeError):
        GlobalStore.init_store(SubStore())
