import pytest

torch = pytest.importorskip(modname="torch", reason="torch is not installed")

from medkit.training.utils import BatchData

TEST_CUDA = torch.cuda.is_available()


def test_get_attributes():
    data = BatchData(inputs=[0, 1, 2])
    assert data["inputs"] == [0, 1, 2]
    with pytest.raises(KeyError):
        _ = data["outputs"]
    assert data[0] == {"inputs": 0}


def test_index_int():
    data = BatchData(inputs=["hello", "world"], outputs=["bonjour", "monde"])
    assert data[0] == {"inputs": "hello", "outputs": "bonjour"}
    assert data[1] == {"inputs": "world", "outputs": "monde"}
    assert data[0:1] == {"inputs": ["hello"], "outputs": ["bonjour"]}
    assert data[:] == {"inputs": ["hello", "world"], "outputs": ["bonjour", "monde"]}


def test_dict_properties():
    data = BatchData(inputs=["hello", "world"], outputs=["bonjour", "monde"])
    assert list(data) == ["inputs", "outputs"]
    assert list(data.values()) == [["hello", "world"], ["bonjour", "monde"]]

    data = BatchData({"inputs": ["hello", "world"], "outputs": ["bonjour", "monde"]})
    assert list(data) == ["inputs", "outputs"]
    assert list(data.values()) == [["hello", "world"], ["bonjour", "monde"]]


@pytest.mark.skipif(not TEST_CUDA, reason="cuda is not available")
def test_to_device():
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")
    data = BatchData(inputs=["hello", "world"], outputs=torch.tensor([0, 1]))
    new_data = data.to_device(gpu)
    for tensor_cpu, tensor_gpu in zip(data["outputs"], new_data["outputs"]):
        assert tensor_cpu.device == cpu
        assert tensor_gpu.device == gpu
        assert tensor_cpu.item() == tensor_gpu.item()


def test_to_cpu():
    cpu = torch.device("cpu")
    data = BatchData(inputs=["hello", "world"], outputs=[torch.tensor(0), torch.tensor(1)])
    new_data = data.to_device(cpu)
    for old_tensor, new_tensor in zip(data["outputs"], new_data["outputs"]):
        assert old_tensor.device == cpu
        assert new_tensor.device == cpu
        assert old_tensor.item() == new_tensor.item()
