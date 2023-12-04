import uuid

from medkit.core import generate_deterministic_id


def test_deterministic_id():
    uid = "UNIQUE_UUID"
    first_deterministic_uuid = generate_deterministic_id(uid)
    assert isinstance(first_deterministic_uuid, uuid.UUID)
    # simulate another call
    second_deterministic_uuid = generate_deterministic_id(uid)
    assert first_deterministic_uuid == second_deterministic_uuid
