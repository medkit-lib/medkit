__all__ = ["generate_id", "generate_deterministic_id"]

import random
import uuid


def generate_id() -> str:
    return str(uuid.uuid1())


def generate_deterministic_id(reference_id: str) -> uuid.UUID:
    """Generate a deterministic UUID based on reference_id.
    The generated UUID will be the same if the reference_id is the same.

    Parameters
    ----------
    reference_id
        A string representation of an UID

    Returns
    -------
    uuid.UUID
        The UUID object
    """
    rng = random.Random(reference_id)
    return uuid.UUID(int=rng.getrandbits(128))
