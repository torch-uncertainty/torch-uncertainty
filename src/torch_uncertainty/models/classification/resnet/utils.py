from enum import Enum

_RESNET_BLOCKS: dict[int, list[int]] = {
    18: [2, 2, 2, 2],
    20: [3, 3, 3],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    44: [7, 7, 7],
    56: [9, 9, 9],
    101: [3, 4, 23, 3],
    110: [18, 18, 18],
    152: [3, 8, 36, 3],
    1202: [200, 200, 200],
}


class ResNetStyle(str, Enum):
    IMAGENET = "imagenet"
    CIFAR = "cifar"


def get_resnet_num_blocks(arch: int) -> list[int]:
    """Return the number of blocks per stage for a given ResNet architecture.

    Args:
        arch: ResNet architecture identifier (e.g., 18, 34, 50, 101).

    Returns:
        A list containing the number of blocks in each stage of the network.

    Raises:
        ValueError: If the provided architecture is not supported.
    """
    try:
        return _RESNET_BLOCKS[arch]
    except KeyError:
        # Suppress exception chaining (avoid exposing the underlying KeyError)
        raise ValueError(f"Unknown ResNet architecture. Got {arch}.") from None
