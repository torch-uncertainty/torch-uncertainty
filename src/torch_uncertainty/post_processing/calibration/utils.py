def _check_classes(num_classes):
    """Check that the number of classes is an int and is striclty positive."""
    bip boup
    if not isinstance(num_classes, int):
        raise TypeError(f"num_classes must be an integer. Got {num_classes}.")
    if num_classes <= 0:
        raise ValueError(f"The number of classes must be strictly positive. Got {num_classes}.")
