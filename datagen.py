import numpy as np


def get_random_block(N=16, batch=256):
    """Generates a random batch of blocks in binary {-1, 1}.

    Args:
        N: How many bits each block is.
        batch: How many blocks to generate.

    Returns:
        Batch of random blocks.
    """
    return 2 * np.random.randint(2, size=(batch, N)) - 1
