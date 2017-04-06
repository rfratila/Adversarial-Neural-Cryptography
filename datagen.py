import numpy as np
import random


def get_plain_text(N=16,to_generate=256):
    '''
    Generates a plain text message in binary values
    N: How many bits the message is
    to_generate: How many messages to generate
    '''
    block_text = []
    for i in range(0,to_generate):
        text = []
        for j in range(0,N):
            text += [-1] if random.randint(0,1) == 0 else [1]
        block_text += [text]
    return np.array(block_text)

def get_key(N=16,batch=256):
    '''
    Generates a key value for Bob and Alice to use
    N: How many bits the key is
    '''
    key = []
    for j in range(0,N):
        key += [-1] if random.randint(0,1) == 0 else [1]
    key_collection = []
    for i in range(0,batch):
        key_collection += [key]
    return np.array(key_collection)

if __name__ == "__main__":
    pass
