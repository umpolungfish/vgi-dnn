"""Keep model tensors and returned optimizer tensors in one safetensors carrier."""
from safetensors.torch import load_file

PREFIX='__optimizer.'

def split_state(state):
    return ({k:v for k,v in state.items() if not k.startswith(PREFIX)},
            {k[len(PREFIX):]:v for k,v in state.items() if k.startswith(PREFIX)})

def load_model(path):
    return split_state(load_file(str(path)))[0]
