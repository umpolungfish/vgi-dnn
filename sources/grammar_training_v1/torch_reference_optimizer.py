"""Independent Torch optimizer initialized from the returned shared checkpoint."""
import torch
from pathlib import Path
from safetensors.torch import load_file
from optimizer_state import split_state

def make_optimizer(model,build,state=None):
    kind=build.get('optimizer','sgd')
    optimizer=(torch.optim.Adam if kind=='adam' else torch.optim.SGD)(model.parameters(),lr=build['learning_rate'])
    if state is None:_,state=split_state(load_file(str(Path(build['parent'])/'model.safetensors')))
    if state:
        assert kind=='adam';position=0;parameters=dict(model.named_parameters())
        for entry in build['parameters']:
            p=parameters[entry['key']];end=position+entry['size']
            optimizer.state[p]={'step':torch.tensor(float(state['clock'][0]),dtype=torch.float64),
                                'exp_avg':state['m'][position:end].reshape_as(p).clone(),
                                'exp_avg_sq':state['v'][position:end].reshape_as(p).clone()}
            position=end
    return optimizer
