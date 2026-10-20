#!/usr/bin/env python3
"""Compare all typed candidate scores and gradients against compiled execution."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from optimizer_state import load_model
from typed_tuple_learning import choices
from bootstrap import AXES


def verify(root, runtime, returned, out):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    metadata=json.loads((runtime/'build.json').read_text())
    model=DialetheicTransformerTorch(**metadata['config']).double()
    model.load_state_dict(load_model(Path(metadata['parent'])/'model.safetensors'))
    native=ctypes.CDLL(str(runtime/'model.so'))
    native.vgi_typed_axis.argtypes=[ctypes.c_long];native.vgi_typed_axis.restype=ctypes.c_long
    native.vgi_typed_score.argtypes=[ctypes.c_long]*2;native.vgi_typed_score.restype=ctypes.c_double
    native.vgi_full_gradient.argtypes=[ctypes.c_long];native.vgi_full_gradient.restype=ctypes.c_double
    native.vgi_full_parameter.argtypes=[ctypes.c_long];native.vgi_full_parameter.restype=ctypes.c_double
    native.vgi_retained_reset()
    prompt=metadata['typed_tuple']['prompt'];address=metadata['typed_tuple']['address']
    results=[];named=dict(model.named_parameters())
    for axis,(_,_,alphabet) in enumerate(AXES):
        model.zero_grad(set_to_none=True)
        scores,_=choices(model,prompt,address[:axis],axis)
        loss=F.cross_entropy(scores[None],torch.tensor([alphabet.index(address[axis])]))
        loss.backward()
        status=native.vgi_typed_axis(axis)
        assert status>=0,(axis,status)
        observed=torch.tensor([native.vgi_typed_score(axis,i) for i in range(len(alphabet))],dtype=torch.double)
        gradient=torch.cat([named[p['key']].grad.flatten() for p in metadata['parameters']])
        actual=torch.tensor([native.vgi_full_gradient(i) for i in range(len(gradient))],dtype=torch.double)
        score_error=float((observed-scores.detach()).abs().max())
        gradient_error=float((actual-gradient).abs().max())
        assert score_error<1e-8 and gradient_error<1e-7,(axis,score_error,gradient_error)
        results.append({'axis':axis,'score_error':score_error,'gradient_error':gradient_error})
        print(json.dumps(results[-1]),flush=True)
    assert native.vgi_typed_update()==0
    reference=load_model(returned/'model.safetensors')
    expected=torch.cat([reference[p['key']].flatten() for p in metadata['parameters']])
    observed=torch.tensor([native.vgi_full_parameter(i) for i in range(len(expected))],dtype=torch.double)
    error=float((expected-observed).abs().max());assert error<1e-8,error
    report={'status':'passed','axes':results,'update_parameter_error':error,
        'module_sha256':hashlib.sha256((runtime/'model.imasm').read_bytes()).hexdigest(),
        'scope':'all twelve teacher-conditioned typed candidate scores and parameter gradients, then shared clipped Adam update; CPU versus compiled native'}
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','returned','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.root.resolve(),a.runtime.resolve(),a.returned.resolve(),a.out.resolve())
