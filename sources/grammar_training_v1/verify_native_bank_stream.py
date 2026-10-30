#!/usr/bin/env python3
"""Continue a complete native bank return through the existing CMREENT1 path."""
import argparse
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from safetensors.torch import load
from composed_format import decode,restore_packet
from composed_graph_reference import forward
from export_composed_graph import build
from optimizer_state import split_state
from train_cuda_trajectories import adam_step,sha


def verify(root,native,runtime,dataset,out):
    torch.set_num_threads(1);sys.path.insert(0,str(root/'bob3ect/reentry_dnn'))
    from dtransformer_torch import DialetheicTransformerTorch
    evidence=json.loads((native/'verification.json').read_text())
    metadata=json.loads((runtime/'build.json').read_text())
    returned=(native/'vox_profile.stdout').read_bytes()
    weights,retained,discrete,continuous=decode(returned)
    out.mkdir(parents=True,exist_ok=False);checkpoint=out/'checkpoint';checkpoint.mkdir()
    (checkpoint/'model.safetensors').write_bytes(weights)
    for name in ('config.json','execution_input.json','tokenizer.json'):
        shutil.copyfile(Path(evidence['parent'])/name,checkpoint/name)
    (checkpoint/'retained_state.json').write_text(json.dumps(retained,indent=2)+'\n')
    packet=restore_packet(returned,passes=2,steps=1,local_weight=.25)
    (out/'reentry.packet').write_bytes(packet)
    build(root,checkpoint,Path(metadata['graph_library']),dataset/'vocabulary.json',out/'runtime')
    run=subprocess.run([str(out/'runtime/model')],input=packet,capture_output=True,check=True)
    (out/'returned.stdout').write_bytes(run.stdout)
    model_bytes,new_retained,new_discrete,new_state=decode(run.stdout)
    initial,opt=split_state(load(weights));model=DialetheicTransformerTorch(**metadata['config']).double();model.load_state_dict(initial)
    lib=ctypes.CDLL(str(out/'runtime/model.so'));oracle=lib.ig_flow_gate
    oracle.argtypes=[ctypes.c_uint]*5;oracle.restype=ctypes.c_uint
    vocabulary=json.loads((dataset/'vocabulary.json').read_text());row=evidence['graph']
    result=forward(model,row,vocabulary,oracle,2,.25,initial=continuous)
    result['loss'].backward();adam_step(model,opt,metadata['parameters'],metadata['learning_rate'],metadata['stream_gradient_clip'])
    with torch.no_grad():after=forward(model,row,vocabulary,oracle,2,.25,initial=continuous)
    expected={**model.state_dict(),**{'__optimizer.'+k:v for k,v in opt.items()}};actual=load(model_bytes)
    assert actual.keys()==expected.keys()
    model_error=max(float((value.double()-actual[key].double()).abs().max()) for key,value in expected.items())
    memory_error=max(abs(value.item()-new_state[key][e][c]) for e,pair in enumerate(after['memory'])
        for channel,key in enumerate(('memory_t','memory_f')) for c,value in enumerate(pair[channel].flatten()))
    assert max(model_error,memory_error)<1e-8
    assert new_state['source_edges']==row['edge_values'] and new_state['source_return']==row['return_bits']
    corrupt=bytearray(packet);corrupt[72]^=1  # graph seed in GGSTATE1 header
    rejected=subprocess.run([str(out/'runtime/model')],input=bytes(corrupt),capture_output=True)
    assert rejected.returncode!=0
    report={'passed':True,'model_and_optimizer_error':model_error,'memory_error':memory_error,
        'source_binding_control_exit':rejected.returncode,'initial_return_sha256':sha(native/'vox_profile.stdout'),
        'packet_sha256':sha(out/'reentry.packet'),'returned_sha256':sha(out/'returned.stdout'),
        'scope':'fresh native executable bakes returned model and Adam, accepts complete CMREENT1 carrier through existing checked restore path, executes one graph step and closes source and memory; corrupted graph seed rejected'}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','native','runtime','dataset','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();verify(a.root.resolve(),a.native.resolve(),a.runtime.resolve(),a.dataset.resolve(),a.out.resolve())
