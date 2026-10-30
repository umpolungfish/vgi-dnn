#!/usr/bin/env python3
"""Lift graph-wide learning with continuous edge memory in the shared transformer."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from export_full_learning import export, compile_model


def build(root, checkpoint, library, vocabulary, out, passes=2, steps=1, local_weight=.25, detach=False, capacity=8000000):
    if not 1 <= passes <= 16 or not 0 <= steps <= 100 or not 0 <= local_weight <= 100 or capacity < 100000:
        raise ValueError('invalid composed graph bounds')
    export(root, checkpoint, out, retained=True, context_length=9, optimization='O2',
           graph_library=library, graph_vocabulary=vocabulary, optimizer='adam')
    cpp = out / 'full_model.cpp'; source = cpp.read_text()
    source = re.sub(r'#define VGI_TAPE_CAPACITY \d+', '#define VGI_TAPE_CAPACITY ' + str(capacity), source, count=1)
    source = source[:source.rindex('extern "C" __attribute__((force_align_arg_pointer)) void _start()')]
    source += '\n' + Path(__file__).with_name('composed_graph.hpp').read_text()
    declaration='extern "C" unsigned int ig_flow_gate(unsigned int,unsigned int,unsigned int,unsigned int,unsigned int);'
    source=source.replace('ig_flow_gate(', 'composed_bound_gate(')
    instrumented=declaration.replace('ig_flow_gate(', 'composed_bound_gate(')
    if source.count(instrumented)!=1:raise ValueError('source graph oracle declaration changed')
    source=source.replace(instrumented,declaration+'''
static long composed_oracle_calls,composed_oracle_invocations;
static unsigned int composed_bound_gate(unsigned int a,unsigned int b,unsigned int c,unsigned int d,unsigned int e){
    composed_oracle_calls++;composed_oracle_invocations++;return ig_flow_gate(a,b,c,d,e);}
''',1)
    fields = ['graph_nodes', 'graph_edges', 'graph_readout', 'graph_seed', 'graph_mode', 'graph_op', 'graph_in', 'graph_outdegree', 'graph_from', 'graph_to', 'graph_slot', 'graph_input_slot', 'graph_value', 'graph_before', 'graph_after', 'graph_expected', 'graph_diagnostics', 'graph_started', 'graph_total_passes', 'composed_memory', 'composed_probability', 'composed_memory_gradient', 'composed_source_edges', 'composed_source_nodes', 'composed_source_return', 'composed_passes', 'composed_updates', 'composed_started', 'composed_detach', 'composed_binding', 'composed_loss', 'composed_local_loss', 'composed_terminal_loss', 'composed_local_targets', 'composed_queries', 'composed_peak_nodes', 'composed_trace_count', 'composed_trace_model', 'composed_trace_info', 'composed_trace_memory', 'composed_trace_gradient', 'composed_trace_probability', 'composed_trace_probability_gradient', 'composed_source_diagnostics', 'composed_total_passes', 'composed_oracle_calls']
    source += "\nstruct ComposedRollback {\n" + "\n".join("decltype(::" + field + ") saved_" + field + ";" for field in fields) + "\n};\nstatic ComposedRollback composed_rollback;\n"
    source += "static void composed_copy(void*dst,const void*src,unsigned long n){auto*d=(unsigned char*)dst;auto*s=(const unsigned char*)src;for(unsigned long i=0;i<n;i++)d[i]=s[i];}\n"
    for operation in ("snapshot", "rollback"):
        source += "static void composed_io_" + operation + "(){\n"
        for field in fields:
            saved = "composed_rollback.saved_" + field
            destination, origin = (saved, field) if operation == "snapshot" else (field, saved)
            source += "composed_copy(&" + destination + ",&" + origin + ",sizeof(" + field + "));\n"
        source += "}\n"
    source += Path(__file__).with_name("composed_io.hpp").read_text()
    source += '''
extern "C" __attribute__((force_align_arg_pointer)) void _start(){
    vgi_retained_reset();long result=vgi_composed_stream(PASSES,STEPS,LOCAL,DETACH);long status=result<0?2:0;
    if(vgi_composed_save()<0)status=4;
    __asm__ volatile("syscall"::"a"(60L),"D"(status):"rcx","r11","memory");__builtin_unreachable();}
'''.replace('PASSES',str(passes)).replace('STEPS',str(steps)).replace('LOCAL',str(round(local_weight*1000000))).replace('DETACH',str(int(detach)))
    cpp.write_text(source); report = json.loads((out / 'build.json').read_text())
    report['composed_compile_commands'] = compile_model(root, cpp, out, 'O2', library)
    report.update(composed_graph={'passes': passes, 'steps': steps, 'local_weight': local_weight,
                                  'detach_incoming_memory': detach, 'tape_capacity': capacity,
                                  'aggregation': 'componentwise information join; equal-value pullbacks split equally',
                                  'entry': 'GGRAPH1 through existing graph admission'},
                  source_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
                  module_sha256=hashlib.sha256((out / 'model.imasm').read_bytes()).hexdigest())
    (out / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'runtime': str(out), 'composed_graph': report['composed_graph']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','library','vocabulary','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--passes',type=int,default=2);p.add_argument('--steps',type=int,default=1)
    p.add_argument('--local-weight',type=float,default=.25);p.add_argument('--detach',action='store_true')
    p.add_argument('--capacity',type=int,default=8000000)
    a=p.parse_args();build(a.root.resolve(),a.checkpoint.resolve(),a.library.resolve(),a.vocabulary.resolve(),
                          a.out.resolve(),a.passes,a.steps,a.local_weight,a.detach,a.capacity)
