#!/usr/bin/env python3
"""Interleave source-bound shared-port queries with retained training."""
import argparse
from pathlib import Path
from add_operation_replay import build

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('batch','records','out','cursor'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--count',type=int,default=512);a=p.parse_args();build(a.batch.resolve(),a.records.resolve(),a.out.resolve(),a.count,a.cursor.resolve())
