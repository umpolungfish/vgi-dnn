"""Qubit SIC numerical verification gate. Qubit SIC numerical reconstruction and frame certificate.
Tetra vertices N,T,F,B are coordinate labels, not FOUR truth values.
"""
import json
import math
from dataclasses import dataclass
import numpy as np

LABELS = ('N','T','F','B')
VERTICES = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=np.float64)/math.sqrt(3)
I = np.eye(2,dtype=complex)
PAULI = (np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.array([[1,0],[0,-1]],complex))
PROJECTORS = np.array([(I+sum(float(n[k])*PAULI[k] for k in range(3)))/2 for n in VERTICES])
EFFECTS = PROJECTORS/2
DUALS = 3*PROJECTORS-I
BASIS = (np.array([[1,0],[0,0]],complex),np.array([[0,1],[0,0]],complex),np.array([[0,0],[1,0]],complex),np.array([[0,0],[0,1]],complex))


def analyze(op):
    op=np.asarray(op,dtype=complex)
    if op.shape != (2,2): raise ValueError('requires 2x2 operator')
    return np.array([np.trace(e @ op) for e in EFFECTS])


def synthesize(coords):
    coords=np.asarray(coords,dtype=complex)
    if coords.shape != (4,): raise ValueError('requires four SIC coordinates')
    return np.einsum('i,ijk->jk',coords,DUALS)


def bloch_to_probs(r, tolerance=1e-12):
    r=np.asarray(r,dtype=float)
    if r.shape != (3,) or not np.isfinite(r).all() or float(r@r)> (1+tolerance)**2:
        raise ValueError('invalid Bloch vector')
    return (1+VERTICES@r)/4


def probs_to_bloch(p, tolerance=1e-12):
    p=np.asarray(p,dtype=float)
    if p.shape!=(4,) or not np.isfinite(p).all() or np.min(p)<-tolerance or abs(float(sum(p))-1)>tolerance:
        raise ValueError('not a normalized four-coordinate distribution')
    r=3*(p@VERTICES)
    if float(r@r)>(1+tolerance)**2: raise ValueError('SIC distribution lies outside quantum region')
    return r


def urgleichung_coefficients(p):
    """Signed dual coefficients. Negatives must not be clipped."""
    p=np.asarray(p,dtype=float)
    probs_to_bloch(p)
    return 3*p-0.5


def certificate():
    frob=lambda m:float(np.linalg.norm(m,ord='fro'))
    geometry=VERTICES@VERTICES.T
    ideal=np.full((4,4),-1/3);np.fill_diagonal(ideal,1)
    basis_units=[v for b in BASIS for v in (b,1j*b)]
    matrix_unit_error=max(frob(synthesize(analyze(b))-b) for b in basis_units)
    purity=max(frob(p@p-p) for p in PROJECTORS)
    equiangular=max(abs(float(np.trace(PROJECTORS[i]@PROJECTORS[j]).real)-1/3) for i in range(4) for j in range(i+1,4))
    second_moment=frob(np.einsum('ia,ib->ab',VERTICES,VERTICES)/4-np.eye(3)/3)
    return {
      'matrix_unit_reconstruction_frobenius_max':matrix_unit_error,
      'normalization_frobenius':frob(sum(EFFECTS)-I),
      'equiangular_max':float(equiangular),
      'geometry_max':float(np.max(abs(geometry-ideal))),
      'projector_purity_max':purity,
      'duality_max':float(np.max(abs(np.array([[np.trace(EFFECTS[i]@DUALS[j]) for j in range(4)] for i in range(4)])-np.eye(4)))),
      'positivity_min_projector_eigenvalue':float(min(np.linalg.eigvalsh(p).min() for p in PROJECTORS)),
      'second_moment_frobenius':second_moment,
      'state_examples':[{'bloch':r,'p':bloch_to_probs(r).tolist(),'reconstructed_bloch':probs_to_bloch(bloch_to_probs(r)).tolist()} for r in ([0,0,0],[0,0,1],[0.5,-0.3,0.2])],
      'zeta_continuum_link':'zeta link tracked by the exact SIC/zeta module',
      'native_sic_tool_crosscheck':'not run: native executable not provided'
    }


@dataclass(frozen=True)
class Reading:
    proposition:str
    source:str
    side:str # support or refute


class FourEvidence:
    def __init__(self):self.records={}
    def put(self,proposition,source,side):
        if side not in ('support','refute'):raise ValueError('side must be support or refute')
        self.records[(proposition,source)]=Reading(proposition,source,side)
    def verdict(self,proposition):
        supporters={r.source for r in self.records.values() if r.proposition==proposition and r.side=='support'}
        refuters={r.source for r in self.records.values() if r.proposition==proposition and r.side=='refute'}
        if supporters and refuters and any(a!=b for a in supporters for b in refuters):return 'B'
        if supporters:return 'T'
        if refuters:return 'F'
        return 'N'


def main():
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--tolerance',type=float,default=1e-12);a.add_argument('--out',default='sic_gate_report.json');args=a.parse_args()
    if args.tolerance<=0:raise ValueError('tolerance must be positive')
    cert=certificate()
    residuals={k:v for k,v in cert.items() if k.endswith(('_max','_frobenius')) or k=='normalization_frobenius'}
    # Positivity is a lower eigenvalue, not a residual.
    cert['numerical_gate_pass']=all(v<=args.tolerance for v in residuals.values()) and cert['positivity_min_projector_eigenvalue']>=-args.tolerance
    cert['tolerance']=args.tolerance
    with open(args.out,'w') as f:json.dump(cert,f,indent=2)
    print(json.dumps({'numerical_gate_pass':cert['numerical_gate_pass'],'max_residual':max(residuals.values()),'zeta_continuum_link':cert['zeta_continuum_link'],'report':args.out}))
    if not cert['numerical_gate_pass']:raise SystemExit(1)
if __name__=='__main__':main()
