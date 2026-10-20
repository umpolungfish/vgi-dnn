"""Exact rational SIC Gram/dual-Gram oracle for d=2,4,12.

Implements the finite-dimensional matrix identities supplied in
Imscribing.Millennium.ZetaSICDualFrame, independently of Lean.
Do not conflate this check with verified Lean compilation or a global oracle.
"""
from fractions import Fraction as F
import json

ZETA_NEG_ONE = F(-1, 12)  # value cited from Lean's riemannZeta_neg_nat_eq_bernoulli


def gram_coefficients(d):
    if d not in (2, 4, 12):
        raise ValueError('Only dimensions supported by the supplied witness: 2, 4, 12')
    return F(d, d+1), F(1, d+1)  # a I + b J


def dual_coefficients(d):
    if d == 2:
        return F(3, 2), 3 * ZETA_NEG_ONE
    if d == 4:
        return 1 - 3 * ZETA_NEG_ONE, F(3, 4) * ZETA_NEG_ONE
    if d == 12:
        return 1 - ZETA_NEG_ONE, ZETA_NEG_ONE / 12
    raise ValueError('Unsupported dimension')


def apply_uniform_frame(coeffs, vector):
    a, b = coeffs
    total = sum(vector, F(0))
    return [a * x + b * total for x in vector]


def split(c, d):
    if len(c) != d*d: raise ValueError(f'Expected {d*d} coordinates')
    return apply_uniform_frame(gram_coefficients(d), c)


def fuse(p, d):
    if len(p) != d*d: raise ValueError(f'Expected {d*d} coordinates')
    return apply_uniform_frame(dual_coefficients(d), p)


def composition_coefficients(first, second, n):
    a, b = first
    c, e = second
    return (a*c, a*e + b*c + n*b*e)


def exact_certificate(d):
    n=d*d
    gram=gram_coefficients(d)
    dual=dual_coefficients(d)
    left=composition_coefficients(dual,gram,n)
    right=composition_coefficients(gram,dual,n)
    assert left == (F(1),F(0)) and right == (F(1),F(0)), (d,left,right)
    # Exact check on every coordinate basis vector (not floating-point sampling).
    for i in range(n):
        v=[F(int(j==i)) for j in range(n)]
        assert fuse(split(v,d),d)==v
    return {'dimension':d,'coordinates':n,
        'gram_I':str(gram[0]),'gram_J':str(gram[1]),
        'dual_I':str(dual[0]),'dual_J':str(dual[1]),
        'left_inverse_exact':True,'right_inverse_exact':True,
        'basis_vectors_checked':n,'max_error_exact':'0'}


def report():
    from sic_gate import certificate
    sic=certificate()
    residual_keys=['matrix_unit_reconstruction_frobenius_max', 'normalization_frobenius',
        'equiangular_max','geometry_max','projector_purity_max','duality_max',
        'second_moment_frobenius']
    numerical_pass=all(sic[k]<=1e-12 for k in residual_keys) and sic['positivity_min_projector_eigenvalue']>=-1e-12
    return {'zeta_value_used':'-1/12', 'zeta_theorem':'riemannZeta (-1) = -1/12 (Lean theorem supplied by the author)',
        'exact_gram_inverses':[exact_certificate(d) for d in (2,4,12)],
        'qubit_numerical_frame_pass':numerical_pass,
        'qubit_max_residual':max(sic[k] for k in residual_keys),
        'scope':{'proved_by_this_program':'rational SIC Gram inversion and reconstruction for d=2,4,12',
                 'not_verified_here':['Lean module compilation','Grammar Frobenius self-fusion theorem','banked-weight State traces','analytic continuation beyond zeta(-1)','existence of WH SIC fiducials in general dimensions','native IMASM execution']}}

if __name__=='__main__':
    result=report()
    with open('zeta_sic_oracle_report.json','w') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    if not result['qubit_numerical_frame_pass']:raise SystemExit(1)
