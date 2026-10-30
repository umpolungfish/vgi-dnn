# Grammar zeta–SIC exact inversion oracle

This extends the earlier `GRAMMAR_SIC_GATE` rather than replacing it. The source is the user's `Imscribing.Millennium.ZetaSICDualFrame` Lean excerpt.

The finite-dimensional SIC Gram operator at dimension d is `G = d/(d+1) I + 1/(d+1) J` on d² coordinates. Its inverse is `G⁻¹ = (d+1)/d I − 1/d² J`. The supplied Lean construction expresses inverse coefficients using `riemannZeta (-1) = -1/12` for d=2,4,12. `zeta_oracle.py` uses rational arithmetic with `fractions.Fraction` and checks both exact inverse directions and reconstruction of all basis vectors. This is an independently executable arithmetic verification of the stated formulas, not a Lean proof checker.

The package also runs the earlier numerical tetrahedral operator-frame checks and independent-source FOUR evidence tests. The different levels of evidence remain separate.

Run in the extracted folder:

```bash
chmod +x ./run.sh
./run.sh
```

Writes `zeta_sic_oracle_report.json`.

**Boundaries:** The provided Lean code states `analytic_dual_frame_witness`, which conjoins the analytic zeta value, tetrahedral dual inversion, and existing Grammar self-fusion theorem. The conjunctive theorem does not alone construct a map from an entire analytical continuum into Grammar states; likewise the ququart and 12-dimensional Gram identities do not, by themselves, exhibit physical SIC projectors in those dimensions. This package does not run Lean, the native SIC Rust code, or IMASM, and does not assert these as verified.
