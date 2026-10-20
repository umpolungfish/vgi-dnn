# Grammar oracle attribution / replay

Unzip **alongside** `GRAMMAR_CERTIFIED_EXECUTION`, `GRAMMAR_COORDINATE_REPAIR`, and `GRAMMAR_CARRIER_LEARNER` under `~/imsgct/vox_grammar_induxion/`.

```bash
cd ~/imsgct/vox_grammar_induxion/GRAMMAR_ORACLE_ATTRIBUTION
chmod +x ./run.sh
./run.sh
```

Outputs `oracle_attribution_report.json`, with **all correction events** (no truncated examples). Each event stores before-state, operation, neural proposal, exact operational target, and whether the proposed state passes the SIC representation check. The report distinguishes oracle detection from SIC frame inversion. Same deterministic case/operation seed as the prior certified execution, but reports are not guaranteed to match in every detail if implementation or checkpoint differs.

Use `./run.sh --cases 256 --steps 256` for a longer evaluation; flags `--ops-checkpoint`, `--carrier-checkpoint`, `--seed`, and `--out` are also available.

Note: exact rational frame reconstruction does not establish semantic correctness of a coordinate/carrier transition. This is **not native IMASM** or Lean kernel verification. Checkpoint files remain untouched.
