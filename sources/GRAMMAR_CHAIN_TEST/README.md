# Grammar coordinate-chain stress test

Copy `chain_stress.py` and `run_chain_test.sh` into your **existing** `IMASM_GRAMMAR_DNN_OPS` directory. The test uses the checkpoint that *you trained* there, not an older bundled checkpoint.

Then run `./run_chain_test.sh`. The exact mixed-radix codec supplies the oracle; the learned model is rolled forward step by step with its own predictions as inputs. Results go to `chain_stress_report.json`.

Metrics distinguish **oracle-conditioned single-step accuracy** from **free-rollout final exact tuple accuracy** and **never-diverged trajectory accuracy**. None of these metrics is a measurement of actual IMASM VM execution or SIXTEEN_3 transitions.

Options: `--examples 256 --lengths 1,2,4,8,16,32,64 --device cpu`.
