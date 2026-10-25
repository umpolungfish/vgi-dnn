#!/usr/bin/env python3
"""Recurrent retained inference with a dialetheic decoding heuristic."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from optimizer_state import load_model
from retained_conditioning import evaluate


class Cut:
    T = 0b0001
    F = 0b0010
    t = 0b0100
    f = 0b1000


def info_mass(state):
    return bin(state & (Cut.t | Cut.f)).count("1") / 2.0


def state_name(state):
    names = []
    if state & Cut.T: names.append('T')
    if state & Cut.F: names.append('F')
    if state & Cut.t: names.append('t')
    if state & Cut.f: names.append('f')
    return "".join(names) or "N"


def generate(root, checkpoint, prompts, out, max_new=96, temperature=0.,
             refute_weight=0., seed=7, typed_tuple=False):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    sys.path.insert(0, str(root / "bob3ect/reentry_dnn"))
    from dtransformer_torch import DialetheicTransformerTorch

    model = DialetheicTransformerTorch(
        **json.loads((checkpoint / "config.json").read_text())).double()
    model.load_state_dict(load_model(checkpoint / "model.safetensors"))
    model.eval()

    results = []
    with torch.no_grad():
        for prompt in prompts:
            if typed_tuple:
                from typed_tuple_learning import infer
                result = infer(model, prompt)
                results.append({"prompt": prompt, "text": result["address"],
                                "typed_return": result})
                print(json.dumps({"prompt": prompt, "tuple": result["address"]},
                                 ensure_ascii=False), flush=True)
                continue

            prefix = [256] + list(prompt.encode("utf8")) + [257]
            generated, trace = [], []
            terminated = False
            current_state = Cut.T | Cut.F | Cut.t | Cut.f
            allowed = torch.tensor(list(range(256)) + [258])

            for _ in range(max_new):
                result = evaluate(model, prefix + generated, [258], context=16)
                support, refutation = result["first_logits"]
                sup, ref = support[allowed], refutation[allowed]

                # Peak readouts of each branch over the allowed alphabet.
                si, ri = int(sup.argmax()), int(ref.argmax())
                # Tension: both branches are confident, but about DIFFERENT bytes.
                tension = 0.0
                if si != ri:
                    tension = max(0.0, min(float(sup[si]), float(ref[ri])))

                # Macroscopic carrier update (not per-byte logit sign).
                if tension > 2.0:
                    current_state = Cut.T | Cut.F | Cut.t | Cut.f
                elif float(sup[si]) >= float(ref[ri]):
                    current_state = Cut.T | Cut.t
                else:
                    current_state = Cut.F | Cut.f

                scores = sup + refute_weight * torch.nn.functional.logsigmoid(-ref)
                if (current_state & Cut.T) and (current_state & Cut.F):
                    glue = torch.sqrt(torch.relu(sup) * torch.relu(ref) + 1e-6)
                    scores = scores + 1.5 * glue

                if temperature:
                    chosen = int(torch.multinomial(torch.softmax(scores / temperature, 0), 1))
                else:
                    chosen = int(scores.argmax())
                token = int(allowed[chosen])

                trace.append({
                    "token": token,
                    "support": float(support[token]),
                    "refutation": float(refutation[token]),
                    "state": state_name(current_state),
                    "mass": info_mass(current_state),
                    "tension": tension,
                    "source_tokens_processed": result["source_tokens_processed"],
                })

                if token == 258:
                    terminated = True
                    break
                generated.append(token)

            text = bytes(generated).decode("utf8", errors="replace")
            results.append({"prompt": prompt, "text": text,
                            "generated_bytes": generated,
                            "ended_with_eos": terminated, "trace": trace})
            print(json.dumps({"prompt": prompt, "reply": text},
                             ensure_ascii=False), flush=True)

    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(
            (checkpoint / "model.safetensors").read_bytes()).hexdigest(),
        "parameters": model.num_parameters(),
        "execution": "retained recurrent inference with dialetheic decoding heuristic",
        "temperature": temperature, "refute_weight": refute_weight,
        "max_new_bytes": max_new, "results": results,
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "checkpoint", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--prompt", action="append", required=True)
    p.add_argument("--max-new-bytes", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.)
    p.add_argument("--refute-weight", type=float, default=0.)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--typed-tuple", action="store_true")
    a = p.parse_args()
    if a.max_new_bytes < 1 or a.temperature < 0 or a.refute_weight < 0:
        p.error("invalid generation settings")
    generate(a.root.resolve(), a.checkpoint.resolve(), a.prompt, a.out.resolve(),
             a.max_new_bytes, a.temperature, a.refute_weight, a.seed, a.typed_tuple)
