"""Generate a deterministic demo corpus of labelled paper abstracts.

The corpus is engineered so that several query-expansion scenarios are
reproducible:

* the baseline query ``quantum error correction`` misses relevant abstracts
  phrased with synonym vocabulary (surface code / logical qubit / syndrome /
  decoherence), so expansion genuinely raises recall;
* some candidate terms (``neural``, ``network``) also occur in many
  non-relevant machine-learning abstracts, creating a real false-detection
  trade-off;
* one relevant abstract deliberately uses only rare, single-document words,
  so min_df=2 makes 100% recall unreachable — exercising the failure path;
* the four topics share a little vocabulary to keep the problem non-trivial.

Run:  python3 scripts/generate_demo_data.py
Out:  data/demo_corpus.csv  (id,title,abstract,relevant)
"""

from __future__ import annotations

import csv
import os
import random

RNG = random.Random(20261122)

# Each abstract is assembled from fragments.  Every content word below occurs
# in at least two abstracts so the TF-IDF vocabulary (min_df=2) keeps it.
QEC_CORE = [
    "Quantum error correction protects fragile quantum states against noise.",
    "Fault tolerant quantum error correction requires repeated syndrome checks.",
    "We analyze quantum error correction thresholds under depolarizing noise.",
    "Quantum error correction overhead is estimated for superconducting qubits.",
    "Encoded quantum error correction circuits reduce logical error rates.",
    "This quantum error correction scheme interleaves gates with measurement.",
]
QEC_SYNONYM = [  # relevant abstracts that avoid the literal query phrase
    "A surface code implementation suppresses errors on a logical qubit array.",
    "The surface code lattice maps each logical qubit onto physical qubits.",
    "Syndrome decoding identifies bit flip and phase flip events efficiently.",
    "Real time syndrome extraction enables fault tolerant stabilizer rounds.",
    "Decoherence mitigation via repeated stabilizer measurement extends coherence time.",
    "We benchmark logical qubit lifetime after surface code stabilizer cycles.",
    "Syndrome statistics reveal correlated decoherence channels in the device.",
    "A rotated surface code layout reduces physical qubit overhead per logical qubit.",
    "Decoherence limits the achievable code distance of near term processors.",
]
QEC_HARD = [  # deliberately unreachable: every content word is document-unique
    "Bosonic cat-cat poisoning mitigates transmon dephasing "
    "via Kerr-driven pumping within niobium resonators.",
    "Fluxonium parity-selective tunneling proceeds without ancilla "
    "feedback atop a tantalum wafer.",
]
ML = [
    "Graph neural network models learn molecular representations for drug discovery.",
    "A deep neural network combines attention with graph convolution layers.",
    "Neural architecture search reduces training cost of large network models.",
    "We train a graph network on protein structures to predict binding affinity.",
    "Attention based neural models outperform kernel methods on benchmark datasets.",
    "A hybrid neural decoder compresses graph features for downstream classification.",
    "Transfer learning warms up the network before full neural fine tuning.",
    "Graph convolution layers aggregate neighbor features inside the network.",
    "We compare neural baselines against random forests on chemical datasets.",
    "Self supervised pretraining improves neural generalization on small graphs.",
    # The next two deliberately share vocabulary with relevant QEC abstracts:
    # they make 'surface' / 'code' / 'syndrome' polluted candidate terms,
    # creating a genuine recall-versus-false-detection trade-off.
    "A neural syndrome classifier processes surface code measurement streams for inference.",
    "Graph networks emulate surface code decoding under syndrome noise statistics.",
    "Tensor network inspired layers broaden the neural architecture search space.",
    "The neural decoder runs on a control stack for real time statistical inference.",
]
CRISPR = [
    "CRISPR Cas9 gene editing modifies target loci with guide RNA specificity.",
    "Off target effects of CRISPR editing are measured by deep sequencing.",
    "Base editing extends CRISPR precision without double strand DNA breaks.",
    "Guide RNA design tools improve CRISPR Cas9 specificity in human cells.",
    "We apply CRISPR screening to identify cancer dependency genes.",
    "Prime editing combines reverse transcription with CRISPR nickase activity.",
    "Delivery vectors limit CRISPR therapeutic adoption in clinical settings.",
    "CRISPR interference silences promoters without altering the DNA sequence.",
]
CLIMATE = [
    "Climate models project sea surface temperature trends under emission scenarios.",
    "Ensemble climate simulations quantify uncertainty in regional rainfall patterns.",
    "We downscale climate projections for coastal flood risk assessment.",
    "Cloud feedbacks remain the largest source of climate model uncertainty.",
    "Satellite observations constrain climate model parameterization of aerosols.",
    "A reduced order climate emulator reproduces century scale warming trajectories.",
]

TITLES_QEC = [
    "Thresholds of fault tolerant encoding",
    "Stabilizer circuits on superconducting hardware",
    "Logical error rate estimation",
    "Encoded circuit benchmarking",
    "Overhead analysis for encoded processors",
    "Measurement interleaved protection schemes",
    "Surface code demonstration on a qubit array",
    "Lattice surgery for encoded qubits",
    "Efficient decoding of stabilizer events",
    "Real time stabilizer extraction",
    "Coherence preservation via stabilizers",
    "Logical lifetime benchmarking",
    "Correlated noise in stabilizer data",
    "Rotated layout optimization",
    "Code distance limits in near term devices",
    "Bosonic mode stabilization",
    "Fluxonium parity pulse engineering",
]


def _build() -> list[dict]:
    records: list[dict] = []
    idx = 1

    def add(title: str, abstract: str, relevant: bool, topic: str) -> None:
        nonlocal idx
        records.append({
            "id": f"P{idx:03d}",
            "title": f"{title} ({topic})",
            "abstract": abstract,
            "relevant": "true" if relevant else "false",
        })
        idx += 1

    # --- relevant: quantum error correction (16 docs) ------------------- #
    for frag in QEC_CORE:
        add(TITLES_QEC[idx - 1], frag, True, "QEC")
    for frag in QEC_SYNONYM:
        add(TITLES_QEC[idx - 1], frag, True, "QEC")
    for frag in QEC_HARD:
        add(TITLES_QEC[idx - 1], frag, True, "QEC")

    # --- non-relevant: other topics (32 docs) --------------------------- #
    ml_titles = [
        "Molecular representation learning", "Attention graph models",
        "Architecture search at scale", "Protein binding prediction",
        "Attention versus kernels", "Hybrid graph decoders",
        "Transfer learning workflows", "Neighbor feature aggregation",
        "Baselines for chemical data", "Pretraining on small graphs",
        "Quantum inspired learning", "Parameterized circuits reviewed",
        "Tensor inspired layers", "Real time neural control",
    ]
    for title, frag in zip(ml_titles, ML):
        add(title, frag, False, "ML")
    crispr_titles = [
        "Cas9 specificity analysis", "Off target sequencing study",
        "Precision base editing", "Guide design for human cells",
        "Cancer dependency screens", "Prime editing workflow",
        "Delivery for therapeutics", "Interference based silencing",
    ]
    for title, frag in zip(crispr_titles, CRISPR):
        add(title, frag, False, "CRISPR")
    climate_titles = [
        "Sea temperature projections", "Ensemble uncertainty quantification",
        "Downscaled flood risk", "Cloud feedback uncertainty",
        "Satellite constrained aerosols", "Reduced order warming emulator",
    ]
    for title, frag in zip(climate_titles, CLIMATE):
        add(title, frag, False, "CLIMATE")

    RNG.shuffle(records)  # shuffle so labels aren't positionally ordered
    return records


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.normpath(os.path.join(out_dir, "demo_corpus.csv"))
    records = _build()
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "title", "abstract", "relevant"])
        writer.writeheader()
        writer.writerows(records)
    n_rel = sum(1 for r in records if r["relevant"] == "true")
    print(f"wrote {len(records)} records ({n_rel} relevant) -> {out_path}")


if __name__ == "__main__":
    main()
