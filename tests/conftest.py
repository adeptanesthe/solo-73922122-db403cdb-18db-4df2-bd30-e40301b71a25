"""Shared pytest fixtures: build a small labelled corpus in-memory."""

import pytest

from app.engine import Corpus


@pytest.fixture(scope="session")
def records():
    return [
        # relevant docs (6) — query literal vs synonym split
        {"id": "r1", "title": "t", "abstract": "quantum error correction thresholds",
         "relevant": True},
        {"id": "r2", "title": "t", "abstract": "quantum error correction circuits",
         "relevant": True},
        {"id": "r3", "title": "t", "abstract": "surface code logical qubit syndrome",
         "relevant": True},
        {"id": "r4", "title": "t", "abstract": "surface code syndrome decoherence",
         "relevant": True},
        {"id": "r5", "title": "t", "abstract": "logical qubit decoherence mitigation",
         "relevant": True},
        {"id": "r6", "title": "t", "abstract": "syndrome stabilizer logical qubit gates",
         "relevant": True},
        # non-relevant docs (6)
        {"id": "n1", "title": "t", "abstract": "neural network graph model",
         "relevant": False},
        {"id": "n2", "title": "t", "abstract": "neural attention network layers",
         "relevant": False},
        {"id": "n3", "title": "t", "abstract": "graph neural convolution features",
         "relevant": False},
        {"id": "n4", "title": "t", "abstract": "crispr gene editing guide rna",
         "relevant": False},
        {"id": "n5", "title": "t", "abstract": "crispr screening off target effects",
         "relevant": False},
        {"id": "n6", "title": "t", "abstract": "climate model ensemble projections",
         "relevant": False},
    ]


@pytest.fixture(scope="session")
def corpus(records):
    return Corpus.from_records(records)
