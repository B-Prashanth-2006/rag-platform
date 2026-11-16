"""Unit tests for Reciprocal Rank Fusion (RRF) — written first (TDD)."""

from __future__ import annotations

import uuid

import pytest

from rag.domain.entities import RetrievedChunk
from rag.retrieval.fusion import reciprocal_rank_fusion


def _make_chunk(
    chunk_id: uuid.UUID | None = None,
    score: float = 1.0,
    source: str = "dense",
    text: str = "chunk text",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=text,
        page=1,
        section="Section 1",
        score=score,
        source=source,
    )


class TestReciprocalRankFusion:
    def test_hand_computed_expected_scores(self) -> None:
        cid_a = uuid.UUID("00000000-0000-0000-0000-000000000001")
        cid_b = uuid.UUID("00000000-0000-0000-0000-000000000002")
        cid_c = uuid.UUID("00000000-0000-0000-0000-000000000003")

        # List 1: [A, B] (ranks 1, 2)
        list1 = [
            _make_chunk(cid_a, score=0.9, source="dense"),
            _make_chunk(cid_b, score=0.8, source="dense"),
        ]
        # List 2: [C, A] (ranks 1, 2)
        list2 = [
            _make_chunk(cid_c, score=0.95, source="sparse"),
            _make_chunk(cid_a, score=0.75, source="sparse"),
        ]

        # k = 60
        # A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
        # C: 1/(60+1) = 1/61
        # B: 1/(60+2) = 1/62
        fused = reciprocal_rank_fusion([list1, list2], k=60)

        assert len(fused) == 3
        assert fused[0].chunk_id == cid_a
        assert fused[1].chunk_id == cid_c
        assert fused[2].chunk_id == cid_b

        expected_score_a = (1.0 / 61.0) + (1.0 / 62.0)
        expected_score_c = 1.0 / 61.0
        expected_score_b = 1.0 / 62.0

        assert pytest.approx(fused[0].score, rel=1e-5) == expected_score_a
        assert pytest.approx(fused[1].score, rel=1e-5) == expected_score_c
        assert pytest.approx(fused[2].score, rel=1e-5) == expected_score_b

    def test_chunk_in_both_lists_outranks_chunk_in_one(self) -> None:
        cid_shared = uuid.uuid4()
        cid_single1 = uuid.uuid4()
        cid_single2 = uuid.uuid4()

        # In list 1: single1 is rank 1, shared is rank 2
        list1 = [_make_chunk(cid_single1), _make_chunk(cid_shared)]
        # In list 2: single2 is rank 1, shared is rank 2
        list2 = [_make_chunk(cid_single2), _make_chunk(cid_shared)]

        fused = reciprocal_rank_fusion([list1, list2], k=60)
        # Shared score = 1/62 + 1/62 = 2/62 ≈ 0.032258
        # single1 score = 1/61 ≈ 0.016393
        # single2 score = 1/61 ≈ 0.016393
        assert fused[0].chunk_id == cid_shared

    def test_weights_scale_ranks(self) -> None:
        cid_1 = uuid.UUID("00000000-0000-0000-0000-000000000001")
        cid_2 = uuid.UUID("00000000-0000-0000-0000-000000000002")

        list1 = [_make_chunk(cid_1)]
        list2 = [_make_chunk(cid_2)]

        # Weight list2 heavily
        fused = reciprocal_rank_fusion([list1, list2], k=60, weights=[1.0, 3.0])
        # cid_2 score = 3 * 1/61 = 3/61
        # cid_1 score = 1 * 1/61 = 1/61
        assert fused[0].chunk_id == cid_2
        assert fused[1].chunk_id == cid_1
        assert pytest.approx(fused[0].score, rel=1e-5) == 3.0 / 61.0

    def test_deterministic_tie_breaking_by_chunk_id(self) -> None:
        # Two chunks with identical rank in two disjoint single lists
        cid_smaller = uuid.UUID("00000000-0000-0000-0000-000000000001")
        cid_larger = uuid.UUID("00000000-0000-0000-0000-000000000002")

        list1 = [_make_chunk(cid_larger)]
        list2 = [_make_chunk(cid_smaller)]

        fused = reciprocal_rank_fusion([list1, list2], k=60)
        assert fused[0].chunk_id == cid_smaller
        assert fused[1].chunk_id == cid_larger

    def test_output_source_is_fused(self) -> None:
        c = _make_chunk(source="dense")
        fused = reciprocal_rank_fusion([[c]])
        assert fused[0].source == "fused"

    def test_empty_lists(self) -> None:
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []
