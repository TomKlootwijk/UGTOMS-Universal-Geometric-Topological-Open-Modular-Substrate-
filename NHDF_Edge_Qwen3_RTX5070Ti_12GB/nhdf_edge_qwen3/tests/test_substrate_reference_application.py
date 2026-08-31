from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from nhdf_edge.substrate_contract import validate_application_manifest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "ugtoms_sclp_reference.py"
EVIDENCE = ROOT / "substrate" / "applications" / "evidence" / "ugtoms-sclp-reference-v0.1.json"
MANIFEST = ROOT / "substrate" / "applications" / "ugtoms-sclp-reference-v0.1.json"


def _run(output: Path) -> bytes:
    subprocess.run(
        [sys.executable, "-B", str(EXAMPLE), "--output", str(output)],
        cwd=ROOT,
        check=True,
    )
    return output.read_bytes()


def test_reference_cli_is_byte_deterministic_and_matches_committed_evidence(tmp_path: Path) -> None:
    first = _run(tmp_path / "first.json")
    second = _run(tmp_path / "second.json")

    assert first == second == EVIDENCE.read_bytes()
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
    assert first.endswith(b"\n") and b"\r\n" not in first


def test_reference_manifest_validates_current_kernel_profiles_and_evidence() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    result = validate_application_manifest(manifest, repository_root=ROOT)

    assert result["ok"] is True
    assert result["selected_profiles"] == ["nhdf-v0.1", "sclp-foundational"]
    assert result["mapping_count"] == 12
    assert result["profile_requirement_count"] == 10
    assert manifest["format"] == "ugtoms-application-manifest-0.2"
    assert set(manifest["mappings"]) == {
        "typed",
        "vector",
        "kinematic",
        "geometry",
        "topology",
        "packing",
        "predicate",
        "operator",
        "self_reference",
    }
    evidence_record = next(row for row in manifest["evidence"] if row["id"] == "reference-replay")
    assert evidence_record["bytes"] == EVIDENCE.stat().st_size
    assert evidence_record["sha256"] == hashlib.sha256(EVIDENCE.read_bytes()).hexdigest()
    coverage = {
        (row["profile_id"], row["requirement_id"])
        for row in evidence_record["claim_coverage"]
    }
    declared = {
        (profile_id, requirement["id"])
        for profile_id in ("nhdf-v0.1", "sclp-foundational")
        for requirement in json.loads(
            (ROOT / "substrate" / "profiles" / f"{profile_id}.json").read_text(
                encoding="utf-8"
            )
        )["evidence_requirements"]
    }
    assert coverage == declared
    assert manifest["self_reference"]["may_propose_extensions"] is False


def test_reference_has_bounded_next_generation_semantics_and_stable_display_prefix() -> None:
    result = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    feedback = result["graph"]["feedback"]
    logic = result["logic"]
    prefix = result["packing"]["stable_display_prefix"]
    key_round_trips = result["packing"]["key_round_trips"]
    sweep = result["geometry"]["translational_sweep"]

    assert result["graph"]["fixed_point_engine"] is False
    assert feedback["fixed_point_claim"] is False
    assert feedback["target_generation"] == feedback["source_generation"] + 1
    assert result["transition"]["generation"] == {"source": 0, "target": 1}

    assert logic["semantics"] == "three-valued"
    assert logic["fuzzy_logic"] is False
    assert set(logic["allowed_predicates"]) == {"FALSE", "TRUE", "INDETERMINATE"}
    assert set(logic["event"].values()) <= {
        "FALSE",
        "TRUE",
        "INDETERMINATE",
        "VERIFIED",
    }
    assert logic["indeterminate_probe"] == "INDETERMINATE"
    assert key_round_trips["contiguous_round_trip"] is True
    assert key_round_trips["morton_round_trip"] is True
    assert sweep["certified"] is True
    assert sweep["earliest_impact_claim"] is False
    assert sweep["parameter_interval"][0] < sweep["parameter_interval"][1]
    assert sweep["endpoint_distances"][0] > 0.0
    assert sweep["endpoint_distances"][1] <= 0.0
    assert all(
        proof["passed"] is True
        for profile in result["proof_inventory"].values()
        for proof in profile.values()
    )
    assert all(
        row["bypassed"] is True for row in result["scope"]["bypassed"].values()
    )

    assert prefix["short_ids"] == prefix["long_prefix_ids"]
    assert prefix["short_count"] < prefix["long_count"] < result["packing"]["recipe"]["instance_count"]
    assert result["display_boundary"] == {
        "authoritative_component_records": 1,
        "collider_identity": False,
        "ecs_identity": False,
        "forbidden_generated_authority": [
            "entity_id",
            "collider",
            "gameplay_state",
            "ecs_component",
        ],
        "gameplay_authority": False,
        "generated_type": "GeneratedDisplayInstance",
        "render_only": True,
    }
