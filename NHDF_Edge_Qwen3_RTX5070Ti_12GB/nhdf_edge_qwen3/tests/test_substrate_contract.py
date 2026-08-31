from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from nhdf_edge import substrate_contract as contracts


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sealed_record(
    root: Path,
    path: Path,
    *,
    record_id: str,
    role: str,
    title: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "id": record_id,
        "role": role,
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": contracts.sha256_file(path),
    }
    if title is not None:
        record["title"] = title
    return record


def _committed_source_record(
    root: Path,
    path: Path,
    *,
    source_id: str | None = None,
    source_class: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "availability": "COMMITTED",
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": contracts.sha256_file(path),
    }
    if source_id is not None:
        record["source_id"] = source_id
    if source_class is not None:
        record["class"] = source_class
    return record


def _bounds(prefix: str) -> list[dict[str, object]]:
    return [
        {
            "id": f"{prefix}-bound",
            "resource": "bounded working state",
            "limit": 64,
            "unit": "records",
            "enforcement": "Reject generation before the bound is exceeded.",
        }
    ]


def _failures(prefix: str) -> list[dict[str, str]]:
    return [
        {
            "id": f"{prefix}-failure",
            "condition": "The declared relation is outside its bounded domain.",
            "observable": "The operation returns an explicit invalid status.",
            "handling": "Stop the transition and retain the previous lineage head.",
        }
    ]


def _mappings(
    prefix: str,
    *,
    evidence_id: str,
    disposition: str,
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for category in contracts.MAPPING_CATEGORIES:
        mapping_id = f"{prefix}-{category}"
        result[category] = [
            {
                "id": mapping_id,
                "domain": [f"{category} input state"],
                "codomain": [f"{category} bounded result"],
                "definition": (
                    f"A deterministic {category} mapping with declared units, "
                    "limits, and an observable failure result."
                ),
                "primitive_refs": [category],
                "bound_refs": [f"{prefix}-bound"],
                "failure_refs": [f"{prefix}-failure"],
                "evidence_refs": [evidence_id],
                "disposition": disposition,
            }
        ]
    return result


def _profile(
    profile_id: str,
    prefix: str,
    *,
    kernel_sha256: str,
    source_record: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "format": contracts.PROFILE_FORMAT,
        "profile_id": profile_id,
        "title": f"Selectable {profile_id} profile",
        "classification": "SELECTABLE_PROFILE",
        "kernel_id": contracts.KERNEL_ID,
        "kernel_sha256": kernel_sha256,
        "kernel_stage_mapping": {
            "profile_pipeline": "local_support -> compatibility -> guard_crossing -> verified_event -> route_transition -> lineage -> novelty_log"
        },
        "mappings": {
            category: f"Explicit {category} mapping for {profile_id}."
            for category in contracts.MAPPING_CATEGORIES
        },
        "resource_bounds": ["finite state", "finite branches"],
        "failure_modes": ["invalid typed input", "resource bound exceeded"],
        "evidence_requirements": ["reference vectors", "bounded replay trace"],
        "source_records": [] if source_record is None else [source_record],
    }


def _contract_repository(tmp_path: Path, *, provisional_sclp: bool = False) -> Path:
    root = tmp_path / "repository"
    sources = root / "sources"
    sources.mkdir(parents=True)
    ugts_source = sources / "ugts-foundation.txt"
    nhdf01_source = sources / "nhdf-v0.1.txt"
    nhdf03_source = sources / "nhdf-v0.3.txt"
    sclp_source = sources / "sclp-foundational.txt"
    ugts_source.write_text("typed UGTS/base-NHDF foundation\n", encoding="utf-8")
    nhdf01_source.write_text("NHDF v0.1 profile\n", encoding="utf-8")
    nhdf03_source.write_text("NHDF v0.3 CCD profile\n", encoding="utf-8")
    sclp_source.write_text("SCLP foundational profile\n", encoding="utf-8")

    kernel_source = _committed_source_record(
        root,
        ugts_source,
        source_id="kernel-source",
        source_class="FOUNDATIONAL_SOURCE",
    )
    kernel = {
        "format": contracts.KERNEL_FORMAT,
        "kernel_id": contracts.KERNEL_ID,
        "title": "Compact UGTS/base-NHDF kernel",
        "lineage": {
            "semantic_base": "Base NHDF representation and closed update.",
            "early_executable_algebra": "Early UGTS typed executable algebra.",
            "referential_layer": "Content-addressed typed definition graph.",
            "foundational_correction": "Swept-cone log-polar packing correction.",
            "execution_discipline": "Bounded support-to-novelty handoff.",
            "later_profiles_do_not_replace_base": True,
        },
        "identity": {
            "definition": "A finite typed algebra that compactly generates geometric-topological state.",
            "not_a_renderer": True,
            "not_query_calculus_only": True,
            "not_general_ai_by_claim": True,
            "learned_components_are_optional_consumers": True,
        },
        "state_schema": {
            "continuous": ["position", "linear_time", "velocity", "acceleration"],
            "discrete": ["generation", "orientation", "generative_address", "lineage_head"],
            "quality": ["uncertainty_interval", "resource_status", "failure_status"],
        },
        "symbol_firewall": [
            {
                "symbols": ["linear_time", "phase"],
                "rule": "Linear time and wrapped phase are distinct.",
            }
        ],
        "definition_graph": {
            "node_fields": [
                "id",
                "kind",
                "domain",
                "codomain",
                "dependencies",
                "evaluation_phase",
                "equation_or_algorithm",
                "units",
                "bounds",
                "failure_modes",
                "provenance",
                "content_hash",
            ],
            "same_generation": "A content-addressed acyclic typed graph.",
            "instance_rule": "Instances reference definitions; coordinates are not identity.",
            "pipeline_rule": "Execution order is explicit and topologically resolved.",
            "self_reference_rule": "Feedback enters the next generation through an explicit edge.",
            "unproven_claim": "Unrestricted fixed points are not implemented.",
        },
        "composed_execution": list(contracts.COMPOSED_EXECUTION),
        "canonical_chain": [
            {"id": stage, "role": f"Bounded {stage} handoff."}
            for stage in contracts.CANONICAL_TRANSITION_CHAIN
        ],
        "mappings": {
            category: {
                "required": list(contracts.KERNEL_MAPPING_REQUIREMENTS[category]),
                "rule": f"Explicit bounded {category} semantics.",
            }
            for category in contracts.MAPPING_CATEGORIES
        },
        "source_records": [kernel_source],
        "policy": {
            "determinism": {
                "required": True,
                "discrete_state": "Canonical JSON, fixed ordering, and SHA-256.",
                "geometric_numeric_backend": "Declared binary64 reference.",
                "rounding": "Explicit per codec.",
                "uncertainty": "Bounded interval or INDETERMINATE.",
                "randomness": "Only a recorded deterministic seed is permitted.",
            },
            "resource_bounds_required": True,
            "automatic_extension_promotion": False,
            "same_generation_cycles": False,
            "exogenous_events_must_be_logged": True,
            "legacy_repository_mode": "READ_ONLY_SELECTIVE_PROVENANCE",
            "bulk_legacy_import": False,
            "learned_semantics_may_only_rank_or_propose": True,
        },
    }
    kernel_path = root / contracts.DEFAULT_KERNEL_PATH
    _write_json(kernel_path, kernel)
    kernel_sha256 = contracts.sha256_file(kernel_path)

    profile_specs = (
        (
            "nhdf-v0.1",
            "nhdf01",
            nhdf01_source,
            "FORMAL_PROFILE",
            "ACTIVE",
        ),
        (
            "nhdf-v0.3-ccd",
            "nhdf03",
            nhdf03_source,
            "FORMAL_PROFILE",
            "ACTIVE",
        ),
        (
            "sclp-foundational",
            "sclp",
            sclp_source,
            "PROFILE_SOURCE",
            "PROVISIONAL" if provisional_sclp else "ACTIVE",
        ),
    )
    entries: list[dict[str, object]] = []
    for profile_id, prefix, source_path, _role, status in profile_specs:
        source_record = None
        if status == "ACTIVE":
            source_record = _committed_source_record(root, source_path)
        profile = _profile(
            profile_id,
            prefix,
            kernel_sha256=kernel_sha256,
            source_record=source_record,
        )
        profile_path = root / "substrate" / "profiles" / f"{profile_id}.json"
        _write_json(profile_path, profile)
        entries.append(
            {
                "profile_id": profile_id,
                "path": profile_path.relative_to(root).as_posix(),
                "sha256": contracts.sha256_file(profile_path),
                "status": status,
                "selectable": status == "ACTIVE",
            }
        )

    registry = {
        "format": contracts.PROFILE_REGISTRY_FORMAT,
        "kernel": {
            "kernel_id": contracts.KERNEL_ID,
            "path": kernel_path.relative_to(root).as_posix(),
            "sha256": kernel_sha256,
        },
        "automatic_promotion": False,
        "profiles": entries,
    }
    _write_json(root / contracts.DEFAULT_PROFILE_REGISTRY_PATH, registry)
    return root


def _application_evidence(root: Path) -> dict[str, object]:
    evidence_path = root / "evidence" / "application-test.txt"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text("reference replay passed\n", encoding="utf-8")
    return _sealed_record(
        root,
        evidence_path,
        record_id="application-evidence",
        role="TEST",
        title="Bounded application replay",
    )


def _application_fields(root: Path) -> dict[str, object]:
    return {
        "mappings": _mappings(
            "application",
            evidence_id="application-evidence",
            disposition="IMPLEMENTED",
        ),
        "resource_bounds": _bounds("application"),
        "failure_modes": _failures("application"),
        "evidence": [_application_evidence(root)],
    }


def _extension_evidence(root: Path) -> dict[str, object]:
    evidence_path = root / "evidence" / "proposal-analysis.txt"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text("candidate remains quarantined\n", encoding="utf-8")
    return _sealed_record(
        root,
        evidence_path,
        record_id="proposal-evidence",
        role="ANALYSIS",
        title="Candidate mapping analysis",
    )


def test_loads_verified_kernel_and_exact_bounded_profile_registry(tmp_path: Path) -> None:
    root = _contract_repository(tmp_path)

    bundle = contracts.load_contract_bundle(root)

    assert bundle.kernel["kernel_id"] == contracts.KERNEL_ID
    assert set(bundle.profiles) == contracts.REQUIRED_PROFILE_IDS
    assert set(bundle.profile_sha256s) == contracts.REQUIRED_PROFILE_IDS
    result = contracts.validate_kernel_contract(bundle.kernel, repository_root=root)
    assert result["ok"] is True
    assert result["transition_chain"] == list(contracts.CANONICAL_TRANSITION_CHAIN)
    assert result["mapping_count"] == len(contracts.MAPPING_CATEGORIES)


def test_committed_source_hash_and_path_escape_fail_closed(tmp_path: Path) -> None:
    root = _contract_repository(tmp_path)
    kernel_path = root / contracts.DEFAULT_KERNEL_PATH
    kernel = json.loads(kernel_path.read_text(encoding="utf-8"))
    source_path = root / "sources" / "ugts-foundation.txt"
    source_path.write_bytes(b"x" * source_path.stat().st_size)

    with pytest.raises(contracts.SubstrateContractError, match="SHA-256 mismatch"):
        contracts.validate_kernel_contract(kernel, repository_root=root)

    kernel["source_records"][0]["path"] = "../outside.txt"
    with pytest.raises(contracts.SubstrateContractError, match="repository-relative"):
        contracts.validate_kernel_contract(kernel, repository_root=root, verify_sources=False)


def test_external_provenance_is_registered_but_never_treated_as_redistributed(
    tmp_path: Path,
) -> None:
    root = _contract_repository(tmp_path)
    kernel = json.loads((root / contracts.DEFAULT_KERNEL_PATH).read_text(encoding="utf-8"))
    kernel["source_records"].append(
        {
            "source_id": "external-reference",
            "class": "EARLY_EXECUTABLE_ALGEBRA",
            "availability": "READ_ONLY_EXTERNAL",
            "sha256": "1" * 64,
            "redistributed": False,
        }
    )

    result = contracts.validate_kernel_contract(kernel, repository_root=root)
    assert result["external_sources_registered"] == 1

    kernel["source_records"][-1]["redistributed"] = True
    with pytest.raises(contracts.SubstrateContractError, match="must not claim redistribution"):
        contracts.validate_kernel_contract(kernel, repository_root=root)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda kernel: kernel["canonical_chain"].reverse(), "transition/handoff"),
        (
            lambda kernel: kernel["policy"].__setitem__("automatic_extension_promotion", True),
            "automatic_extension_promotion",
        ),
        (
            lambda kernel: kernel["identity"].__setitem__("not_a_renderer", False),
            "not_a_renderer",
        ),
    ],
)
def test_kernel_architecture_invariants_fail_closed(tmp_path: Path, mutation, match: str) -> None:
    root = _contract_repository(tmp_path)
    kernel = json.loads((root / contracts.DEFAULT_KERNEL_PATH).read_text(encoding="utf-8"))
    mutation(kernel)

    with pytest.raises(contracts.SubstrateContractError, match=match):
        contracts.validate_kernel_contract(kernel, repository_root=root)


def test_registry_rejects_profile_file_tampering(tmp_path: Path) -> None:
    root = _contract_repository(tmp_path)
    profile_path = root / "substrate" / "profiles" / "nhdf-v0.3-ccd.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["title"] = "tampered title"
    _write_json(profile_path, profile)

    with pytest.raises(contracts.SubstrateContractError, match="does not match the referenced profile"):
        contracts.load_contract_bundle(root)


def test_application_manifest_binds_profiles_mappings_evidence_and_self_reference(
    tmp_path: Path,
) -> None:
    root = _contract_repository(tmp_path)
    fields = _application_fields(root)
    self_reference = {
        "enabled": True,
        "bounded_generations": 2,
        "may_propose_extensions": True,
        "may_promote_extensions": False,
        "proposal_disposition": "QUARANTINED",
    }

    manifest = contracts.create_application_manifest(
        "orbit-event-demo",
        "0.1.0",
        repository_root=root,
        selected_profiles=("nhdf-v0.1", "sclp-foundational"),
        profile_selection_rationale="Both profiles map the bounded event application.",
        self_reference=self_reference,
        **fields,
    )
    result = contracts.validate_application_manifest(manifest, repository_root=root)

    assert result["ok"] is True
    assert result["selected_profiles"] == ["nhdf-v0.1", "sclp-foundational"]
    assert result["mapping_count"] == len(contracts.MAPPING_CATEGORIES)
    assert result["self_reference_enabled"] is True
    assert manifest["self_reference"]["may_promote_extensions"] is False


def test_application_rejects_unbound_profile_missing_mapping_and_stale_evidence(
    tmp_path: Path,
) -> None:
    root = _contract_repository(tmp_path)
    fields = _application_fields(root)
    manifest = contracts.create_application_manifest(
        "bounded-demo",
        "1",
        repository_root=root,
        selected_profiles=("nhdf-v0.1",),
        profile_selection_rationale="The NHDF v0.1 mapping is used explicitly.",
        **fields,
    )

    bad_digest = copy.deepcopy(manifest)
    bad_digest["profiles"][0]["sha256"] = "0" * 64
    with pytest.raises(contracts.SubstrateContractError, match="does not bind"):
        contracts.validate_application_manifest(bad_digest, repository_root=root)

    missing_mapping = copy.deepcopy(manifest)
    del missing_mapping["mappings"]["packing"]
    with pytest.raises(contracts.SubstrateContractError, match="missing required keys"):
        contracts.validate_application_manifest(missing_mapping, repository_root=root)

    (root / "evidence" / "application-test.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(contracts.SubstrateContractError, match="mismatch"):
        contracts.validate_application_manifest(manifest, repository_root=root)


def test_kernel_only_application_requires_explicit_rationale(tmp_path: Path) -> None:
    root = _contract_repository(tmp_path)
    fields = _application_fields(root)

    with pytest.raises(contracts.SubstrateContractError, match="kernel-only"):
        contracts.create_application_manifest(
            "kernel-demo",
            "1",
            repository_root=root,
            profile_selection_rationale="No optional profile was selected.",
            **fields,
        )

    manifest = contracts.create_application_manifest(
        "kernel-demo",
        "1",
        repository_root=root,
        profile_selection_rationale="This is deliberately a kernel-only application.",
        **fields,
    )
    assert manifest["profiles"] == []


def test_provisional_profile_is_verified_but_cannot_be_selected(tmp_path: Path) -> None:
    root = _contract_repository(tmp_path, provisional_sclp=True)
    bundle = contracts.load_contract_bundle(root)
    assert "sclp-foundational" in bundle.profiles
    fields = _application_fields(root)

    with pytest.raises(contracts.SubstrateContractError, match="provisional"):
        contracts.create_application_manifest(
            "premature-sclp-app",
            "1",
            repository_root=root,
            selected_profiles=("sclp-foundational",),
            profile_selection_rationale="Attempt to select unaudited provenance.",
            **fields,
        )


def test_self_referential_extension_stays_quarantined_and_cannot_promote(
    tmp_path: Path,
) -> None:
    root = _contract_repository(tmp_path)
    evidence = _extension_evidence(root)
    self_reference = {
        "enabled": True,
        "bounded_generations": 1,
        "may_propose_extensions": True,
        "may_promote_extensions": False,
        "proposal_disposition": "QUARANTINED",
    }
    proposal = contracts.create_extension_proposal(
        "candidate-relation-map",
        "0.1",
        repository_root=root,
        origin_kind="SELF_REFERENCE",
        origin_description="A bounded replay produced a candidate relation mapping.",
        origin_source_refs=("proposal-evidence",),
        admission_forms=("RELATION_SURFACE", "TRANSITION_ROUTING_MAP"),
        canonical_chain_effect="UNCHANGED",
        admission_justification="The candidate maps typed state without changing handoff order.",
        mappings=_mappings(
            "proposal",
            evidence_id="proposal-evidence",
            disposition="PROPOSED",
        ),
        resource_bounds=_bounds("proposal"),
        failure_modes=_failures("proposal"),
        evidence=(evidence,),
        target_profiles=("nhdf-v0.1",),
        self_reference=self_reference,
    )

    result = contracts.validate_extension_proposal(proposal, repository_root=root)
    assert result["status"] == "QUARANTINED"
    assert result["automatic_promotion"] is False
    assert result["promoted"] is False

    promoted = copy.deepcopy(proposal)
    promoted["promotion"]["promoted"] = True
    with pytest.raises(contracts.SubstrateContractError, match="cannot be promoted"):
        contracts.validate_extension_proposal(promoted, repository_root=root)

    self_promoting = copy.deepcopy(proposal)
    self_promoting["self_reference"]["may_promote_extensions"] = True
    with pytest.raises(contracts.SubstrateContractError, match="cannot promote"):
        contracts.validate_extension_proposal(self_promoting, repository_root=root)


def test_unknown_contract_fields_require_a_format_bump(tmp_path: Path) -> None:
    root = _contract_repository(tmp_path)
    kernel = json.loads((root / contracts.DEFAULT_KERNEL_PATH).read_text(encoding="utf-8"))
    kernel["entertainment_features"] = ["unbounded-content-pack"]

    with pytest.raises(contracts.SubstrateContractError, match="unknown keys"):
        contracts.validate_kernel_contract(kernel, repository_root=root)
