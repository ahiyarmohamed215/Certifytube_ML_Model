import json
from pathlib import Path
from typing import Dict, List

CONTRACTS_DIR = Path("ml/contracts")


class ContractError(Exception):
    pass


def load_contract(version: str) -> List[str]:
    if version != "v1.0":
        raise ContractError(f"Unsupported feature_version '{version}'. Expected 'v1.0'.")

    path = CONTRACTS_DIR / "feature_contract_v1.json"
    if not path.exists():
        raise ContractError("Feature contract file missing: feature_contract_v1.json")

    with open(path, "r") as f:
        contract = json.load(f)

    if contract.get("feature_version") != "v1.0":
        raise ContractError("Contract version mismatch inside contract file.")

    features = contract.get("features")
    if not isinstance(features, list) or not features:
        raise ContractError("Contract file must contain a non-empty 'features' list.")

    return features


def validate_payload(features: Dict[str, float], expected: List[str]) -> None:
    missing = [c for c in expected if c not in features]
    extra = [k for k in features.keys() if k not in expected]

    if missing:
        raise ContractError(f"Missing features (contract mismatch): {missing}")

    if extra:
        raise ContractError(f"Unexpected features (contract mismatch): {extra}")

    for k, v in features.items():
        if not isinstance(v, (int, float)):
            raise ContractError(f"Feature '{k}' must be numeric")
