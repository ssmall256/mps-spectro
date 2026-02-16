import json
from pathlib import Path

import pytest
import torch

from mps_spectro.istft import mps_istft_forward


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "istft_golden_fixtures.json"


def _assert_mps() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required")


def _load_cases() -> list[dict]:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    if not cases:
        raise RuntimeError(f"No golden cases found in {FIXTURE_PATH}")
    return cases


CASES = _load_cases()
CASE_IDS = [c["id"] for c in CASES]


def _case_by_id(case_id: str) -> dict:
    for case in CASES:
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)


@pytest.mark.parametrize("kernel_layout", ["native", "transposed", "auto"])
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_golden_fixture_output(case_id: str, kernel_layout: str) -> None:
    _assert_mps()
    case = _case_by_id(case_id)

    spec_real = torch.tensor(case["spec_real"], dtype=torch.float32, device="mps")
    spec_imag = torch.tensor(case["spec_imag"], dtype=torch.float32, device="mps")
    spec = torch.complex(spec_real, spec_imag)
    window = torch.tensor(case["window"]["values"], dtype=torch.float32, device="mps")
    expected = torch.tensor(case["expected_output"], dtype=torch.float32, device="cpu")

    out = mps_istft_forward(
        spec,
        n_fft=int(case["n_fft"]),
        hop_length=int(case["hop_length"]),
        win_length=int(case["win_length"]),
        window=window,
        center=bool(case["center"]),
        normalized=bool(case["normalized"]),
        onesided=bool(case["onesided"]),
        length=case["length"],
        allow_fused=True,
        kernel_dtype="float32",
        kernel_layout=kernel_layout,
        long_mode_strategy="custom",
    ).to("cpu", dtype=torch.float32)

    max_abs = (out - expected).abs().max().item()
    mse = ((out - expected) ** 2).mean().item()
    assert max_abs <= 1.0e-5
    assert mse <= 1.0e-7
