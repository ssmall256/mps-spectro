from pathlib import Path

import torch.utils.cpp_extension

_PROJECT_DIR = Path(__file__).resolve().parent
_NATIVE_ROOT = Path("/tmp/codex_torch_mps_istft")
_NATIVE_ROOT.mkdir(parents=True, exist_ok=True)

_SRC_REAL = (_PROJECT_DIR / "metal" / "custom_mps_istft.mm").resolve()
_HDR_REAL = (_PROJECT_DIR / "metal" / "custom_mps_istft_kernel.h").resolve()
_SRC = _NATIVE_ROOT / "custom_mps_istft.mm"
_HDR = _NATIVE_ROOT / "custom_mps_istft_kernel.h"

for link, target in ((_SRC, _SRC_REAL), (_HDR, _HDR_REAL)):
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
    except FileExistsError:
        # Concurrent imports can race on symlink creation; reuse if target matches.
        if link.is_symlink() and link.resolve() == target:
            pass
        else:
            link.unlink(missing_ok=True)
            link.symlink_to(target)

_BUILD_DIR = _NATIVE_ROOT / "build"
_BUILD_DIR.mkdir(parents=True, exist_ok=True)

# Work around cpp_extension linker issues when torch lives under a path with spaces.
# We patch TORCH_LIB_PATH to a space-free symlink path.
torch_lib_path = Path(torch.utils.cpp_extension.TORCH_LIB_PATH).resolve()
if " " in str(torch_lib_path):
    alias_dir = _NATIVE_ROOT / "torch_lib_alias"
    try:
        if alias_dir.exists() or alias_dir.is_symlink():
            alias_dir.unlink()
        alias_dir.symlink_to(torch_lib_path, target_is_directory=True)
    except FileExistsError:
        if not (alias_dir.is_symlink() and alias_dir.resolve() == torch_lib_path):
            alias_dir.unlink(missing_ok=True)
            alias_dir.symlink_to(torch_lib_path, target_is_directory=True)
    torch.utils.cpp_extension.TORCH_LIB_PATH = str(alias_dir)

compiled_lib = torch.utils.cpp_extension.load(
    name="CustomMPSISTFT",
    sources=[str(_SRC)],
    extra_cflags=["-std=c++17"],
    build_directory=str(_BUILD_DIR),
)
