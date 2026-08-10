"""Make CTranslate2's SYCL runtime win the DLL name race on Windows.

PyTorch's XPU build ships its own copy of the Intel runtime, and Windows binds
imports by base name to whichever module is already loaded. If torch is imported
first - whisperx/__main__.py does exactly that - then ctranslate2.dll binds to
torch's sycl8.dll. When that copy is older than the one CTranslate2 was built
against it lacks some of the SYCL runtime's internal entry points, and the load
fails with ERROR_PROC_NOT_FOUND (WinError 127).

Loading our copies here, before torch gets a chance to, registers those names
first. The direction is safe: a newer SYCL runtime exports a superset, so the one
copy serves both CTranslate2 and torch. It is not safe the other way round, which
is why this cannot be left to import order.

The whole bundled set is loaded, not just sycl8.dll. Checking exported symbols
suggests the other Intel libraries are interchangeable with the copies torch
ships, but they are not: the SYCL runtime, its Unified Runtime loader and adapters,
and oneMKL are coupled at runtime, and a process that mixes 2025.1 and 2025.3
components crashes partway through a model instead of failing to load.

To take effect this module has to be imported before torch, i.e. at interpreter
startup. Drop a one-line file next to it in site-packages:

    ctranslate2_preload.pth   containing   import ctranslate2_preload

Startup is only where the *hook* is installed - the libraries themselves are not
loaded until something actually imports torch or ctranslate2. Loading them eagerly
would break pip: every interpreter in the environment would hold the bundled DLLs
mapped, pip included, and Windows refuses to overwrite a mapped image, so
`pip install` of this very package fails with ERROR_ACCESS_DENIED on whichever DLL
it reaches first.
"""

import os
import sys

# Importing any of these means the Intel runtime is about to be needed.
_TRIGGERS = frozenset(("torch", "ctranslate2"))


def _load_runtime():
    import ctypes
    import glob

    package_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ctranslate2")
    if not os.path.isfile(os.path.join(package_dir, "sycl8.dll")):
        return  # A build without the SYCL backend, or the runtime is not bundled.

    try:
        os.add_dll_directory(package_dir)
    except OSError:
        pass
    # The Unified Runtime loader chain resolves through the legacy search order,
    # which add_dll_directory() does not cover.
    os.environ["PATH"] = package_dir + os.pathsep + os.environ.get("PATH", "")

    # ctranslate2.dll last: it is the one with the dependencies, and loading it first
    # would resolve them against whatever else is already registered.
    dlls = sorted(
        glob.glob(os.path.join(package_dir, "*.dll")),
        key=lambda p: os.path.basename(p).lower() == "ctranslate2.dll",
    )
    # A few passes, so a library whose dependency comes later alphabetically still
    # gets loaded from this directory rather than from elsewhere on the search path.
    for _ in range(3):
        pending = []
        for dll in dlls:
            try:
                ctypes.CDLL(dll)
            except OSError:
                pending.append(dll)
        if not pending or pending == dlls:
            break
        dlls = pending


class _RuntimeFinder:
    """Loads the bundled runtime on the first import of torch or ctranslate2."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in _TRIGGERS:
            # Before loading, so a failure in there cannot make us fire twice.
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            try:
                _load_runtime()
            except Exception:
                pass  # Leave it to ctranslate2/__init__.py to report a usable error.
        return None  # Never claim the module; the real finders handle the import.


if sys.platform == "win32":
    sys.meta_path.insert(0, _RuntimeFinder())
