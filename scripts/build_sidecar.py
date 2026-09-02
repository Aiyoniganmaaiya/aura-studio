"""Build Aura Studio engine as a standalone executable via PyInstaller.

Usage:
    python scripts/build_sidecar.py

This creates a single executable `aura-engine.exe` in src-tauri/binaries/
that Tauri bundles as a sidecar for production builds.
"""

import os
import shutil
import subprocess
import sys
import platform
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    backend_dir = project_root / "backend"
    target_dir = project_root / "src-tauri" / "binaries"

    os.makedirs(target_dir, exist_ok=True)

    # Detect target triple
    machine = platform.machine().lower()
    if machine == "amd64" or machine == "x86_64":
        arch = "x86_64"
    elif machine == "arm64" or machine == "aarch64":
        arch = "aarch64"
    else:
        arch = "x86_64"

    system = platform.system().lower()
    if system == "windows":
        triple = f"{arch}-pc-windows-msvc"
        ext = ".exe"
    elif system == "darwin":
        triple = f"{arch}-apple-darwin"
        ext = ""
    else:
        triple = f"{arch}-unknown-linux-gnu"
        ext = ""

    output_name = f"aura-engine-{triple}{ext}"
    output_path = target_dir / output_name

    print(f"[build_sidecar] Target: {output_name}")
    print(f"[build_sidecar] Backend: {backend_dir}")
    print(f"[build_sidecar] Output: {output_path}")

    # Install PyInstaller if not available
    try:
        import PyInstaller  # noqa
    except ImportError:
        print("[build_sidecar] Installing PyInstaller...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyinstaller"],
            check=True,
            cwd=project_root,
        )

    # Build the executable
    print("[build_sidecar] Running PyInstaller...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--onefile",
            "--name", output_path.stem,
            "--distpath", str(target_dir),
            "--workpath", str(project_root / "build" / "sidecar-build"),
            "--specpath", str(project_root / "build"),
            "--add-data", f"{backend_dir / 'engines'}{os.pathsep}engines",
            "--add-data", f"{backend_dir / 'api'}{os.pathsep}api",
            "--hidden-import", "diffusers",
            "--hidden-import", "transformers",
            "--hidden-import", "torch",
            "--hidden-import", "accelerate",
            "--hidden-import", "huggingface_hub",
            "--hidden-import", "uvicorn",
            "--hidden-import", "fastapi",
            "--hidden-import", "websockets",
            "--hidden-import", "pydantic",
            "--hidden-import", "PIL",
            "--hidden-import", "safetensors",
            "--collect-all", "diffusers",
            "--collect-all", "transformers",
            str(backend_dir / "main.py"),
        ],
        check=True,
        cwd=project_root,
    )

    # Rename to include triple
    built_path = target_dir / f"{output_path.stem}{ext}"
    if built_path.exists():
        if output_path.exists():
            os.remove(output_path)
        shutil.move(str(built_path), str(output_path))
        print(f"[build_sidecar] ✅ Sidecar built: {output_path}")
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[build_sidecar] Size: {size_mb:.1f} MB")
    else:
        print(f"[build_sidecar] ❌ Build failed — output not found at {built_path}")
        sys.exit(1)

    # Clean up build artifacts
    build_dir = project_root / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
        print("[build_sidecar] Build artifacts cleaned")


if __name__ == "__main__":
    main()
