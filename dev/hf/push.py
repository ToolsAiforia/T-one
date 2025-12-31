import argparse
import shutil
from pathlib import Path

from huggingface_hub import login, upload_folder

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_ID = "AiphoriaTech/T-one"
MODEL_NAME = "model.nemo"
LEXICON_NAME = "kenlm.flashlight_lexicon"
VAD_ONNX_RELATIVE_PATH = Path("onnx/silero_vad.onnx")
TENSORRT_SUBDIR = Path("TensorRT/25.04/l40s")
DEFAULT_TRITON_MODEL_DIR = REPO_ROOT / "triton" / "model" / "1"
DEFAULT_EXPORT_PATH_TO_PRETRAINED = "t-tech/T-one"
DEFAULT_EXPORT_CHUNK_DURATION_MS = 400
DEFAULT_RESOURCES_DIR = Path(__file__).resolve().parent / "resources"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload T-one artifacts to Hugging Face.")
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face repo ID to upload to.",
    )
    parser.add_argument(
        "--resources-dir",
        type=Path,
        default=DEFAULT_RESOURCES_DIR,
        help="Directory with artifacts to upload.",
    )
    parser.add_argument(
        "--nemo-path",
        type=Path,
        default=MODEL_NAME,
        help="Path to an existing .nemo file to upload as model.nemo.",
    )
    parser.add_argument(
        "--triton-model-dir",
        type=Path,
        default=DEFAULT_TRITON_MODEL_DIR,
        help="Path to Triton model directory with kenlm.bin and model.plan.",
    )
    parser.add_argument(
        "--path-to-pretrained",
        type=str,
        default=DEFAULT_EXPORT_PATH_TO_PRETRAINED,
        help="Path to pretrained checkpoint for NeMo export.",
    )
    parser.add_argument(
        "--chunk-duration-ms",
        type=int,
        default=DEFAULT_EXPORT_CHUNK_DURATION_MS,
        help="Chunk duration in ms for export (aligned with scripts/onnx_build.sh).",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing resources when collecting artifacts (default: true).",
    )
    parser.add_argument(
        "--skip-preprocessor",
        action="store_true",
        default=True,
        help="Export model expecting log-mel features instead of raw audio (default: true).",
    )
    parser.add_argument(
        "--no-skip-preprocessor",
        dest="skip_preprocessor",
        action="store_false",
        help="Export model expecting raw audio instead of log-mel features.",
    )
    return parser.parse_args()


def _copy_resource(source: Path, target: Path, label: str, force: bool) -> None:
    if target.exists() and not force:
        return
    if not source.exists():
        raise FileNotFoundError(f"Missing {label} source at: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)


def _export_nemo(output_path: Path, args: argparse.Namespace) -> None:
    from tone.scripts.export import _export_nemo as export_nemo

    export_nemo(args.path_to_pretrained, output_path)


def _ensure_model_nemo(resources_dir: Path, nemo_path: Path | None, args: argparse.Namespace) -> Path:
    resources_dir = resources_dir.expanduser()
    resources_dir.mkdir(parents=True, exist_ok=True)
    target = resources_dir / MODEL_NAME
    if target.exists() and not args.force:
        return target

    if nemo_path is not None:
        nemo_path = nemo_path.expanduser()
        if nemo_path.exists():
            if nemo_path.resolve() != target.resolve():
                shutil.copy2(nemo_path, target)
            return target

    _export_nemo(target, args)
    return target


def _ensure_lexicon(resources_dir: Path, force: bool) -> Path:
    target = resources_dir / LEXICON_NAME
    if target.exists():
        return target

    raise ValueError(f"{target} not found")


def _ensure_vad_onnx(resources_dir: Path) -> Path:
    target = resources_dir / VAD_ONNX_RELATIVE_PATH
    if not target.exists():
        raise FileNotFoundError(f"Missing required VAD ONNX at: {target}")
    return target


def _collect_resources(resources_dir: Path, args: argparse.Namespace) -> None:
    resources_dir = resources_dir.expanduser()
    resources_dir.mkdir(parents=True, exist_ok=True)

    _ensure_model_nemo(resources_dir, args.nemo_path, args)
    _copy_resource(
        args.triton_model_dir.expanduser() / "kenlm.bin",
        resources_dir / "kenlm.bin",
        "kenlm.bin",
        args.force,
    )
    _copy_resource(
        args.triton_model_dir.expanduser() / "model.plan",
        resources_dir / TENSORRT_SUBDIR / "model.plan",
        "TensorRT model.plan",
        args.force,
    )
    _ensure_lexicon(resources_dir, args.force)
    _ensure_vad_onnx(resources_dir)


def main() -> None:
    args = _parse_args()
    resources_dir = args.resources_dir.expanduser()
    if args.nemo_path is None:
        args.nemo_path = resources_dir / MODEL_NAME
    _collect_resources(resources_dir, args)

    # Push your model files
    upload_folder(
        folder_path=str(resources_dir),
        repo_id=args.repo_id,
        repo_type="model",
    )


if __name__ == "__main__":
    main()
