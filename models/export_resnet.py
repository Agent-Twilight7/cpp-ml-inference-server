"""Export torchvision ResNet-18 to ONNX with a dynamic batch dimension.

Also writes a fixed reference batch (input + PyTorch output) as raw float32
files so the C++ side can check it computes the same thing.

Usage:  python models/export_resnet.py [--out-dir models]
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

REF_BATCH = 8
SEED = 1234


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "resnet18.onnx"

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()

    example = torch.randn(1, 3, 224, 224)
    batch = torch.export.Dim("batch", min=1, max=1024)
    onnx_program = torch.onnx.export(
        model,
        (example,),
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: batch}},
        dynamo=True,
    )
    onnx_program.save(str(model_path), external_data=False)
    onnx.checker.check_model(str(model_path))
    print(f"wrote {model_path}")

    # Reference batch: deterministic random input, PyTorch output.
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((REF_BATCH, 3, 224, 224), dtype=np.float32)
    with torch.no_grad():
        y_torch = model(torch.from_numpy(x)).numpy()

    # Sanity check with Python onnxruntime before the C++ side ever runs.
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    y_ort = sess.run(None, {"input": x})[0]
    max_diff = float(np.abs(y_ort - y_torch).max())
    print(f"python onnxruntime vs pytorch: max abs diff = {max_diff:.3e}")
    assert max_diff < 1e-3, "ONNX export does not match PyTorch"

    x.tofile(out_dir / "ref_input.bin")
    y_torch.astype(np.float32).tofile(out_dir / "ref_output.bin")
    print(f"wrote reference batch of {REF_BATCH} to {out_dir}/ref_input.bin, ref_output.bin")


if __name__ == "__main__":
    main()
