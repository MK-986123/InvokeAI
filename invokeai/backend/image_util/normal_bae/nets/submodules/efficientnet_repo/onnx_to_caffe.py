import argparse
from pathlib import Path

import onnx
from caffe2.python.onnx.backend import Caffe2Backend


parser = argparse.ArgumentParser(description="Convert ONNX to Caffe2")

parser.add_argument("model", help="The ONNX model")
parser.add_argument("--c2-prefix", required=True,
    help="The output file prefix for the caffe2 model init and predict file. ")


def _assert_outputs_under_cwd(c2_prefix: str) -> None:
    base = Path.cwd().resolve()
    for suffix in (".init.pb", ".predict.pb"):
        out = Path(f"{c2_prefix}{suffix}").resolve()
        try:
            out.relative_to(base)
        except ValueError as e:
            raise ValueError(
                "Invalid path: output must remain under the current working directory."
            ) from e


def main():
    args = parser.parse_args()
    _assert_outputs_under_cwd(args.c2_prefix)

    onnx_model = onnx.load(args.model)
    caffe2_init, caffe2_predict = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)

    caffe2_init_str = caffe2_init.SerializeToString()
    with open(args.c2_prefix + '.init.pb', "wb") as f:
        f.write(caffe2_init_str)
    caffe2_predict_str = caffe2_predict.SerializeToString()
    with open(args.c2_prefix + '.predict.pb', "wb") as f:
        f.write(caffe2_predict_str)


if __name__ == "__main__":
    main()
