"""
Fix SCRFD ONNX outputs cho DeepStream/TensorRT.

Vấn đề: SCRFD ONNX gốc (det_500m.onnx từ buffalo_sc) flatten output thành
2-D [N, K] với N = batch * anchors. TRT/DeepStream xem N là batch dim →
inferDims chỉ còn [K] = [1]/[4]/[10] → parser không suy được anchors → 0
detections.

Fix: thêm Reshape sau mỗi output thành [-1, anchors, K]. Giữ batch dynamic.
"""

import argparse
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--input-name", default="input.1")
    ap.add_argument("--input-h", type=int, default=640)
    ap.add_argument("--input-w", type=int, default=640)
    args = ap.parse_args()

    # 1) Suy anchors per output bằng inference batch=1.
    sess = ort.InferenceSession(args.src,
                                providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, 3, args.input_h, args.input_w), dtype=np.float32)
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(None, {args.input_name: dummy})
    out_info = {n: (o.shape[0], o.shape[1]) for n, o in zip(out_names, outs)}
    for n, (a, k) in out_info.items():
        print(f"  {n}: anchors={a} inner={k}")

    # 2) Load ONNX, append Reshape per output.
    m = onnx.load(args.src)
    new_outputs = []
    for out in list(m.graph.output):
        name = out.name
        if name not in out_info:
            new_outputs.append(out)
            continue
        anchors, inner = out_info[name]

        shape_init_name = f"{name}_target_shape"
        shape_init = numpy_helper.from_array(
            np.array([-1, anchors, inner], dtype=np.int64),
            name=shape_init_name)
        m.graph.initializer.append(shape_init)

        new_name = f"{name}_3d"
        node = helper.make_node(
            "Reshape",
            inputs=[name, shape_init_name],
            outputs=[new_name],
            name=f"reshape_{name}_3d")
        m.graph.node.append(node)

        new_out = helper.make_tensor_value_info(
            new_name, TensorProto.FLOAT, ["batch", anchors, inner])
        new_outputs.append(new_out)

    m.graph.ClearField("output")
    for o in new_outputs:
        m.graph.output.append(o)

    onnx.checker.check_model(m)
    onnx.save(m, args.dst)
    print(f"\nSaved: {args.dst}")

    # 3) Verify by inference.
    sess2 = ort.InferenceSession(args.dst,
                                 providers=["CPUExecutionProvider"])
    outs2 = sess2.run(None, {args.input_name: dummy})
    print("\nNew output shapes:")
    for o, info in zip(sess2.get_outputs(), outs2):
        print(f"  {o.name}: {info.shape}")


if __name__ == "__main__":
    main()
