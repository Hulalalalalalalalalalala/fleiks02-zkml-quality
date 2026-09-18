"""Reproduce the small example network; these are illustrative fixed weights."""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

root = Path(__file__).resolve().parent.parent
w1 = np.array([[1, 0, 0.5, 0], [0, 1, 0, 0.5], [0.5, 0, 1, 0],
               [0, 0.5, 0, 1], [0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]], dtype=np.float32)
w2 = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]], dtype=np.float32)
initializers = [numpy_helper.from_array(value, name) for name, value in
                (("w1", w1), ("b1", np.full(4, -0.125, dtype=np.float32)),
                 ("w2", w2), ("b2", np.array([1.0, 0.0], dtype=np.float32)))]
nodes = [helper.make_node("MatMul", ["features", "w1"], ["h0"]),
         helper.make_node("Add", ["h0", "b1"], ["h1"]),
         helper.make_node("Relu", ["h1"], ["h2"]),
         helper.make_node("MatMul", ["h2", "w2"], ["o0"]),
         helper.make_node("Add", ["o0", "b2"], ["scores"])]
graph = helper.make_graph(nodes, "quality-example", [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 6])],
                          [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2])], initializers)
model = helper.make_model(graph, producer_name="quality-example", opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
destination = root / "models" / "quality.onnx"
destination.parent.mkdir(exist_ok=True)
onnx.save(model, destination)
print(destination)
