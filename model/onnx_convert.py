import torch
from torch import Tensor
import onnx
import model
import model_multi
from onnx import checker
import os
PATH_TO_ESP_DL = input("Enter the path to the esp-dl repository: ")
os.environ['ESP_DL_PATH'] = PATH_TO_ESP_DL
from onnxruntime.quantization.preprocess import quant_pre_process
from esp_quantize_onnx import main as quantize
from export_onnx_model import dump_tvm_onnx_project

script_dir = os.path.dirname(__file__)
relative_path = "CSImodel_snr14_24_sir1_15.pth"
file_path = os.path.join(script_dir, relative_path)

CSImodel = model_multi.CNN()
CSImodel.load_state_dict(torch.load(file_path))
onnx_file_path = os.path.join(script_dir, 'CSImodel.onnx')
t = Tensor(1, 2, 242)
torch.onnx.export(CSImodel, t, onnx_file_path, verbose = True, input_names=['input'], output_names=['output'])

model_proto = onnx.load_model(onnx_file_path)
checker.check_graph(model_proto.graph)

opt_file_path = os.path.join(script_dir, 'CSImodel_opt.onnx')
quant_pre_process(onnx_file_path, opt_file_path)

dump_tvm_onnx_project('esp32', opt_file_path, os.path.join(script_dir, '../data/train_multi_csi_sample.npy'), PATH_TO_ESP_DL+'/tools/tvm/template_project_for_model', PATH_TO_ESP_DL+'/examples')