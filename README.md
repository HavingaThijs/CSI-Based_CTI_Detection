# CSI-Based Cross-Technology Interference Detection and Classification on Wi-Fi 6 IoT Hardware

This repository contains a Convolutional Neural Network (CNN) trained to identify Cross-Technology Interference (CTI) based on the Channel State Information (CSI) obtained from Wi-Fi 6 devices, in particular the ESP32-C6.

There are two different folders:
- `esp32c6` contains the ESP-IDF project to run it on an ESP32-C6 device, which is partly automatically generated and partly based on the iPerf example.
- `model` contains the Python scripts and outputs to train and test the model using PyTorch.

The trained model is available as a serialized PyTorch state dictionary (`model/CSImodel_snr14_24_sir1_15.pth`) and as optimized model in ONNX format (`model/CSImodel_opt.onnx`).

Follow [this guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/linux-macos-setup.html) to build the ESP-IDF project and flash it to an ESP32-C6 device.

For access to the prepared data, please contact me at Thijs&#46;Havinga&#64;UGent&#46;be.
If you want to retrain and deploy the model yourself, execute `model/train.py` (which may take several hours). Then follow [this guide](https://docs.espressif.com/projects/esp-dl/en/latest/esp32/tutorials/deploying-models-through-tvm.html) to download the ESP-DL library first, and then use `model/onnx_convert.py` to generate the ESP-IDF project.