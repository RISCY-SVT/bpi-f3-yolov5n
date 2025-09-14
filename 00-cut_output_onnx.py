#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script modifies an ONNX model to include specific output nodes.
# It loads the model, adds specified output nodes, and saves the modified model.

import onnx, sys, onnx.helper as h		
m = onnx.load('yolov5/yolov5n.onnx')		
g = m.graph
wanted = ['/model.24/m.0/Conv_output_0',		
'/model.24/m.1/Conv_output_0',		
'/model.24/m.2/Conv_output_0']		

for name in wanted:		
    vi = h.ValueInfoProto() ; vi.name = name ; g.output.extend([vi])
    
onnx.save(m, 'yolov5n_out3.onnx')
	