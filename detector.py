import itertools
import math
import numpy as np
import os
import cv2
from collections import namedtuple


class Detector(object):
    def __init__(self, ie, model_path, device='CPU', threshold=0.5):
        model = ie.read_network(
            model=model_path, weights=os.path.splitext(model_path)[0] + '.bin')

        # assert len(model.input_info) == 1, "Expected 1 input blob"
        # assert len(model.outputs) == 2, "Expected 2 output blobs"
        print(len(model.input_info))
        print(len(model.outputs))
        self._input_layer_name = next(iter(model.input_info))
        self._output_layer_names = sorted(model.outputs)

        # assert model.outputs[self._output_layer_names[0]].shape[1] == \
        #     model.outputs[self._output_layer_names[1]
        #                   ].shape[1], "Expected the same dimension for boxes and scores"
        # assert model.outputs[self._output_layer_names[0]
        #                      ].shape[2] == 4, "Expected 4-coordinate boxes"
        # assert model.outputs[self._output_layer_names[1]
        #                      ].shape[2] == 2, "Expected 2-class scores(background, face)"

        self._ie = ie
        self._exec_model = self._ie.load_network(model, device)
        self.infer_time = -1
        _, channels, self.input_height, self.input_width = model.input_info[
            self._input_layer_name].input_data.shape
        assert channels == 3, "Expected 3-channel input"
        print(model.input_info[self._input_layer_name].input_data.shape)
        print('model shape: ' +
              str(model.outputs[self._output_layer_names[0]].shape[1]))

        self.min_sizes = [[32, 64, 128], [256], [512]]
        self.steps = [32, 64, 128]
        self.variance = [0.1, 0.2]
        self.confidence_threshold = threshold
        self.nms_threshold = 0.3
        self.keep_top_k = 750

    def preprocess(self, image):
        return cv2.resize(image, (self.input_width, self.input_height))

    def infer(self, image):
        t0 = cv2.getTickCount()
        inputs = {self._input_layer_name: image}
        output = self._exec_model.infer(inputs=inputs)
        self.infer_time = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        return output

    def postprocess(self, raw_output, image_sizes):
        # boxes, scores = raw_output
        image_id, label, conf, x_min, y_min, x_max, y_max = raw_output[0]
        # print(boxes)
        # detection = namedtuple(
        #    'image_id','label','conf' , 'score, x_min, y_min, x_max, y_max')
        detection = namedtuple(
            'detection', 'image_id label conf x_min y_min x_max y_max')
        detections = []
        image_info = [self.input_height, self.input_width]

    def detect(self, image):
        print(image.shape)
        print(image.shape[:2])
        image_sizes = image.shape[:2]
        image = self.preprocess(image)
        print('---\n')
        image = np.transpose(image, (2, 0, 1))
        output = self.infer(image)

        for name in self._output_layer_names:
            print('output: '+str(output[name].shape[2]))

            print(name)
        #     print(output[name][0][0])
        #     dettections = self.postprocess(output[name][0][0], image_sizes)
            for i in range(output[name].shape[2]):
                print(output[name][0][0][i])
        # print(self._output_layer_names)

        # detections = self.postprocess(
        #     [output[name][0] for name in self._output_layer_names], image_sizes)
