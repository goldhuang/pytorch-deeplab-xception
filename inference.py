import os
import numpy as np
import tqdm
import torch
import config
import torch.nn.functional as F

from PIL import Image, ImageOps, ImageFilter
from modeling.deeplab import *
from dataloaders import utils, make_data_loader
from utils.metrics import Evaluator

class Tester(object):
    def __init__(self):
        #path = 'run/carvana/resnet/experiment_1/checkpoint.pth.tar'
        path = 'run/carvana/resnet/model_best_1.pth.tar'
        self.cropsize = config.INPUT_SIZE
        if not os.path.isfile(path):
            raise RuntimeError("no checkpoint found at '{}'".format(path))
        self.color_map = utils.get_carvana_labels()
        self.nclass = 2

        # Define model
        model = DeepLab(num_classes=self.nclass,
                        backbone='resnet',
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)

        self.model = model
        device = torch.device('cpu')
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.evaluator = Evaluator(self.nclass)

    def save_image(self, array, id, op):
        text = 'gt'
        if op == 0:
            text = 'pred'
        file_name = id
        r = array.copy()
        g = array.copy()
        b = array.copy()

        for i in range(self.nclass):
            r[array == i] = self.color_map[i][0]
            g[array == i] = self.color_map[i][1]
            b[array == i] = self.color_map[i][2]

        rgb = np.dstack((r, g, b))
        save_img = Image.fromarray(rgb.astype('uint8'))
        save_img.save(config.PREDICTIONS_PATH + os.sep + file_name)

    def inference(self):
        self.model.eval()
        self.evaluator.reset()

        DATA_DIR = config.TEST_IMAGES_PATH
        SAVE_DIR = config.PREDICTIONS_PATH

        for idx, test_file in enumerate(os.listdir(DATA_DIR)):
            if test_file == '.DS_Store':
                continue
            test_img = Image.open(os.path.join(DATA_DIR, test_file)).convert('RGB')
            test_img = test_img.resize((self.cropsize, self.cropsize), Image.BILINEAR)
            test_array = np.array(test_img).astype(np.float32)
            # print(test_array.shape)

            # Normalize
            test_array /= 255.0
            test_array -= config.IMG_MEAN
            test_array /= config.IMG_STD
            width = test_array.shape[1]
            height = test_array.shape[0]

            inference_imgs = np.zeros((1280, 1918), dtype=np.float32)

            test_array = test_array.transpose((2, 0, 1))
            # print(test_array.shape)
            test_array_batch = np.expand_dims(test_array, axis=0)
            test_tensor = torch.from_numpy(test_array_batch)

            with torch.no_grad():
                output = self.model(test_tensor)
            # print(output.shape)
            output = F.softmax(output, dim=1)
            # print(output.shape)
            output = F.interpolate(output, size=(1280, 1918), mode='bilinear')
            # print(output.shape)
            inference_imgs[:,:] = output[0][1][:, :]*255

            print('inference ... {}/{}'.format(idx + 1, len(os.listdir(DATA_DIR))))
            # gray mode
            save_image = Image.fromarray(inference_imgs.astype('uint8'))
            save_image.save(os.path.join(SAVE_DIR, test_file))

def main():
    tester = Tester()
    print('predict...')
    tester.inference()


if __name__ == "__main__":
    main()