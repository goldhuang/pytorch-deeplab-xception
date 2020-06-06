import os
import numpy as np
from tqdm import tqdm
import torch
import config
import torch.nn.functional as F

from modeling.deeplab import *
from dataloaders import utils

from PIL import Image

class Tester(object):
    def __init__(self):
        path = 'run/carvana/half-fold4/model_best.pth.tar'
        path1 = 'run/carvana/half-fold2-new/model_best.pth.tar'

        self.cropsize = config.INPUT_SIZE
        if not os.path.isfile(path):
            raise RuntimeError("no checkpoint found at '{}'".format(path))
        self.color_map = utils.get_carvana_labels()
        self.nclass = 2

        # Define model
        model0 = DeepLab(num_classes=self.nclass,
                        backbone='resnet',
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)
        self.model0 = model0.cuda()
        checkpoint0 = torch.load(path)
        self.model0.load_state_dict(checkpoint0['state_dict'])

        model1 = DeepLab(num_classes=self.nclass,
                         backbone='resnet',
                         output_stride=16,
                         sync_bn=False,
                         freeze_bn=False)
        self.model1 = model1.cuda()
        checkpoint1 = torch.load(path1)
        self.model1.load_state_dict(checkpoint1['state_dict'])

    def save_image(self, array, id, op):
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

    def rle_encode(self, mask_image):
        pixels = mask_image.flatten()
        pixels[0] = 0
        pixels[-1] = 0
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
        runs[1::2] = runs[1::2] - runs[:-1:2]
        return runs

    def rle_to_string(self, runs):
        return ' '.join(str(x) for x in runs)

    def inference(self):
        self.model0.eval()
        self.model1.eval()

        with torch.no_grad():
            DATA_DIR = config.TEST_IMAGES_PATH

            with open('submission.csv', 'w') as submission_csv:
                submission_csv.write('img,rle_mask\n')

                for idx, test_file in enumerate(os.listdir(DATA_DIR)):
                    if test_file == '.DS_Store':
                        continue
                    test_img = Image.open(os.path.join(DATA_DIR, test_file)).convert('RGB')
                    test_img = test_img.resize((self.cropsize, self.cropsize), Image.BILINEAR)
                    test_array = np.array(test_img).astype(np.float32)
                    # Normalize
                    test_array /= 255.0
                    test_array -= config.IMG_MEAN
                    test_array /= config.IMG_STD

                    test_array = test_array.transpose((2, 0, 1))
                    test_array_batch = np.expand_dims(test_array, axis=0)
                    test_tensor = torch.from_numpy(test_array_batch).cuda()

                    output0 = self.model0(test_tensor)
                    output0 = F.softmax(output0, dim=1)

                    output1 = self.model1(test_tensor)
                    output1 = F.softmax(output1, dim=1)

                    output_final = 0.5 * (output0 + output1)
                    output_final = F.interpolate(output_final, size=(1280, 1918), mode='bilinear', align_corners=True)
                    output_final = output_final.cpu().numpy()[0][1]
                    output_final = output_final > 0.5

                    rle = self.rle_encode(output_final)
                    submission_csv.write('{},{}\n'.format(
                        test_file, ' '.join(map(str, rle))))
                    print('inference ... {}/{}'.format(idx + 1, len(os.listdir(DATA_DIR))))

def main():
    tester = Tester()
    print('predict...')
    tester.inference()

if __name__ == "__main__":
    main()