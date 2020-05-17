import os

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'dataset/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return 'dataset/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

def data_path(path):
    return os.path.join('dataset/', path)


TRAIN_IMAGES_HQ_PATH = data_path('train_hq')
TRAIN_MASKS_PATH = data_path('train_masks')
TEST_IMAGES_PATH = data_path('test_hq')
SAMPLE_SUBMISSION_PATH = data_path('sample_submission.csv')
PREDICTIONS_PATH = 'predictions'


NUM_WORKERS = 4
SEED = 42


IMG_MEAN = [0.698228, 0.690886, 0.683951]
IMG_STD = [0.244182, 0.248307, 0.245187]