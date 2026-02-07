import numpy
import cv2
import random
import math
import time
import torch
import tqdm
from utils import util
from nets import nn
from google.colab.patches import cv2_imshow

@torch.no_grad()
def demo(filenames, model=None):
    input_size = 800

    model = nn.DBNet()
    checkpoint = torch.load('./weights/db_last.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.float()#.cuda()

    model.eval()

    mean = numpy.array([0.406, 0.456, 0.485]).reshape((1, 1, 3)).astype('float32')
    std = numpy.array([0.225, 0.224, 0.229]).reshape((1, 1, 3)).astype('float32')
    for filename in tqdm.tqdm(filenames):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        print(image.shape)

        shape = image.shape[:2]

        width = shape[1] * input_size / shape[0]
        width = math.ceil(width / 32) * 32

        x = cv2.resize(image, dsize=(width, input_size))
        x = x.astype('float32') / 255.0
        x = x - mean
        x = x / std
        x = x.transpose((2, 0, 1))[::-1]
        x = numpy.ascontiguousarray(x)
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        #x = x.cuda()

        init = time.time()
        output = model(x)

        output = util.mask_to_box(targets={'shape': [shape]}, outputs=output.cpu(), is_polygon=True)

        print(time.time()-init)
        boxes, scores = output[0][0], output[1][0]

        for box in boxes:
            box = numpy.array(box).reshape((-1, 1, 2))
            cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=5)

        cv2_imshow(image)
        #cv2.imwrite("/content/DBNet/"+filename.split('/')[-1], image) #f'./data/{os.path.basename(filename)}', image)

filenames = []
with open('Dataset/test_aug.txt') as f:
  for filename in f.readlines():
    filename = filename.rstrip()
    filenames.append('Dataset/images_aug/' + filename)
demo(filenames)
