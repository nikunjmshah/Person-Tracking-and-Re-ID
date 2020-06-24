#    Import torchreid

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import torchreid

from torchreid.data import ImageDataset

#    Create own dataset

class NewDataset(ImageDataset):
    dataset_dir = 'Test_folder'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).

        # Assuming image format is pid_camid_info.jpg
        # Query pid and camid don't matter can be wrong too

        q_dir = 'query/'
        g_dir = 'gallery/'

        q_dir = osp.join(self.dataset_dir, q_dir)
        g_dir = osp.join(self.dataset_dir, g_dir)

        train = [['temp.jpg',0,0]]
        query = []
        q_files = os.listdir(q_dir)
        # print(q_files)
        for file in q_files:
            if file.endswith(".jpg"):
                filename = file.split('.')[0]
                query.append([q_dir + file, int(filename.split('_')[1]), filename.split('_')[0]])
        # print(query)

        gallery = []
        g_files = os.listdir(g_dir)
        # print(g_files)
        for file in g_files:
            if file.endswith(".jpg"):
                filename = file.split('.')[0]
                # print(filename)
                gallery.append([g_dir + file, int(filename.split('_')[1]), filename.split('_')[0]])
        # print(query)


        super(NewDataset, self).__init__(train, query, gallery, **kwargs)

#   Register dataset
torchreid.data.register_image_dataset('Test_dataset', NewDataset)


#    Load data manager

datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='Test_dataset',
    targets='Test_dataset',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop']
)

#    Build model, optimizer and lr_scheduler
# market1501 train ids = 751
model = torchreid.models.build_model(
    name='resnet50',
    num_classes=751,
    loss='softmax',
    pretrained=True
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

#    Build engine

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

#    Run training and test

#model.load_weights('log/resnet50/model.pth.tar-60')
torchreid.utils.load_pretrained_weights(model, 'log/resnet50/model.pth.tar-60')

engine.run(
    save_dir='reid-data/',
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=True,
    visrank = True,
    visrank_topk = 5
)
