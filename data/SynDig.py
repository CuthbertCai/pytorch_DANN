from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import zipfile
import os
import urllib
import os.path
import numpy as np

class SynDig(data.Dataset):

    url = 'https://doc-08-a8-docs.googleusercontent.com/docs/securesc/4gco78h4r5v7n2eq50hcumr89oar2vtn/' \
          'j64h6ekj56csgpnthf6revr2h1sogunh/1513591200000/02005382952228186512/07954859324473388693/' \
          '0B9Z4d7lAwbnTSVR1dEFSRUFxOUU?e=download&nonce=i254fkf8136em&user=' \
          '07954859324473388693&hash=cbcagg6svrku8ot6c9e27m3saorf50m1'
    zipname = 'SynDigits.zip'
    split_list = {
        'train': ["synth_train_32x32.mat"],
        'train_small': ["synth_train_32x32_small.mat"],
        'test': ["synth_test_32x32.mat",""],
        'test_small': ["synth_test_32x32_small.mat"]
    }

    def __init__(self, root, split= 'train', transform= None,
                 target_transform= None, download= False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="train_small or split="test" or split="test_small"')
        self.filename = self.split_list[self.split][0]

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               'You can use download=True to download it.')

        import scipy.io as sio

        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))
        self.data = loaded_mat['X']
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):

        img, target = self.data[index], self.labels[index]

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.filename))


    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.zipname)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        file = zipfile.ZipFile(filename)
        file.extractall()
        file.close()
        print("[DONE]")
        return