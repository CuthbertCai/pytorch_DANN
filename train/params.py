from models import models

# utility params
fig_mode = None
embed_plot_epoch=10

# model params
use_gpu = True
dataset_mean = (0.5, 0.5, 0.5)
dataset_std = (0.5, 0.5, 0.5)

batch_size = 512
epochs = 1000
gamma = -10
theta = 1

# path params
data_root = './data'

mnist_path = data_root + '/MNIST'
mnistm_path = data_root + '/MNIST_M'
svhn_path = data_root + '/SVHN'
syndig_path = data_root + '/SynthDigits'
save_dir = './experiment'


# specific dataset params
extractor_dict = {'MNIST_MNIST_M': models.Extractor(),
                  'SVHN_MNIST': models.SVHN_Extractor(),
                  'SynDig_SVHN': models.SVHN_Extractor()}

class_dict = {'MNIST_MNIST_M': models.Class_classifier(),
              'SVHN_MNIST': models.SVHN_Class_classifier(),
              'SynDig_SVHN': models.SVHN_Class_classifier()}

domain_dict = {'MNIST_MNIST_M': models.Domain_classifier(),
               'SVHN_MNIST': models.SVHN_Domain_classifier(),
               'SynDig_SVHN': models.SVHN_Domain_classifier()}
