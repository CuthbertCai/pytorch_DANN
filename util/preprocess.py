import os, shutil

data_dir = './data/MNIST_M'
train_labels = './data/MNIST_M/mnist_m_train_labels.txt'
test_labels = './data/MNIST_M/mnist_m_test_labels.txt'
train_images = './data/MNIST_M/mnist_m_train'
test_images = './data/MNIST_M/mnist_m_test'

def mkdirs(path):
    train_dir = path + '/' + 'train'
    test_dir = path + '/' + 'test'
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    for i in range(0, 10):
        if not os.path.exists(train_dir + '/' + str(i)):
            os.mkdir(train_dir + '/' + str(i))
        if not os.path.exists(test_dir + '/' + str(i)):
            os.mkdir(test_dir + '/' + str(i))

def process(labels_path, images_path, data_dir):
    with open(labels_path) as f:
        for line in f.readlines():
            img = images_path + '/' + line.split()[0]
            dir = data_dir + '/' + line.split()[1]
            shutil.move(img, dir)

mkdirs(data_dir)
process(train_labels, train_images, data_dir + '/train')
process(test_labels, test_images, data_dir + '/test')
os.remove(train_images)
os.remove(test_images)