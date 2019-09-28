import scipy.misc
import os

save_dir = '/home/yzs/PycharmProjects/face recognize/datasets/MNIST/raw'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)
for i in range(20):
    image = mnist.train.images[i, :]
    image = image.reshape(28, 28)
    file = save_dir + 'mnist_train_%d.jpg' % i
    scipy.misc.toimage(image, cmin=0.0, cmax=1.0).save(file)
