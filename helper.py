import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def random_aug(img):
    maxr = 1.0/(float(max(np.reshape(img[:,:,0], (-1,1)))[0])/256.0)
    maxg = 1.0/(float(max(np.reshape(img[:,:,1], (-1,1)))[0])/256.0)
    maxb = 1.0/(float(max(np.reshape(img[:,:,2], (-1,1)))[0])/256.0)


    img[:,:,0] = img[:,:,0]*random.uniform(0.3, maxr)
    img[:,:,1] = img[:,:,1]*random.uniform(0.3, maxg)
    img[:,:,2] = img[:,:,2]*random.uniform(0.3, maxb)
    return img


def gen_batch_function(data_folder, image_shape, labels = None):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        label_path = data_folder+ '/CameraSegAug'
        image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
        label_paths = glob(os.path.join(data_folder, 'CameraSegAug', '*.png'))
        background_color = np.array([0, 0, 0])
        

        random.shuffle(image_paths)
        for batch_i in range(0, 50, batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_path+'/'+image_file.split('/')[-1]

                image = random_aug(scipy.misc.imresize(scipy.misc.imread(image_file), image_shape))
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
#                if random.randint(1,2) == 2:
#                    image = np.flipud(image)
#                    gt_image = np.flipud(image)
                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                if not labels:
                    gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                else:
                    lbs = []
                    for l in labels:
                        lb = np.all(gt_image == (l,0,0), axis=2)
                        lb = lb.reshape(*lb.shape, 1)
                        lbs.append(lb)
                    gt_image = np.concatenate(lbs, axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmaxtmp = im_softmax
        im_softmax = im_softmax[0][:, 0].reshape(image_shape[0], image_shape[1])
        im_softmax2 = im_softmaxtmp[0][:, 1].reshape(image_shape[0], image_shape[1])
#        im_softmaxbg = im_softmaxtmp[0][:, 0].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > float(1/2)).reshape(image_shape[0], image_shape[1], 1)
        segmentation2 = (im_softmax2 > float(1/2)).reshape(image_shape[0], image_shape[1], 1)
 #       segmentationbg = (im_softmaxbg > float(1/3)).reshape(image_shape[0], image_shape[1], 1)

        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask2 = np.dot(segmentation2, np.array([[0, 0,255, 127]]))
  #      mask3 = np.dot(segmentationbg, np.array([[255, 0, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        mask2 = scipy.misc.toimage(mask2, mode='RGBA')
  #      mask3 = scipy.misc.toimage(mask3, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        street_im.paste(mask2, box=None, mask=mask2 )
   #     street_im.paste(mask3, box=None, mask=mask3)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, data_dir+'/CameraRGB/', image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
