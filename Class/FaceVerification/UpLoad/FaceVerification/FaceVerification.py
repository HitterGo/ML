import numpy as np
import tensorflow as tf
import facenet
from scipy import misc
from scipy import ndimage

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def crop(image, image_size):
    image = ndimage.interpolation.zoom(image, (image_size/image.shape[0], image_size/image.shape[1], 1.0))
    return image


def FaceVerification(X1, X2):

    # face feature extraction
    image_size = 160
    facenet_model = './models/feature_extract/20170512-110547/20170512-110547.pb'
    face_path = [X1, X2]

    with tf.Session() as sess1:
        # load feature extraction model
        facenet.load_model(facenet_model)

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # calculate features
        images = np.zeros((2, image_size, image_size, 3))
        for i in range(2):
            img = misc.imread(face_path[i])
            if img.ndim == 2:
                img = to_rgb(img)
            img = crop(img, image_size)
            img = prewhiten(img)
            images[i, :, :, :] = img

        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        emb_array = sess1.run(embeddings, feed_dict=feed_dict)

    dist_pairs = emb_array[0::2] - emb_array[1::2]
    verify_model = './models/classifier/'


    with tf.Session() as sess2:
        # load verification model
        ckpt = tf.train.get_checkpoint_state(verify_model)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
        saver.restore(sess2, ckpt.model_checkpoint_path)
        x = tf.get_collection("x")[0]
        predict_issame = tf.get_collection("pred_issame")[0]

        # fave verification
        predict_results = sess2.run(predict_issame, feed_dict={x: dist_pairs})
        predict_results = np.argmax(predict_results, 1)

    Y = predict_results[0]
    return Y