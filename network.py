import logging
import os
import time
import numpy as np
from PIL import Image
from io import BytesIO

tf = None  # Define temp modules
ndi = None

logger = logging.getLogger(__name__)

def model_detect():
    global tf, ndi
    import scipy.ndimage as ndi
    import tensorflow as tf
    return TFSegmentation()


class TFSegmentation(object):
    
    def __init__(self):
        # Environment init
        self.INPUT_TENSOR_NAME = 'ImageTensor:0'
        self.OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
        self.INPUT_SIZE = 513
        self.FROZEN_GRAPH_NAME = 'frozen_inference_graph'
        # Start load process
        self.graph = tf.Graph()
        try:
            graph_def = tf.compat.v1.GraphDef.FromString(open(os.path.join("model", "frozen_inference_graph.pb"),
                                                              "rb").read())
        except FileNotFoundError:
            raise FileNotFoundError("No pre-trained model found!")

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.compat.v1.Session(graph=self.graph)

    def process_image(self, path: str):
        start_time = time.time()  # Time counter
        logger.debug('Load image')
        try:
            jpeg_str = open(path, "rb").read()
            image = Image.open(BytesIO(jpeg_str))
        except IOError:
            logger.error('Cannot retrieve image. Please check file: ' + path)
            return False
        seg_map = self.__predict__(image)
        logger.debug('Finished mask creation')
        image = image.convert('RGB')
        logger.debug("Mask overlay completed")
        image = self.__draw_segment__(image, seg_map)
        logger.debug("Finished! Time spent: {}".format(time.time() - start_time))
        return image

    def __predict__(self, image):
        """Image processing."""
        # Get image size
        width, height = image.size
        # Calculate scale value
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        # Calculate future image size
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        # Resize image
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        # Send image to model
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        # Get model output
        seg_map = batch_seg_map[0]
        # Get new image size and original image size
        width, height = resized_image.size
        width2, height2 = image.size
        # Calculate scale
        scale_w = width2 / width
        scale_h = height2 / height
        # Zoom numpy array for original image
        seg_map = ndi.zoom(seg_map, (scale_h, scale_w))
        return seg_map

    @staticmethod
    def __draw_segment__(image, alpha_channel):
        """Postprocessing. Saves complete image."""
        # Get image size
        width, height = image.size
        # Create empty numpy array
        dummy_img = np.zeros([height, width, 4], dtype=np.uint8)
        # Create alpha layer from model output
        for x in range(width):
            for y in range(height):
                color = alpha_channel[y, x]
                (r, g, b) = image.getpixel((x, y))
                if color == 0:
                    dummy_img[y, x, 3] = 0
                else:
                    dummy_img[y, x] = [r, g, b, 255]
        # Restore image object from numpy array
        img = Image.fromarray(dummy_img)
        return img
