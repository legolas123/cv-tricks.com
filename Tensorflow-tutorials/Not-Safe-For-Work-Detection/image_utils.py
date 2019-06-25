VGG_MEAN = [104, 117, 123]


def create_yahoo_image_loader(expand_dims=True):
    """Yahoo open_nsfw image loading mechanism

    Approximation of the image loading mechanism defined in
    https://github.com/yahoo/open_nsfw/blob/79f77bcd45076b000df71742a59d726aa4a36ad1/classify_nsfw.py#L40
    """
    import numpy as np
    import skimage
    import skimage.io
    from PIL import Image
    from io import BytesIO

    def load_image(image_path):
        pimg = open(image_path, 'rb').read()

        img_data = pimg
        im = Image.open(BytesIO(img_data))

        if im.mode != "RGB":
            im = im.convert('RGB')

        imr = im.resize((256, 256), resample=Image.BILINEAR)

        fh_im = BytesIO()
        imr.save(fh_im, format='JPEG')
        fh_im.seek(0)

        image = (skimage.img_as_float(skimage.io.imread(fh_im, as_gray=False))
                        .astype(np.float32))

        H, W, _ = image.shape
        h, w = (224, 224)

        h_off = max((H - h) // 2, 0)
        w_off = max((W - w) // 2, 0)
        image = image[h_off:h_off + h, w_off:w_off + w, :]

        # RGB to BGR
        image = image[:, :, :: -1]

        image = image.astype(np.float32, copy=False)
        image = image * 255.0
        image -= np.array(VGG_MEAN, dtype=np.float32)

        if expand_dims:
            image = np.expand_dims(image, axis=0)

        return image

    return load_image


def create_tensorflow_image_loader(session, expand_dims=True,
                                   options=None,
                                   run_metadata=None):
    """Tensorflow image loader

    Results seem to deviate quite a bit from yahoo image loader due to
    different jpeg encoders/decoders and different image resize
    implementations between PIL, skimage and tensorflow

    Only supports jpeg images.

    Relevant tensorflow issues:
        * https://github.com/tensorflow/tensorflow/issues/6720
        * https://github.com/tensorflow/tensorflow/issues/12753
    """
    import tensorflow as tf

    def load_image(image_path):
        image = tf.read_file(image_path)
        image = __tf_jpeg_process(image)

        if expand_dims:
            image_batch = tf.expand_dims(image, axis=0)
            return session.run(image_batch,
                               options=options,
                               run_metadata=run_metadata)

        return session.run(image,
                           options=options,
                           run_metadata=run_metadata)

    return load_image


def load_base64_tensor(_input):
    import tensorflow as tf

    def decode_and_process(base64):
        _bytes = tf.decode_base64(base64)
        _image = __tf_jpeg_process(_bytes)

        return _image

    # we have to do some preprocessing with map_fn, since functions like
    # decode_*, resize_images and crop_to_bounding_box do not support
    # processing of batches
    image = tf.map_fn(decode_and_process, _input,
                      back_prop=False, dtype=tf.float32)

    return image


def __tf_jpeg_process(data):
    import tensorflow as tf

    # The whole jpeg encode/decode dance is neccessary to generate a result
    # that matches the original model's (caffe) preprocessing
    # (as good as possible)
    image = tf.image.decode_jpeg(data, channels=3,
                                 fancy_upscaling=True,
                                 dct_method="INTEGER_FAST")

    image = tf.image.convert_image_dtype(image, tf.float32, saturate=True)
    image = tf.image.resize_images(image, (256, 256),
                                   method=tf.image.ResizeMethod.BILINEAR,
                                   align_corners=True)

    image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)

    image = tf.image.encode_jpeg(image, format='', quality=75,
                                 progressive=False, optimize_size=False,
                                 chroma_downsampling=True,
                                 density_unit=None,
                                 x_density=None, y_density=None,
                                 xmp_metadata=None)

    image = tf.image.decode_jpeg(image, channels=3,
                                 fancy_upscaling=False,
                                 dct_method="INTEGER_ACCURATE")

    image = tf.cast(image, dtype=tf.float32)

    image = tf.image.crop_to_bounding_box(image, 16, 16, 224, 224)

    image = tf.reverse(image, axis=[2])
    image -= VGG_MEAN

    return image
