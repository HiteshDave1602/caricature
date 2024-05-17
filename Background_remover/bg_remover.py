# import os
# import cv2
# from glob import glob
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers


# IMAGE_SIZE = 512
# BATCH_SIZE = 4
# NUM_CLASSES = 20
# DATA_DIR = "Training"
# NUM_TRAIN_IMAGES = 1000
# NUM_VAL_IMAGES = 50


# train_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[:NUM_TRAIN_IMAGES]
# train_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[:NUM_TRAIN_IMAGES]

# val_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[NUM_TRAIN_IMAGES:NUM_TRAIN_IMAGES+NUM_VAL_IMAGES]
# val_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[NUM_TRAIN_IMAGES:NUM_TRAIN_IMAGES+NUM_VAL_IMAGES]



# def read_image(image_path, mask=False):
#     image = tf.io.read_file(image_path)
#     if mask:
#         image = tf.image.decode_png(image, channels=1)
#         image.set_shape([None, None, 1])
#         image  = tf.image.resize(images =  image, size=[IMAGE_SIZE, IMAGE_SIZE])
#     else:
#         image = tf.image.decode_png(image, channels=3)
#         image.set_shape([None, None, 3])
#         image  = tf.image.resize(images =  image, size=[IMAGE_SIZE, IMAGE_SIZE])
#         image = tf.keras.applications.resnet50.preprocess_input(image)
#     return image

# def load_data(image_list, mask_list):
#     image = read_image(image_list)
#     mask = read_image(mask_list, mask=True)
#     return image, mask


# def data_generator(image_list, mask_list):
    
#     dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
#     dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
#     dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
#     return dataset

# train_dataset = data_generator(train_images, train_masks)
# val_dataset = data_generator(val_images, val_masks)

# train_dataset


# def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, padding='same', use_bias=False,):
#     x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate = dilation_rate, padding= "same", use_bias= use_bias,kernel_initializer=keras.initializers.HeNormal(),
# )(block_input)
#     x = layers.BatchNormalization()(x)
#     return tf.nn.relu(x)


# def DilatedSpatialPyramidPooling(dspp_input):
#     dims = dspp_input.shape
#     x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
#     x = convolution_block(x, kernel_size=1, use_bias=True)
#     out_pool = layers.UpSampling2D(
#         size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",)(x)

#     out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
#     out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
#     out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
#     out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

#     x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
#     output = convolution_block(x, kernel_size=1)
#     return output


# def DeeplabV3(image_size, num_classes):
#     model_input = keras.Input(shape=(image_size, image_size, 3))
#     resnet50 = keras.applications.ResNet50(
#         weights="imagenet", include_top=False, input_tensor=model_input)
#     x = resnet50.get_layer("conv4_block6_2_relu").output
#     x = DilatedSpatialPyramidPooling(x)

#     input_a = layers.UpSampling2D(
#         size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
#         interpolation="bilinear",)(x)
#     input_b = resnet50.get_layer("conv2_block3_2_relu").output
#     input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

#     x = layers.Concatenate(axis=-1)([input_a, input_b])
#     x = convolution_block(x)
#     x = convolution_block(x)
#     x = layers.UpSampling2D(
#         size=(image_size // x.shape[1], image_size // x.shape[2]),
#         interpolation="bilinear",)(x)
#     model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
#     return keras.Model(inputs=model_input, outputs=model_output)


# model = DeeplabV3(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
# model.summary()

# loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# optimizer = keras.optimizers.Adam(learning_rate=0.001)
# model.compile(optimizer= optimizer , loss=loss, metrics=['accuracy'])

# history = model.fit(train_dataset,validation_data=val_dataset, epochs=100 )


# Import necessary modules:
import tensorflow_hub as hub
import tensorflow as tf
import cv2
import numpy as np
from urllib.request import urlopen
import matplotlib.pyplot as plt

# Seed random generators:
np.random.seed(42)
tf.random.set_seed(42)

model = hub.KerasLayer("https://www.kaggle.com/models/vaishaknair456/u2-net-portrait-background-remover/tensorFlow2/40_saved_model/1")

def get_image_from_url(url, read_flag=cv2.IMREAD_COLOR):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, read_flag)
    return image

INPUT_IMG_HEIGHT = 512
INPUT_IMG_WIDTH = 512
INPUT_CHANNEL_COUNT = 3

image = get_image_from_url("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.goldcar.es%2Fen%2Fp%2Fcar-hire-19-24-years-old%2F&psig=AOvVaw3Gt6F71xve3aeV6qT3rUxD&ust=1715942464303000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCLjFkP79kYYDFQAAAAAdAAAAABAT")   # png images are also supported
h, w, channel_count = image.shape

if channel_count > INPUT_CHANNEL_COUNT:   
    image = image[..., :INPUT_CHANNEL_COUNT]

x = cv2.resize(image, (INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT))
x = x / 255.0
x = x.astype(np.float32)
x = np.expand_dims(x, axis=0)

probability = model(x)[0].numpy()

# Produce output image:
probability = cv2.resize(probability, dsize=(w, h))
probability = np.expand_dims(probability, axis=-1) 

alpha_image = np.insert(image, 3, 255.0, axis=2) 

PROBABILITY_THRESHOLD = 0.7  

masked_image = np.where(probability > PROBABILITY_THRESHOLD, alpha_image, 0.0)

# Save output to a png file:
cv2.imwrite("./output.png", masked_image)

