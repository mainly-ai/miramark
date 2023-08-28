# This is the MNIST example from tensorflow (https://github.com/tensorflow/datasets/blob/master/docs/keras_example.ipynb)
# but modified to use an unnecessarily complex CNN model to make it more computationally intensive

import json
import time
import tensorflow as tf
import tensorflow_datasets as tfds

def run(with_gpu = False):
	if with_gpu:
		if tf.test.gpu_device_name():
			print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
		else:
			print("Please install GPU version of TF")
			return
	else:
		tf.config.set_visible_devices([], 'GPU')
	(ds_train, ds_test), ds_info = tfds.load(
		'mnist',
		split=['train', 'test'],
		shuffle_files=True,
		as_supervised=True,
		with_info=True,
	)

	start_time_dataset_processing = time.time()
	def normalize_img(image, label):
		"""Normalizes images: `uint8` -> `float32`."""
		return tf.cast(image, tf.float32) / 255., label

	ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
	ds_train = ds_train.cache()
	ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
	ds_train = ds_train.batch(128)
	ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

	ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
	ds_test = ds_test.batch(128)
	ds_test = ds_test.cache()
	ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
	end_time_dataset_processing = time.time()
	
	model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
		tf.keras.layers.Conv2D(64, 3, activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Conv2D(64, 3, activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(10)
	])
	model.compile(
		optimizer=tf.keras.optimizers.Adam(0.001),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
	)

	fit_start_time = time.time()
	model.fit(
		ds_train,
		epochs=5,
		validation_data=ds_test,
		verbose=2,
	)
	fit_end_time = time.time()

	print(json.dumps({
		"dataset_processing_time": end_time_dataset_processing - start_time_dataset_processing,
		"fit_time": fit_end_time - fit_start_time
	}))