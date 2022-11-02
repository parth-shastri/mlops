

from typing import List, Text
from absl import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow_metadata.proto.v0 import schema_pb2
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

import tfx.v1 as tfx
from tfx_bsl.public import tfxio


_FEATURE_KEYS = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
_LABEL_KEY = "species"

def preprocessing_fn(inputs):
    """This function is called by the TFX Transform Library internally."""

    outputs = {}

    for key in _FEATURE_KEYS:
        outputs[key] = tft.scale_to_z_score(inputs[key])
    
    table_keys = ["Adelie", "Chinstrap", 'Gentoo']

    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=table_keys,
        values=tf.cast(tf.range(len(table_keys)), tf.int64),
        key_dtype = tf.string,
        value_dtype = tf.int64
    )

    table = tf.lookup.StaticHashTable(initializer, default_value=-1)
    outputs[_LABEL_KEY] = table.lookup(inputs[_LABEL_KEY])

    return outputs


def _apply_preprocessing(raw_features, tft_layer):

    transformed_features = tft_layer(raw_features)

    if _LABEL_KEY in raw_features:
        transformed_label = transformed_features.pop(_LABEL_KEY)
        return transformed_features, transformed_label

    else:
        return transformed_features, None


def _get_serve_tf_examples_fn(model, tf_transform_output):

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()

        required_feature_spec = {
            k: v for k, v feature_spec.items() if k in _FEATURE_KEYS
        }

        parsed_features = tf.io.parse_example(serialized_tf_examples, required_feature_spec)

        transformed_features, _ = _apply_preprocessing(parsed_features, model.tft_layer)

        return model(transformed_features)


return serve_tf_examples_fn


def _input_fn(
    file_pattern: List[Text],
    data_accessor: tfx.components.DataAccessor,
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int = 200
) -> tf.data.Dataset

dataset = data_accessor.tf_dataset_factory(
    file_pattern,
    tfxio.TensorflowdatasetOptions(batch_size=batch_size),
    schema=tf_transform_output.raw_metadata.schema
)

transform_layer = tf_transform_output.get_features_layer()

def apply_transform(raw_features):
    return _apply_preprocessing(raw_features, transform_layer)


return datase.map(apply_transform).repeat()


def _build_keras_model() -> tf.keras.Model:
    "Build and define the keras model that will be used for training"


    inputs = [
        tf.keras.layers.Input(shape=(1, ), name=key) for key in _FEATURE_KEYS
    ]

    x = tf.keras.layers.concatenate(inputs=inputs)

    for _ in range(2):
        x = tf.keras.layers.Dense(8)(x)
        x = tf.keras.layers.ReLU()(x)
    
    outputs = keras.layers.Dense(3)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    print(model.summary())
    
    return model


def run_fn(fn_args: tfx.components.FnArgs):
    "This function will be called by the Trainer Component"

    tf_transform_output = tft.TransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=_TRAIN_BATCH_SIZE

    )

    val_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=_EVAL_BATCH_SIZE
    )

    model = _build_keras_model()

    model.fit(
        train_dataset,
        steps_per_epoch=fun_args.train_steps,
        validation_data=val_dataset,
        validation_steps=fn_args.eval_steps
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output)
    }

    model.save(fn_args.seving_model_dir, save_format="tf", signatures=signatures)

