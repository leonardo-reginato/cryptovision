"""
Taxonomy-Focal Cross-Loss (TFCL), as formulated in the paper (Eqs. 1-5):

    L_focal = -a (1 - p_t)^g log p_t, p_t = probability of the true class               (1)
    L_fam = -a (1 - p_fam[y_fam])^g log p_fam[y_fam]                                    (2)
    L_gen = -a (1 - p_gen[y_gen])^g log p_gen[y_gen] + l_gen * [-log p_fam[par(y_gen)]] (3)
    L_spe = -a (1 - p_spe[y_spe])^g log p_spe[y_spe] + l_spe * [-log p_gen[par(y_spe)]] (4)
    L_TFCL = L_fam + L_gen + L_spe                                                      (5)

The focal terms are the per-head Keras losses (focal_loss); the cross-head penalties
need two heads at once, so TFCLModel adds them in compute_loss.

Usage:
    base = CryptoVisionModels.basic(...)
    model = TFCLModel(parent_genus, parent_species, inputs=base.inputs, outputs=base.outputs)
    loss = focal_loss()
    model.compile(optimizer, loss={"family": loss, "genus": loss, "species": loss})
"""

import tensorflow as tf


def focal_loss(alpha: float = 0.1, gamma: float = 2.0):
    """
    Focal cross-entropy of one head, Eq. 1 (the first term of Eqs. 2-4):
    -alpha (1 - p_t)^gamma log p_t, with p_t the probability of the true class.
    Returns a Keras loss function y_true (one-hot), y_pred -> per-sample loss.
    """

    def loss_fn(y_true, y_pred):
        # Keras 2 passes float16 tensors under mixed precision; Keras 3 already casts
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        return -alpha * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t + 1e-8)

    return loss_fn


_KERAS2 = int(tf.keras.__version__.split(".")[0]) < 3


class TFCLModel(tf.keras.Model):
    """
    Functional CryptoVision model trained with TFCL. Adds the cross-head penalties
    of Eqs. 3-4 to the compiled per-head focal losses:

        L_gen += lambda_gen * -log p_fam[par(y_gen)]
        L_spe += lambda_spe * -log p_gen[par(y_spe)]

    parent_genus[g] is the family index of genus g; parent_species[s] the genus
    index of species s (see data.make_parent_lists).
    """

    def __init__(
        self,
        parent_genus,
        parent_species,
        lambda_gen: float = 0.1,
        lambda_spe: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.parent_genus = tf.constant(parent_genus, dtype=tf.int32)
        self.parent_species = tf.constant(parent_species, dtype=tf.int32)
        self.lambda_gen = lambda_gen
        self.lambda_spe = lambda_spe
        if _KERAS2:
            self._total_loss = tf.keras.metrics.Mean(name="total_loss")

    @staticmethod
    def _neg_log_parent(parent_map, y_child, p_parent):
        true_child = tf.argmax(y_child, axis=-1, output_type=tf.int32)
        true_parent = tf.gather(parent_map, true_child)
        p = tf.gather(tf.cast(p_parent, tf.float32), true_parent, batch_dims=1)
        return tf.reduce_mean(-tf.math.log(tf.clip_by_value(p, 1e-7, 1.0)))

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, **kwargs):
        loss = super().compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, **kwargs
        )
        p_fam, p_gen, _ = y_pred
        gen_penalty = self._neg_log_parent(self.parent_genus, y["genus"], p_fam)
        spe_penalty = self._neg_log_parent(self.parent_species, y["species"], p_gen)
        return (
            loss
            + self.lambda_gen * tf.cast(gen_penalty, loss.dtype)
            + self.lambda_spe * tf.cast(spe_penalty, loss.dtype)
        )

    if _KERAS2:
        # Keras 2 (TF <= 2.15) reports the compiled loss only, so "loss" and
        # "val_loss" would miss the penalties; track the full loss instead

        @property
        def metrics(self):
            return super().metrics + [self._total_loss]

        def _logs(self, x, y, y_pred, sample_weight, loss):
            self._total_loss.update_state(loss)
            logs = self.compute_metrics(x, y, y_pred, sample_weight)
            logs["loss"] = logs.pop("total_loss")
            return logs

        def train_step(self, data):
            x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compute_loss(x, y, y_pred, sample_weight)
            self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
            return self._logs(x, y, y_pred, sample_weight, loss)

        def test_step(self, data):
            x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
            y_pred = self(x, training=False)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
            return self._logs(x, y, y_pred, sample_weight, loss)
