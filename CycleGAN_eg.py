# cycle_gan_photo_style.py

import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------- Helper Functions -----------
def load_image(path):
    img = Image.open(path).resize((256, 256))
    img = np.array(img) / 127.5 - 1.0
    return tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.float32), axis=0)

def postprocess_image(img_tensor):
    img = img_tensor[0].numpy()
    img = (img + 1.0) * 127.5
    return Image.fromarray(np.uint8(np.clip(img, 0, 255)))

# ----------- Model Definitions -----------
def build_generator():
    inputs = tf.keras.Input(shape=[256, 256, 3])
    x = layers.Conv2D(64, 7, padding='same')(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)

    for _ in range(6):
        y = layers.Conv2D(256, 3, padding='same')(x)
        y = layers.ReLU()(y)
        y = layers.Conv2D(256, 3, padding='same')(y)
        x = layers.add([x, y])

    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(3, 7, padding='same', activation='tanh')(x)

    return tf.keras.Model(inputs, x, name="generator")

def build_discriminator():
    inputs = tf.keras.Input(shape=[256, 256, 3])
    x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    for f in [128, 256, 512]:
        x = layers.Conv2D(f, 4, strides=2, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(1, 4, padding='same')(x)
    return tf.keras.Model(inputs, x, name="discriminator")

# ----------- Loss Functions -----------
loss_obj = tf.keras.losses.MeanSquaredError()

def generator_loss(fake):
    return loss_obj(tf.ones_like(fake), fake)

def discriminator_loss(real, fake):
    return 0.5 * (loss_obj(tf.ones_like(real), real) + loss_obj(tf.zeros_like(fake), fake))

def cycle_loss(real, cycled):
    return tf.reduce_mean(tf.abs(real - cycled))

def identity_loss(real, same):
    return tf.reduce_mean(tf.abs(real - same))

# ----------- Instantiate Models -----------
G = build_generator()   # X -> Y
F = build_generator()   # Y -> X
D_X = build_discriminator()
D_Y = build_discriminator()

# Use legacy optimizer to avoid TF variable issues
from tensorflow.keras.optimizers.legacy import Adam
g_optimizer = Adam(2e-4, beta_1=0.5)
d_optimizer = Adam(2e-4, beta_1=0.5)

# ----------- Training Step -----------
@tf.function
def train_step(real_x, real_y):
    lambda_cycle = 10.0
    lambda_id = 5.0

    with tf.GradientTape(persistent=True) as tape:
        fake_y = G(real_x, training=True)
        cycled_x = F(fake_y, training=True)

        fake_x = F(real_y, training=True)
        cycled_y = G(fake_x, training=True)

        same_x = F(real_x, training=True)
        same_y = G(real_y, training=True)

        disc_real_x = D_X(real_x, training=True)
        disc_real_y = D_Y(real_y, training=True)
        disc_fake_x = D_X(fake_x, training=True)
        disc_fake_y = D_Y(fake_y, training=True)

        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)
        total_cycle_loss = cycle_loss(real_x, cycled_x) + cycle_loss(real_y, cycled_y)
        id_loss = identity_loss(real_x, same_x) + identity_loss(real_y, same_y)

        total_gen_g = gen_g_loss + lambda_cycle * total_cycle_loss + lambda_id * id_loss
        total_gen_f = gen_f_loss + lambda_cycle * total_cycle_loss + lambda_id * id_loss

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    g_grads = tape.gradient(total_gen_g, G.trainable_variables)
    f_grads = tape.gradient(total_gen_f, F.trainable_variables)
    dx_grads = tape.gradient(disc_x_loss, D_X.trainable_variables)
    dy_grads = tape.gradient(disc_y_loss, D_Y.trainable_variables)

    g_optimizer.apply_gradients(zip(g_grads, G.trainable_variables))
    g_optimizer.apply_gradients(zip(f_grads, F.trainable_variables))
    d_optimizer.apply_gradients(zip(dx_grads, D_X.trainable_variables))
    d_optimizer.apply_gradients(zip(dy_grads, D_Y.trainable_variables))

# ----------- Main Execution -----------
if __name__ == '__main__':
    # Load your two images (place photo_x.jpg and photo_y.jpg in the working dir)
    x_img = load_image("Donald_J._Trump.bmp")
    y_img = load_image("Vincent_Willem_van_Gogh.bmp")

    print("Training on 2 images")
    Epoch = 100 # 1000 # 2000
    for i in range(Epoch):
        train_step(x_img, y_img)
        if i % 10 == 0:
            print(f"Step {i} complete")

    # Stylize X to Y style
    stylized = G(x_img, training=False)
    out_img = postprocess_image(stylized)
    out_img.save("stylized_x.jpg")
    # out_img.show()
    # print("Saved transformed image as 'stylized_x.jpg'")

# [EOF]