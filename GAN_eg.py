import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os



# Generator model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim),
        layers.Dense(784, activation='tanh')  # Output in [-1, 1]
    ])
    return model


# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated, training=True)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = (cross_entropy(tf.ones_like(real_output), real_output) +
                     cross_entropy(tf.zeros_like(fake_output), fake_output)) / 2

    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

# Function to generate and save images
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i].numpy().reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Epoch {epoch}')
    os.makedirs('images', exist_ok=True)
    plt.savefig(f'images/image_at_epoch_{epoch:03d}.png')
    plt.close()

# Training loop
def train(dataset, epochs):
    seed = tf.random.normal([16, latent_dim])
    for epoch in range(1, epochs + 1):
        for image_batch in dataset:
            train_step(image_batch)

        if epoch % 5 == 0:
            # generate_and_save_images(generator, epoch, seed)
            print(f"Epoch {epoch} completed.")
            
if __name__ == '__main__':
    # Hyperparameters
    latent_dim = 100
    batch_size = 256
    epochs = 50 # 500 # 1000 # 2000
    
    # Load and preprocess MNIST
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 28*28)).astype("float32")
    x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]
    
    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size)
    
    
    # Instantiate models
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Loss and optimizers
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    gen_optimizer = tf.keras.optimizers.Adam(1e-4)
    disc_optimizer = tf.keras.optimizers.Adam(1e-4)


    # Start training
    train(dataset, epochs)
    
        
    # Set seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Instantiate and generate
    noise = tf.random.normal([16, 100])
    generated = generator(noise, training=False)
    images = generated.numpy().reshape(-1, 28, 28)
    
    # Plot
    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.suptitle("Generated Digits by GAN", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
# [EOF]
