# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:58:10 2024

@author: 6
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras.layers as layers

import time
import numpy as np
import matplotlib.pyplot as plt

try:
    from .transformer import Transformer
    from .self_attention import create_mask
except ImportError:
    from transformer import Transformer
    from self_attention import create_mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(tf.maximum(step, 1), tf.float32)
        d_model = tf.cast(self.d_model, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * tf.math.pow(warmup_steps, -1.5)
        return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
        }


# @tf.function
def train_step(inputs, targets):
    tar_inp = targets[:,:-1]
    tar_real = targets[:,1:]
    # 构造掩码
    encode_padding_mask, combined_mask, decode_padding_mask = create_mask(inputs, tar_inp)


    with tf.GradientTape() as tape:
        predictions, _ = transformer(
            inputs,
            tar_inp,
            training=True,
            encode_padding_mask=encode_padding_mask,
            look_ahead_mask=combined_mask,
            decode_padding_mask=decode_padding_mask,
        )
        loss = loss_fun(tar_real, predictions)
    # 求梯度
    gradients = tape.gradient(loss, transformer.trainable_variables)
    # 反向传播
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    # 记录loss和准确率
    train_loss(loss)
    train_accuracy(tar_real, predictions)


def build_transformer_variables(model, max_length):
    dummy_inputs = tf.zeros((1, max_length), dtype=tf.int32)
    dummy_targets = tf.zeros((1, max_length - 1), dtype=tf.int32)
    encode_padding_mask, combined_mask, decode_padding_mask = create_mask(
        dummy_inputs, dummy_targets
    )
    model(
        dummy_inputs,
        dummy_targets,
        training=False,
        encode_padding_mask=encode_padding_mask,
        look_ahead_mask=combined_mask,
        decode_padding_mask=decode_padding_mask,
    )


def checkpoint_matches_model(checkpoint_path, model):
    checkpoint_vars = tf.train.list_variables(checkpoint_path)
    saved_trainable_shapes = []
    for name, shape in checkpoint_vars:
        prefix = 'optimizer/_trainable_variables/'
        suffix = '/.ATTRIBUTES/VARIABLE_VALUE'
        if name.startswith(prefix) and name.endswith(suffix):
            index = int(name[len(prefix): -len(suffix)])
            saved_trainable_shapes.append((index, list(shape)))

    saved_trainable_shapes.sort(key=lambda item: item[0])
    current_trainable_shapes = [
        var.shape.as_list() for var in model.trainable_variables
    ]

    if len(saved_trainable_shapes) != len(current_trainable_shapes):
        return False, 'trainable variable count differs (checkpoint {}, model {})'.format(
            len(saved_trainable_shapes), len(current_trainable_shapes)
        )

    for (index, saved_shape), current_shape in zip(
        saved_trainable_shapes, current_trainable_shapes
    ):
        if saved_shape != current_shape:
            return False, 'variable {} shape differs (checkpoint {}, model {})'.format(
                index, saved_shape, current_shape
            )

    return True, ''


#定义目标函数
def loss_fun(y_ture, y_pred):
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction='none')
    mask = tf.math.logical_not(tf.math.equal(y_ture, 0))  # 为0掩码标1
    loss_ = loss_object(y_ture, y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)



#%% 
if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    
    import re
    import jieba
    MAX_LENGTH = 12
    TRAIN_LIMIT = int(os.getenv('TRAINER_TRAIN_LIMIT', '20000'))
    EPOCHS = int(os.getenv('TRAINER_EPOCHS', '20'))
    def load_data(file_path):
        words_re = re.compile(r'\w+')
        corpus = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for k in f:
                en_sent, ch_sent, _ = k.rstrip("\n").split("\t")
                if en_sent.find(",") >= 0 or ch_sent.find(",") >= 0:
                    continue
                en_sent = words_re.findall(en_sent.lower())
                ch_sent = ch_sent[:-1]
                if len(en_sent) < MAX_LENGTH and len(ch_sent) < MAX_LENGTH:
                    en_sent = " ".join(en_sent)
                    corpus.append([en_sent, ch_sent])
        return corpus
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus = load_data(os.path.join(script_dir, 'cmn.txt'))
    
    ### 分词
    corpus_format = []
    for k in corpus:
        en = k[0]
        ch = k[1]
        ch = " ".join(jieba.cut(ch, cut_all=False))
        corpus_format.append([en, ch])
    
    if TRAIN_LIMIT <= 0 or TRAIN_LIMIT >= len(corpus_format):
        raise ValueError(
            'TRAINER_TRAIN_LIMIT must be between 1 and {}.'.format(len(corpus_format) - 1)
        )
    train_examples, val_examples = corpus_format[:TRAIN_LIMIT], corpus_format[TRAIN_LIMIT:]
    tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (k[0] for k in train_examples), target_vocab_size=2**13)
    tokenizer_ch = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (k[1] for k in train_examples), target_vocab_size=2**13)
    
    sample_str = 'tom intends to play tennis every day during his summer vacation'
    tokenized_str = tokenizer_en.encode(sample_str)
    print(tokenized_str)
    original_str = tokenizer_en.decode(tokenized_str)
    print(original_str)
    
    #然后将数据转换为Transformer输入的格式：
    
    def encode(lang):
        lang1, lang2 = lang
        lang1 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang1) + [tokenizer_en.vocab_size + 1]
        lang2 = [tokenizer_ch.vocab_size] + tokenizer_ch.encode(lang2) + [tokenizer_ch.vocab_size + 1]
        return [lang1, lang2]
    
    def filter_long_sent(x, y, max_length=MAX_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)
    
    def pad_with_zero(lang, max_length=MAX_LENGTH):
        lang1, lang2 = lang
        n1 = MAX_LENGTH - len(lang1)
        n2 = MAX_LENGTH - len(lang2)
        lang1 = lang1 + [0 for k in range(n1)]
        lang2 = lang2 + [0 for k in range(n2)]
        return [lang1, lang2]
    
    BUFFER_SIZE = 20000
    BATCH_SIZE = 64
    val_sentences = [
        k for k in val_examples
        if len(encode(k)[0]) <= MAX_LENGTH and len(encode(k)[1]) <= MAX_LENGTH
    ]
    train_examples = [encode(k) for k in train_examples]
    train_examples = [k for k in train_examples if len(k[0]) <= MAX_LENGTH and len(k[1]) <= MAX_LENGTH]
    train_examples = [pad_with_zero(k) for k in train_examples]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_examples)
    
    # 使用缓存数据加速读入
    train_dataset = train_dataset.cache()
    
    # 打乱并获取批数据
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    
    # 设置预取数据
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # # 验证集数据
    val_examples = [encode(k) for k in val_examples]
    val_examples = [k for k in val_examples if len(k[0]) <= MAX_LENGTH and len(k[1]) <= MAX_LENGTH]
    val_examples = [pad_with_zero(k) for k in val_examples]
    val_dataset = tf.data.Dataset.from_tensor_slices(val_examples)
    #处理完数据后，定义目标函数、评估指标以及模型：
    
    # 定义超参
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    input_vocab_size = tokenizer_en.vocab_size + 2
    target_vocab_size = tokenizer_ch.vocab_size + 2
    max_seq_len = MAX_LENGTH
    dropout_rate = 0.1
    
    # 定义优化器
    learing_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learing_rate, beta_1=0.9, 
                                        beta_2=0.98, epsilon=1e-9)
    
    # 定义目标函数和评估指标
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                               reduction='none')
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    # 创建模型
    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              max_seq_len, dropout_rate)
    build_transformer_variables(transformer, max_seq_len)
    
    checkpoint_path = os.path.join(script_dir, 'checkpoint', 'train_ch')
    ckpt = tf.train.Checkpoint(transformer=transformer,
                              optimizer=optimizer)
    # ckpt管理器
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
    
    if ckpt_manager.latest_checkpoint:
        is_compatible, reason = checkpoint_matches_model(
            ckpt_manager.latest_checkpoint, transformer
        )
        if is_compatible:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print('last checkpoit restore')
        else:
            print(
                'skip incompatible checkpoint {}: {}'.format(
                    ckpt_manager.latest_checkpoint, reason
                )
            )
    
    #训练模型：
    
    step_list = []
    loss_list = []
    step = 0
    # %%transformer.load_weights('transformer_model.h5')
    for epoch in range(EPOCHS):
        start = time.time()
    
        # 重置记录项
        train_loss.reset_state()
        train_accuracy.reset_state()
    
        # inputs 英语， targets 汉语
    
        for batch, all_inputs in enumerate(train_dataset):
            
            # 训练
            inputs = all_inputs[:, 0, :]
            targets = all_inputs[:, 1, :]
            train_step(inputs, targets)
    
            if batch % 100 == 0:
                loss = train_loss.result()
                print('epoch {}, batch {}, loss:{:.4f}, acc:{:.4f}'.format(
                    epoch+1, batch, loss, train_accuracy.result()
                ))
                step_list.append(step)
                loss_list.append(loss)
            step += 1
    
        if (epoch + 1) % 2 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('epoch {}, save model at {}'.format(
            epoch+1, ckpt_save_path
            ))
    
    
        print('epoch {}, loss:{:.4f}, acc:{:.4f}'.format(
            epoch+1, train_loss.result(), train_accuracy.result()
        ))
    
        print('time in 1 epoch:{} secs\n'.format(time.time()-start))
    plt.plot(step_list, loss_list)
    plt.xlabel('train step')
    plt.ylabel('loss')
    # transformer.save_weights('transformer_model.h5')
    
    #定义预测函数：
    def evaluate(inp_sentence):
        start_token = [tokenizer_en.vocab_size]
        end_token = [tokenizer_en.vocab_size + 1]
        
        # 输入语句是英语，增加开始和结束标记
        inp_sentence = start_token + tokenizer_en.encode(inp_sentence) + end_token
        if len(inp_sentence) > MAX_LENGTH:
            raise ValueError(
                'Encoded input length {} exceeds MAX_LENGTH {}.'.format(
                    len(inp_sentence), MAX_LENGTH
                )
            )
        encoder_input = tf.expand_dims(inp_sentence, 0)
    
        # 因为目标是汉语，输入 transformer 的第一个词应该是
        # 汉语的开始标记。
        decoder_input = [tokenizer_ch.vocab_size]
        output = tf.expand_dims(decoder_input, 0)
    
        for i in range(MAX_LENGTH):
            
            enc_padding_mask, combined_mask, dec_padding_mask = create_mask(encoder_input, output)
    
            predictions, attention_weights = transformer(
                encoder_input,
                output,
                training=False,
                encode_padding_mask=enc_padding_mask,
                look_ahead_mask=combined_mask,
                decode_padding_mask=dec_padding_mask,
            )
    
            # 从 seq_len 维度选择最后一个词
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
    
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
            # 如果 predicted_id 等于结束标记，就返回结果
            if int(predicted_id[0, 0].numpy()) == tokenizer_ch.vocab_size + 1:
                return tf.squeeze(output, axis=0), attention_weights
    
            # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
            output = tf.concat([output, predicted_id], axis=-1)
    
        return tf.squeeze(output, axis=0), attention_weights
        
    def translate(sentence, plot=''):
        result, attention_weights = evaluate(sentence)
        result_tokens = result.numpy().tolist()
        predicted_sentence = tokenizer_ch.decode(
            [i for i in result_tokens if i < tokenizer_ch.vocab_size]
        )
    
        print('输入: {}'.format(sentence))
        print('预测输出: {}'.format(predicted_sentence))
    #下面看一下效果：
    
    for sample_idx in [7, 9, 10, 12]:
        if sample_idx >= len(val_sentences):
            break
        s = val_sentences[sample_idx]
        translate(s[0])
        print("真实输出：" + s[1])
        print("***********")
# [EOF]
