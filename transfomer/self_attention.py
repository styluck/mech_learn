# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:38:47 2024

@author: lich5
"""

import tensorflow as tf
import numpy as np


def get_angles(pos, i, d_model):
    
    angle_rates = 1 / np.power(10000, (2*(i // 2))/ np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    '''
    # 位置Embedding
    '''
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis,:],
                           d_model)
    # 第2i项使用sin
    sines = np.sin(angle_rads[:, 0::2])
    # 第2i+1项使用cos
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# 掩码（mask）
def create_padding_mask(seq):
    '''
    # 为了避免输入中padding的token对句子语义的影响，需要将padding
    # 位mask掉，原来为0的padding项的mask输出为1
    # 获取为0的padding项
    '''
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 扩充维度以便用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :] # (batch_size,1,1,seq_len)


def create_look_ahead_mask(size):
    '''
    # look-ahead mask 
    # 用于对未预测的token进行掩码，这意味着要预测第三个单词，只会使用第
    # 一个和第二个单词。 要预测第四个单词，仅使用第一个，第二个和第三个单
    #词，依此类推。

    # 1 - 对角线和取下三角的全部对角线（-1->全部）
    # 这样就可以构造出每个时刻未预测token的掩码
    '''
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

# 构建掩码
def create_mask(inputs,targets):
    encode_padding_mask = create_padding_mask(inputs)
    # 这个掩码用于掩输入解码层第二层的编码层输出
    decode_padding_mask = create_padding_mask(inputs)

    # look_ahead 掩码， 掩掉未预测的词
    look_ahead_mask = create_look_ahead_mask(tf.shape(targets)[1])
    # 解码层第一层得到padding掩码
    decode_targets_padding_mask = create_padding_mask(targets)

    # 合并解码层第一层掩码
    combine_mask = tf.maximum(decode_targets_padding_mask, look_ahead_mask)

    return encode_padding_mask, combine_mask, decode_padding_mask


def scale_self_attention(q, k, v, mask):
    '''
    # query key 相乘获取匹配关系
    '''
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 掩码
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 通过softmax获取attention权重
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # attention 乘上value
    output = tf.matmul(attention_weights, v) # （.., seq_len_v, depth）

    return output, attention_weights


def scale_self_attention1(q, k, v, mask):
    '''
    # query key 相乘获取匹配关系
    '''
    
    if mask is not None:
        matmul_qk = tf.matmul(q, k, transpose_b=True)
    
        # 使用dk进行缩放
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)


        # 掩码
        scaled_attention_logits += (mask * -1e9)
        # 通过softmax获取attention权重
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
        # attention 乘上value
        output = tf.matmul(attention_weights, v)  # （.., seq_len_v, depth）

    else:
        
        matmul_qk = tf.matmul(q, k, transpose_a=True)

        # 使用dk进行缩放
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # 通过softmax获取attention权重
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
        # attention 乘上value
        output = tf.matmul(v, attention_weights) # （.., seq_len_v, depth）

    return output, attention_weights


if __name__ == '__main__':

    # 测试位置编码：
    pos_encoding = positional_encoding(50, 512)
    print(pos_encoding.shape)

    # mask 测试
    print(create_padding_mask([[1,2,0,0,0],[3,4,5,0,0]]))

    # x = tf.random.uniform((1,3))
    temp = create_look_ahead_mask(3)

    # self-attention测试
    np.set_printoptions(suppress=True)
    
    def print_out(q, k, v):
        temp_out, temp_att = scale_self_attention1(
        q, k, v, None)
        print('attention weight:')
        print(temp_att)
        print('output:')
        print(temp_out)
    
    batch = 32
    n = 16
    d = 8
    k = tf.constant([np.random.randn(batch, n, d)], dtype=tf.float32)  # (4, 3)
    v = tf.constant([np.random.randn(batch, n, d)], dtype=tf.float32)  # (4, 3)
    q = tf.constant([np.random.randn(batch, n, d)], dtype=tf.float32)  # (3, 3)
    print_out(k,v,q)
    
    def softmax(x):
        # Subtract the max for numerical stability
        x = x - np.max(x, axis=-1, keepdims=True)
        # Compute the exponentials
        exp_x = np.exp(x)
        # Compute the sum of exponentials
        sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
        # Normalize by dividing the exponentials by their sum
        softmax_x = exp_x / sum_exp_x
        return softmax_x

    scaled_attention_logits = np.random.randn(batch, batch)
    mask = np.triu(np.ones((batch, batch)), k=1)
    scaled_attention_logits += (mask * -1e9)
    attention_weights = softmax(scaled_attention_logits)
    # output = 
