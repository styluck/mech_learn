# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:51:58 2024

@author: lich5
"""
import numpy as np
import tensorflow as tf
try:
    from .self_attention import scale_self_attention, scale_self_attention1
except ImportError:
    from self_attention import scale_self_attention, scale_self_attention1

# 构造multi head attention层
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, att =False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.att = att
        # d_model 必须可以正确分为各个头
        assert d_model % num_heads == 0
        # 分头后的维度
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key=None, value=None, attention_mask=None):
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size = tf.shape(query)[0]

        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(query)  # (batch_size, seq_len, d_model)
        k = self.wk(key)
        v = self.wv(value)

        # 分头
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        # 通过缩放点积注意力层
        if self.att:
            scaled_attention, attention_weights = scale_self_attention(
        q, k, v, attention_mask)
        else:
            scaled_attention, attention_weights = scale_self_attention1(
        q, k, v, attention_mask)
            
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3]) # (batch_size, seq_len_v, num_heads, depth)

        # 合并多头
        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))

        # 全连接重塑
        output = self.dense(concat_attention)
        return output, attention_weights
    
    
if __name__ == '__main__':
    
    
# 测试Multi-Head Attention
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))
    output, att = temp_mha(y, key=y, value=y, attention_mask=None)
    print(output.shape, att.shape)

    
    
    
# [EOF]
    

