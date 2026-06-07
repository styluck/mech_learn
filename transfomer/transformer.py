# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:25:46 2024

@author: lich5
"""
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
try:
    from .encoderdecoder import Encoder, Decoder
except ImportError:
    from encoderdecoder import Encoder, Decoder


# 创建Transformer
class Transformer(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, diff,
                input_vocab_size, target_vocab_size,
                max_seq_len, drop_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_layers, d_model, n_heads,diff,
                              input_vocab_size, max_seq_len, drop_rate)

        self.decoder = Decoder(n_layers, d_model, n_heads, diff,
                              target_vocab_size, max_seq_len, drop_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    def call(self, inputs, targets, training=False, encode_padding_mask=None,
             look_ahead_mask=None, decode_padding_mask=None):

        encode_out = self.encoder(
            inputs,
            training=training,
            attention_mask=encode_padding_mask,
        )
        # print("encode_out", encode_out.shape)
        decode_out, att_weights = self.decoder(
            targets,
            encoder_out=encode_out,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decode_padding_mask,
        )
        # print("decode_out", decode_out.shape)
        final_out = self.final_layer(decode_out)

        return final_out, att_weights


if __name__ == '__main__':
    
    sample_transformer = Transformer(
        n_layers=2, 
        d_model=512, 
        n_heads=8, 
        diff=1024,
        input_vocab_size=8500, 
        target_vocab_size=8000, 
        max_seq_len=120
    )
    
    temp_input = tf.random.uniform((64, 62))
    temp_target = tf.random.uniform((64, 26))
    
    fn_out, _ = sample_transformer(
        temp_input, 
        temp_target, 
        training=False,
        encode_padding_mask=None,
        look_ahead_mask=None,
        decode_padding_mask=None)
    
    fn_out.shape

# [EOF]


















































# [EOF]
