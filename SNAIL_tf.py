"""
Taekyo Lee, 2023,
A Tensorflow implementation of the SANIL model 
from 'A Simple Neural Attentive Meta-Learner' by Nikhil Mishra et al. (https://arxiv.org/abs/1707.03141).
Permission is hereby granted, free of charge, to any person.
"""       


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Identity, Conv1D
import math, copy


class DenseBlock(Layer):
    def __init__(self, num_out_features, dilation_rate, **kwargs):
        super().__init__(**kwargs)
        self.filters = num_out_features
        self.dilation_rate = dilation_rate
        self.conv1d_1 = Conv1D(filters=self.filters, kernel_size=2, padding='causal', dilation_rate=self.dilation_rate)
        self.conv1d_2 = Conv1D(filters=self.filters, kernel_size=2, padding='causal', dilation_rate=self.dilation_rate)

    def call(self, input):
        """
        input_shape : (task_minibatch_size, seq_len, num_features)
        output_shape : (task_minibatch_size, seq_len, num_features+num_out_features) 
        """
        xf, xg = self.conv1d_1(input), self.conv1d_2(input)
        activations = tf.keras.activations.tanh(xf)*tf.keras.activations.tanh(xg)
        output = tf.concat([input, activations], axis=-1)
        return output


class TCBlock(Model):
    def __init__(self, seq_len, num_out_features, **kwargs):
        '''
        seq_len is required to determine the number of dilations
        '''
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.num_out_features = num_out_features
        self.tcblocks = self.get_tcblock()
        
    def get_tcblock(self):
        num_dilations = int(math.ceil(math.log(self.seq_len, 2)))
        return [DenseBlock(num_out_features=self.num_out_features, dilation_rate=int(math.pow(2, d)))  for d in range(1, num_dilations+1)]


    def call(self, input):
        """
        input_shape : (task_minibatch_size, seq_len, num_features)
        output_shape : (task_minibatch_size, seq_len, num_features+ceil(log_2_seq_len)*num_out_features)        
        """
        x = input
        for tcblock in self.tcblocks:
            x = tcblock(x)
        output = x        
        return output


class AttentionBlock(Model):
    def __init__(self, seq_len, key_dim, val_dim, **kwargs):
        '''
        This is a 'single head' attention block.
        seq_len is required to make mask
        '''
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.query_layer = Dense(self.key_dim)
        self.key_layer = Dense(self.key_dim)
        self.value_layer = Dense(self.val_dim)
        self.sqrt_k = math.sqrt(self.key_dim)
        self.mask = tf.where(tf.linalg.band_part(tf.ones(shape=(self.seq_len, self.seq_len)), -1, 0) == 0, float('-inf'), 0) 

    def call(self, input):
        """
        input_shape : (task_minibatch_size, seq_len, num_features)
        output_shape : (task_minibatch_size, seq_len, num_features+key_dim) 
        """
        queries = self.query_layer(input)
        keys = self.key_layer(input)
        values = self.value_layer(input)
        logits = tf.linalg.matmul(queries, keys, transpose_b=True)/self.sqrt_k
        logits += self.mask 
        probs = tf.nn.softmax(logits)
        read = tf.linalg.matmul(probs, values)
        output = tf.concat([input, read], axis=-1)
        return output



class SNAIL(Model):
    def __init__(self, seq_len, architecture, output_layer=None, **kwargs): 
        '''
        architecture : tuple of dictionaries, for example, ({'attention':(key_dim, val_dim)}, {'tc':num_out_features}, ...)
        '''
        super().__init__(**kwargs)        
        self.seq_len = seq_len
        self.architecture = architecture
        self.snail_layers = []
        for i, layer_info in enumerate(self.architecture):
            try:
                self.snail_layers.append( TCBlock(seq_len=self.seq_len, num_out_features=layer_info['tc'], name=f'SNAIL Block {i+1}: Tomporal convolution block') ) 
            except KeyError:
                self.snail_layers.append( AttentionBlock(seq_len=self.seq_len, key_dim=layer_info['attention'][0], val_dim=layer_info['attention'][1], name=f'SNAIL Block {i+1}: Singlehead attention block') )
        if output_layer:
            self.output_layer = copy.deepcopy(output_layer)
        else:
            self.output_layer = Identity(trainable=False)

                
    def call(self, input, verbose=False):
        '''
        input_shape : (task_minibatch_size, seq_len, input_fearures)
        output_shape : (task_minibatch_size, seq_len, -1)     
        '''
        x = input
        if verbose : print(f'SNAIL input_shape: {x.shape}')
        for i, layer in enumerate(self.snail_layers):
            x = layer(x)
            if verbose : print(f'SNAIL Block {i+1} output_shape: {x.shape}')
        output = self.output_layer(x)
        if verbose : print(f'SNAIL output_shape: {output.shape}\n')
        return output


if __name__=='__main__':
    task_minibatch_size = 2
    seq_len = 10
    sanil_architecture = ({'attention':(64, 32)}, {'tc':128}, {'attention':(256, 128)}, {'tc': 128}, {'attention':(512, 256)})     
    snail = SNAIL(seq_len, sanil_architecture)
    snail_input = tf.random.normal(shape=(task_minibatch_size, seq_len, 6))
    snail_output = snail(snail_input, verbose=True)