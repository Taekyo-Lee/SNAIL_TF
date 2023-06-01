from SNAIL_tf import *
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dropout, MaxPool2D, Dense, Conv1D


class Conv2DBlock(Model):
    def __init__(self, num_filters=64, kernel_size=3, padding='same', activation='relu', batch_norm=True, dropout_rate=0.0, pool_size=2, **kwargs):
        super().__init__(**kwargs)
        self.num_filters, self.kernel_size, self.padding = num_filters, kernel_size, padding
        self.batch_norm, self.activation, self.dropout_rate, self.pool_size = batch_norm, activation, dropout_rate, pool_size 

        self.conv2d_layer = tf.keras.layers.Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, padding=self.padding)
        self.batch_norm_layer = tf.keras.layers.BatchNormalization()
        self.activation_layer = tf.keras.layers.Activation(self.activation)
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.max_pool_layer = tf.keras.layers.MaxPool2D(self.pool_size)
    
    def call(self, input, training=True):
        x = self.conv2d_layer(input)
        if self.batch_norm: x = self.batch_norm_layer(x, training=training)
        x = self.activation_layer(x) 
        if self.dropout_rate > 0.0: x = self.dropout_layer(x, training=training)
        output = self.max_pool_layer(x) 
        return output


class FCBlock(Model):
    def __init__(self, num_out_features=64, activation='linear', **kwargs):
        super().__init__(**kwargs)
        self.num_out_features, self.activation = num_out_features, activation
        self.fc_layer = Dense(units=self.num_out_features)
        self.activation_layer = tf.keras.layers.Activation(self.activation)

    def call(self, input):
        output = self.fc_layer(input)
        if self.activation != 'linear': output = self.activation_layer(output)
        return output



class OmniglotFeatureExtractor(Model):
    def __init__(self, num_conv_block=4, conv2d_num_filters=64, conv2d_kernel_size=3, conv2d_padding='same', conv2d_activation='relu', conv2d_batch_norm=True, conv2d_dropout_rate=0.0, conv2d_pool_size=2, fc_num_out_features=64, fc_activation='linear', **kwargs):
        super().__init__(**kwargs)
        self.num_conv_block, self.conv2d_num_filters, self.conv2d_kernel_size, self.conv2d_padding, self.conv2d_activation = num_conv_block, conv2d_num_filters, conv2d_kernel_size, conv2d_padding, conv2d_activation
        self.conv2d_batch_norm, self.conv2d_dropout_rate, self.conv2d_pool_size, self.fc_num_out_features, self.fc_activation = conv2d_batch_norm, conv2d_dropout_rate, conv2d_pool_size, fc_num_out_features, fc_activation
        self.conv2d_blocks = [Conv2DBlock(self.conv2d_num_filters, self.conv2d_kernel_size, self.conv2d_padding, self.conv2d_activation, self.conv2d_batch_norm, self.conv2d_dropout_rate, self.conv2d_pool_size, name=f'Feature extractor Block {i+1}: Conv2D') for i in range(self.num_conv_block)]
        self.fc_block = FCBlock(self.fc_num_out_features, self.fc_activation, name=f'Feature Extractor {len(self.conv2d_blocks)+1}: FC')

    def call(self, input, training=True):
        '''
        input_shape : (task_minibatch_size, NK+1, 28, 28, 1)
        output_shape : (task_minibatch_size, NK+1, -1)     
        '''
        task_minibatch_size, seq_len = input.shape[0], input.shape[1]
        input = tf.reshape(input, shape=(task_minibatch_size*seq_len, 28, 28, 1)) 
        x = input
        for conv2d_block in self.conv2d_blocks:
            x = conv2d_block(x)
        output = self.fc_block(tf.reshape(x, shape=(task_minibatch_size*seq_len, -1)))
        output = tf.reshape(output, shape=(task_minibatch_size, seq_len, -1))
        return output



def Get_SNAIL_For_Omniglot_Test(N, K):
    seq_len = N*K+1
    sanil_architecture = ({'attention':(64, 32)}, {'tc':128}, {'attention':(256, 128)}, {'tc': 128}, {'attention':(512, 256)})        
    sanil_output_layer = Conv1D(N, kernel_size=1, activation='softmax', name='N-way_softmax layer')
    return SNAIL(seq_len, sanil_architecture, sanil_output_layer)







if __name__=='__main__':
    task_minibatch_size = 2
    N, K = 5, 1
    omniglotfeatureextractor = OmniglotFeatureExtractor()
    snail_for_omniglot_test = Get_SNAIL_For_Omniglot_Test(N, K)


    omniglot_images = tf.random.normal(shape=(task_minibatch_size, N*K+1, 28, 28, 1))
    omniglot_features = omniglotfeatureextractor(omniglot_images)
    snail_output = snail_for_omniglot_test(omniglot_features, verbose=True)

    print(f'omniglot_images shape: {omniglot_images.shape}')
    print(f'omniglot_features shape: {omniglot_features.shape}')
    print(f'snail_output shape: {snail_output.shape}')    