from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, activation=None))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    #simp_rnn = SimpleRNN(units, activation='relu',
    #    return_sequences=True, name='rnn')(bn_cnn)
    simp_rnn = GRU(units, activation='tanh',
        return_sequences=True, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, activation=None))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    x = input_data    
    for i in range(recur_layers):
        x = GRU(units, activation="tanh", return_sequences=True, name=f'gru_{i + 1}')(x) # tanh is best suited for GRU
        
        x = BatchNormalization(name=f'bn_{i + 1}')(x)        
        units = int(units / 2)
        
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, activation=None))(x)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    birn_rnn = Bidirectional(GRU(units, activation="tanh",
        return_sequences=True, name='birnn'))(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(birn_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, activation=None))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, SimpleRNN, GRU, TimeDistributed, Dense, Activation, Bidirectional, MaxPooling1D
from tensorflow.keras.regularizers import l2

def final_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, rnn_units, output_dim, 
                num_bidirectional_layers=1, rnn_type='gru', use_rnn_dropout=False):
    """
    Build a deep network for speech analysis on the LibriSpeech dataset.

    Parameters:
    - input_dim (int): The number of features in the input (e.g., frequency bins from the spectrogram).
    - filters (int): Number of filters in Conv1D layers.
    - kernel_size (int): Kernel size in Conv1D layers.
    - conv_stride (int): Stride for Conv1D layers.
    - conv_border_mode (str): Padding mode for Conv1D ('same' or 'valid').
    - rnn_units (int): Number of units in the RNN layers.
    - output_dim (int): Number of output classes (e.g., 29 for the LibriSpeech dataset).
    - num_bidirectional_layers (int): Number of Bidirectional RNN layers.
    - rnn_type (str): Type of RNN layer to use ('gru' or 'simplernn').
    - use_rnn_dropout (bool): Whether to apply dropout in the RNN layers.

    Returns:
    - model: Compiled Keras model ready for training.
    """
    
    # Main acoustic input (time, frequency bins)
    input_data = Input(name='the_input', shape=(None, input_dim))

    # First Conv1D layer
    conv_1d = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, 
                     activation='relu', name='conv1d')(input_data)
    
    # BatchNormalization after the Conv1D
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)

    # Add Bidirectional RNN layers as specified
    rnn_input = bn_cnn  # The input for the RNN layer comes from the conv layer
    for i in range(num_bidirectional_layers):
        if rnn_type == 'gru':
            rnn_layer = GRU(rnn_units, activation='tanh', return_sequences=True, 
                            dropout=0.2 if use_rnn_dropout else 0.0, kernel_regularizer=l2(1e-4))
        elif rnn_type == 'simplernn':
            rnn_layer = SimpleRNN(rnn_units, activation='tanh', return_sequences=True, dropout=0.2 if use_rnn_dropout else 0.0)
        else:
            raise ValueError("rnn_type must be either 'gru' or 'simplernn'.")

        # Wrap in a Bidirectional layer
        rnn_bi = Bidirectional(rnn_layer, name=f'bidirectional_rnn_{i+1}')(rnn_input)

        # BatchNormalization after each Bidirectional RNN layer
        rnn_input = BatchNormalization(name=f'bn_rnn_{i+1}')(rnn_bi)

        rnn_units = int(rnn_units / 2)

    # TimeDistributed Dense layer for output
    time_dense = TimeDistributed(Dense(output_dim))(rnn_input)

    # Softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Define the model
    model = Model(inputs=input_data, outputs=y_pred)

    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride, dilation=1)   
    
    # Print the model summary
    print(model.summary())

    return model