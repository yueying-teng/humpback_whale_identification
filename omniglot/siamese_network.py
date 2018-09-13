
import os

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
import keras.backend as K

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import data_loader
import modified_sgd



input_shape = (105, 105, 1)
model = []
# Path where the logs will be saved
tensorboard_log_path = './log'
summary_writer = tf.summary.FileWriter(tensorboard_log_path)

learning_rate = 10e-4
batch_size = 32

# Learning Rate multipliers for each layer
learning_rate_multipliers = {}
learning_rate_multipliers['Conv1'] = 1
learning_rate_multipliers['Conv2'] = 1
learning_rate_multipliers['Conv3'] = 1
learning_rate_multipliers['Conv4'] = 1
learning_rate_multipliers['Dense1'] = 1
# l2-regularization penalization for each layer
l2_penalization = {}
l2_penalization['Conv1'] = 1e-2
l2_penalization['Conv2'] = 1e-2
l2_penalization['Conv3'] = 1e-2
l2_penalization['Conv4'] = 1e-2
l2_penalization['Dense1'] = 1e-4


momentum = 0.9
# linear epoch slope evolution
momentum_slope = 0.01
support_set_size = 20
evaluate_each = 1000
number_of_train_iterations = 1000000





def __construct_siamese_architecture(learning_rate_multipliers, l2_regularization_penalization):
    """
    input:
    learning_rate_multipliers: to be applied to each conv and dense layer
    LR_mult_dict = {}
    LR_mult_dict['conv1']=1
    LR_mult_dict['conv2']=1
    LR_mult_dict['dense1']=2
    l2_regularization_penalization: L2 penalization for each layer
    L2_dictionary = {}
    L2_dictionary['conv1']=0.1
    L2_dictionary['conv2']=0.001
    L2_dictionary['dense1']=0.001
    """
    
    conv_net = Sequential()
    conv_net.add(Conv2D(64, (10, 10), activation = 'relu', input_shape = input_shape, kernel_regularizer = l2(l2_regularization_penalization['Conv1']), name = 'Conv1'))
    conv_net.add(MaxPool2D())
    
    conv_net.add(Conv2D(128, (7, 7), activation = 'relu', input_shape = input_shape, kernel_regularizer = l2(l2_regularization_penalization['Conv2']), name = 'Conv2'))
    conv_net.add(MaxPool2D())
        
    conv_net.add(Conv2D(128, (4, 4), activation = 'relu', input_shape = input_shape, kernel_regularizer = l2(l2_regularization_penalization['Conv3']), name = 'Conv3'))
    conv_net.add(MaxPool2D())
        
    conv_net.add(Conv2D(256,(4, 4), activation = 'relu', input_shape = input_shape, kernel_regularizer = l2(l2_regularization_penalization['Conv4']), name = 'Conv4'))
    conv_net.add(MaxPool2D())
    
    conv_net.add(Flatten())
    conv_net.add(Dense(4096, activation = 'sigmoid', kernel_regularizer = l2(l2_regularization_penalization['Dense1']), name = 'Dense1'))

    # pairs of images
    input_image_1 = Input(input_shape)
    input_image_2 = Input(input_shape)

    encoded_image_1 = conv_net(input_image_1)
    encoded_image_2 = conv_net(input_image_2)
    
    # L1 distance layer between the two encoded outputs
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])
    
    # predict if the two images are of the same class or not 
    prediction = Dense(1, activation = 'sigmoid')(l1_distance)
    model = Model([input_image_1, input_image_2], prediction)
    
    # define the optimizer and compile the model
    optimizer = modified_sgd.Modified_SGD(lr = learning_rate, lr_multipliers = learning_rate_multipliers, momentum = 0.5)
    
    model.compile(loss = 'binary_crossentropy', metrics = ['binary_accuracy'], optimizer = optimizer)
    
    return model


def __write_logs_to_tensorboard(current_iteration, train_losses, train_accuracies, validation_accuracy, evaluate_each):
    """
    logs are written to file after a certain number of iterations 
    current_iteration: iteration to be written to log file
    train_losses: training loss from the last evaluate_each iteration, where log is saved to file
    evaluate_each: number of iterations defined to evaluate the one-shot task
    """

    summary = tf.summary()
    for index in range(0, evaluate_each):
        value = summary.value.add()
        value.simple_value = train_losses[index]
        value.tag = 'Train Loss'
        
        value = summary.value.add()
        value.simple_value = train_accuracies[index]
        value.tag = 'Train Accuracy'
        
        if index == (evaluate_each - 1):
            value = summary.value.add()
            value.simple_value = validation_accuracy
            value.tag = 'One-Shot Validation Accuracy'
            
        summary_writer.add_summary(summary, current_iteration - evaluate_each + index + 1)
        summary_writer.flush()
        

model = __construct_siamese_architecture(learning_rate_multipliers = learning_rate_multipliers, l2_regularization_penalization = l2_penalization)


def train_siamese_network(number_of_iterations, support_set_size, final_momentum, momentum_slope, evaluate_each, model_name):
    """
    in every evaluate_each training iteration, evaluate one-shot task using validation and evaluation dataset
    
    support_set_size: number of different characters to use in one-shot task
    final_momentum: mu_j. each layer starts with a 0.5 momentum but evolves linearly to mu_j
    momentum_slop: slope of the momentum linearly evolution
    evaluation_each: number of iterations defined to evaluate the one-shot task
    model_name: name of the model to be saved as 
    """
    
    # train test split the 30 alphabets into training and validation
    # data_loader.split_train_datasets()
    # __train_alphabets, __validation_alphabets, __evaluation_alphabets = data_loader.split_train_datasets()
    
    # store 100 iterations of losses and accuracies, after evaluate_each iterations, these will be passed to tensorboard logs
    train_losses = np.zeros((evaluate_each))
    train_accuracies = np.zeros((evaluate_each))
    count = 0
    early_stop = 0 
    # stopping criteria variabels 
    best_validation_accuracy = 0.0
    best_accuracy_iteration = 0
    validation_accuracy = 0.0
    
    # training loop
    for iteration in range(number_of_iterations):
        # get training batches - same 
        images, labels = data_loader.get_train_batch()
        train_loss, train_accuracy = model.train_on_batch(images, labels)
        
        # learning rate decay, 1% per 500 iteration, and linear update of learning rate, from 0.5 to 1.
        if (iteration + 1) % 500 == 0:
            K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) * 0.99)

        if K.get_value(model.optimizer.momentum) < final_momentum:
            K.set_value(model.optimizer.momentum, K.get_value(model.optimizer.momentum) + momentum_slope)

        train_losses[count] = train_loss
        train_accuracies[count] = train_accuracy

        count += 1

        # validation 
        print ('Iteration %d/%d: Train loss: %f, Train Accuracy: %f, lr = %f' % (iteration + 1, number_of_iterations, train_loss, train_accuracy, K.get_value(model.optimizer.lr)))

        # perform one-shot task after evaluate_each iterations and write the result to tensorboard
        if (iteration + 1) % evaluate_each == 0:
            number_of_runs_per_alphabet = 40
            # use a support_set_size equal to the number of characters in the alphabet
            validation_accuracy = data_loader.one_shot_test(model, support_set_size, number_of_runs_per_alphabet, is_validation = True)

            __write_logs_to_tensorboard(iteration, train_losses, train_accuracies, validation_accuracy, evaluate_each)
            count = 0

            if (validation_accuracy == 1.0 and train_accuracy == 0.5):
                print('Early Stopping: Gradient Explosion')
                print('Validation Accuracy = ' + str(best_validation_accuracy))
                return 0

            elif train_accuracy == 0.0:
                return 0

            else:
                # save the model
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_accuracy_iteration = iteration

                    model_json = model.to_json()

                    if not os.path.exists('./models'):
                        os.makedirs('./models')
                    with open('models/' + model_name + '.json', "w") as json_file: json_file.write(model_json)
                    model.save_weights('models/' + model_name + '.h5')

        # If accuracy does not improve for 10000 batches stop the training
        if iteration - best_accuracy_iteration > 10000:
            print('Early Stopping: validation accuracy did not increase for 10000 iterations')
            print('Best Validation Accuracy = ' + str(best_validation_accuracy))
            print('Validation Accuracy = ' + str(best_validation_accuracy))
            break

    print('Trained Ended!')
    return best_validation_accuracy




