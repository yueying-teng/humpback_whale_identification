import data_loader 
import modified_sgd 
import siamese_network


validation_accuracy = siamese_network.train_siamese_network(number_of_iterations = number_of_train_iterations, support_set_size = support_set_size, final_momentum = momentum, \
                                            momentum_slope = momentum_slope, evaluate_each = evaluate_each, model_name = 'siamese_net_lr10e-4')
if validation_accuracy == 0:
    evaluation_accuracy = 0
else:
    # Load the weights with best validation accuracy
    model.load_weights('.models/siamese_net_lr10e-4.h5')
    evaluation_accuracy = data_loader.one_shot_test(model, 20, 40, False)

print('Final Evaluation Accuracy = ' + str(evaluation_accuracy))
