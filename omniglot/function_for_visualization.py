import data_loader
import random 
import numpy as np
import os


dataset_path = './omniglot_data/'
train_dictionary, evaluation_dictionary = data_loader.load_dataset()
__train_alphabets, __validation_alphabets, __evaluation_alphabets = data_loader.split_train_datasets()


# testing using image_evaluation set

def get_one_shot_batch_image_path(support_set_size, is_validation, current_alphabet_index):
    """
    generate one shot batches for evaluation or validation - only the first pair is of the same character 
    
    input:
    support_set_size: number of **characters** to use in the support set for one-shot tasks, evaluation tasks
    current_alphabet_index: index of the alphabet what to study now, maximum is 20 
    is_validation: validation or testing data
    return:
    list of image paths as one batch
    """

    # variables dependent on the data 
    if is_validation:
        alphabets = __validation_alphabets
        image_folder_name = 'images_background'
        dictionary = train_dictionary
    else:
        alphabets = __evaluation_alphabets
        image_folder_name = 'images_evaluation'
        dictionary = evaluation_dictionary
        
    current_alphabet = alphabets[current_alphabet_index]
    available_characters = list(dictionary[current_alphabet].keys())
    number_of_characters = len(available_characters)
    
    batch_images_path = []
    ordered_batch_images_path = []
    
    test_character_index = random.sample(range(0, number_of_characters), 1)
    # get testing image pairs of the same characters first 
    current_character = available_characters[test_character_index[0]]
    available_images = (dictionary[current_alphabet])[current_character] 
    image_indexes = random.sample(range(0, 20), 2)
    image_path = os.path.join(dataset_path, image_folder_name, current_alphabet, current_character)
    
    # first image
    test_image = os.path.join(image_path, available_images[image_indexes[0]])
    batch_images_path.append(test_image)
    # second 
    image = os.path.join(image_path, available_images[image_indexes[1]])
    batch_images_path.append(image)
        
    # get the pairs of images of differnt characters 
    if support_set_size == -1:
        number_of_support_characters = number_of_characters
    else:
        number_of_support_characters = support_set_size
        
    different_characters = available_characters[:]
    different_characters.pop(test_character_index[0])
    # for testing alphabets with fewer than 20 characters, all characters are used as supporting characters 
    if number_of_characters < number_of_support_characters:
        number_of_support_characters = number_of_characters
        
    support_characters_indexes = random.sample(range(0, number_of_characters -1), number_of_support_characters - 1)
    
    for index in support_characters_indexes:
        current_character = different_characters[index]
        available_images = (dictionary[current_alphabet])[current_character]
        image_path = os.path.join(dataset_path, image_folder_name, current_alphabet, current_character)
        image_indexes = random.sample(range(0, 20), 1)
        image = os.path.join(image_path, available_images[image_indexes[0]])
        batch_images_path.append(image)

    for i in range(len(batch_images_path)):
        ordered_batch_images_path.append(batch_images_path[0])
        ordered_batch_images_path.append(batch_images_path[i])

    batch_images_path = ordered_batch_images_path[2: ]

    return batch_images_path

