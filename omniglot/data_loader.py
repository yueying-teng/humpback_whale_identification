import os
import random 
import numpy as np
import math
from PIL import Image



dataset_path = './omniglot_data/'
train_dictionary = {}
evaluation_dictionary = {}
image_width = 105
image_height = 105
# batch_size = batch_size
# use_augmentation = use_augmentation
__train_alphabets = []
__validation_alphabets = []
__evaluation_alphabets = []
__current_train_alphabet_index = 0
__current_validation_alphabet_index = 0
__current_evaluation_alphabet_index = 0

batch_size = 32



# load_dataset 
train_path = os.path.join(dataset_path, 'images_background')
validation_path = os.path.join(dataset_path, 'images_evaluation')

# training alphabets 
# language folders 
for alphabet in [i for i in os.listdir(train_path) if '.DS_Store' not in i]:
    alphabet_path = os.path.join(train_path, alphabet)
    
    current_alphabet_dictionary = {}
    # character folders under each language 
    for character in [i for i in os.listdir(alphabet_path) if '.DS_Store' not in i]:
        character_path = os.path.join(alphabet_path, character)
        current_alphabet_dictionary[character] = os.listdir(character_path)

    train_dictionary[alphabet] = current_alphabet_dictionary
    
    
# validation alphabets 
for alphabet in [i for i in os.listdir(validation_path) if '.DS_Store' not in i]:
    alphabet_path = os.path.join(validation_path, alphabet)
    
    current_alphabet_dictionary = {}
    # character folders under each language 
    for character in [i for i in os.listdir(alphabet_path) if '.DS_Store' not in i]:
        character_path = os.path.join(alphabet_path, character)
        current_alphabet_dictionary[character] = os.listdir(character_path)

    evaluation_dictionary[alphabet] = current_alphabet_dictionary


# split_train_datasets
# train validation split 24 vs 6

available_alphabets = list(train_dictionary.keys())
number_of_alphabets = len(available_alphabets)

train_indexes = random.sample(range(0, number_of_alphabets -1), int(0.8* number_of_alphabets))

# sort indexes in reverse order and pop them out from the list so that the original index order is not changed
train_indexes.sort(reverse = True)
for index in train_indexes:
    __train_alphabets.append(available_alphabets[index])
    available_alphabets.pop(index)

__validation_alphabets = available_alphabets
__evaluation_alphabets = list(evaluation_dictionary.keys())


def __convert_path_list_to_images_and_labels (path_list, is_one_shot_task):
    """
    return image in np.array format and corresponding label from the image path 
    input:
    path_list: list of images to be loaded in this batch
    is_one_shot_task: if the batch is for one-shot task or for training 
    return:
    pairs_of_images: pairs of images for the current batch
    labels: corresponding labels 1 for same class, 0 for different classes 
    """
    
    path_list = [i for i in path_list if '.DS_Store' not in i]
    
    number_of_pairs = int(len(path_list)/ 2)
    pairs_of_images = [np.zeros((number_of_pairs, image_width, image_height, 1)) for i in range(2)]
    labels = np.zeros((number_of_pairs, 1))

    for pair in range(number_of_pairs):
        image = Image.open(path_list[pair* 2])
        image = np.asarray(image).astype(np.float64)
        image = image/ image.std() - image.mean() 
        # for the first image in the pair
        pairs_of_images[0][pair, :, :, 0] = image

        image = Image.open(path_list[pair* 2 + 1])
        image = np.asarray(image).astype(np.float64)
        image = image/ image.std() - image.mean()  
        # second image in the pair
        pairs_of_images[1][pair, :, :, 0] = image

        if not is_one_shot_task:
            # training and validation set 
            if (pair + 1)% 2 == 0:
                # pair 1, 3, 5 ... are pairs of different characters 
                labels[pair] = 0
            else:
                # pair 0, 2, 4 ... are pairs of the same character
                labels[pair] = 1

        else:
            # evaluation set - only the first set has label 1
            if pair == 0:
                labels[pair] = 1
            else:
                labels[pair] = 0

    if not is_one_shot_task:
        # permute only when it's training task 
        random_permutation = np.random.permutation(number_of_pairs)
        labels = labels[random_permutation]
        pairs_of_images[0][:, :, :, :] = pairs_of_images[0][random_permutation, :, :, :]
        pairs_of_images[1][:, :, :, :] = pairs_of_images[1][random_permutation, :, :, :]

    
    return pairs_of_images, labels


def get_train_batch():
    """
    generate batches of training data. 
    returns:
    pairs_of_images: pairs of images for the current batch
    labels: 1 for the same character and 0 for different characters 
    """
    
    global __current_train_alphabet_index
    
    current_alphabet = __train_alphabets[__current_train_alphabet_index]
    available_characters = list(train_dictionary[current_alphabet].keys())
    number_of_characters = len(available_characters)

    batch_images_path = []
    
    # if the number of characters for this alphabet is less than batch_size/2, repeat the available characters 
    selected_characters_indexes = [random.randint(0, number_of_characters - 1) for i in range(int(batch_size/ 2))]
    
    for index in selected_characters_indexes:
        current_character = available_characters[index]
        available_images = (train_dictionary[current_alphabet])[current_character]
        image_path = os.path.join(dataset_path, 'images_background', current_alphabet, current_character)
            
        # randomly select 3 indexes from the same character (note: there are only 20 samples for each character)
        image_indexes = random.sample(range(0, 19), 3)
        image = os.path.join(image_path, available_images[image_indexes[0]])
        batch_images_path.append(image)
        image = os.path.join(image_path, available_images[image_indexes[1]])
        batch_images_path.append(image)

        # for images from different characters, but the same alphabet
        image = os.path.join(image_path, available_images[image_indexes[2]])
        batch_images_path.append(image)

        different_characters = available_characters[:]
        different_characters.pop(index)
        different_character_index = random.sample(range(0, number_of_characters -1), 1)

        current_character = different_characters[different_character_index[0]]
        available_images = (train_dictionary[current_alphabet])[current_character]
        image_indexes = random.sample(range(0, 19), 1)
        image_path = os.path.join(dataset_path, 'images_background', current_alphabet, current_character)
        image = os.path.join(image_path, available_images[image_indexes[0]])
        batch_images_path.append(image)
    
    __current_train_alphabet_index += 1
    
    # train validation split ratio is 24 vs 6, i.e. length of __train_alphabets is 24
    if __current_train_alphabet_index > 23:
        __current_train_alphabet_index = 0
     
    # image here is pairs of images 
    images, labels = __convert_path_list_to_images_and_labels(batch_images_path, is_one_shot_task = False)
    
    # todo: get random transformation if augmentation is on 
    
    return images, labels 


def get_one_shot_batch(support_set_size, is_validation):
    """
    generate one shot batches for evaluation or validation - only the first pair is of the same character 
    input:
    support_set_size: number of **characters** to use in the support set for one-shot tasks, evaluation tasks
    is_validation: validation or testing data
    return:
    images in np.arary format and their corresponding labels
    """
    global __current_validation_alphabet_index
    global __current_evaluation_alphabet_index
    
    # variables dependent on the data 
    if is_validation:
        alphabets = __validation_alphabets
        current_alphabet_index = __current_validation_alphabet_index
        image_folder_name = 'images_background'
        dictionary = train_dictionary
    else:
        alphabets = __evaluation_alphabets
        current_alphabet_index = __current_evaluation_alphabet_index
        image_folder_name = 'images_evaluation'
        dictionary = evaluation_dictionary
        
    current_alphabet = alphabets[current_alphabet_index]
    available_characters = list(dictionary[current_alphabet].keys())
    number_of_characters = len(available_characters)
    
    batch_images_path = []
    
    test_character_index = random.sample(range(0, number_of_characters), 1)
    # get testing image pairs of the same characters first 
    current_character = available_characters[test_character_index[0]]
    available_images = (dictionary[current_alphabet])[current_character] 
    image_indexes = random.sample(range(0, 19), 2)
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
    # for testing alphabets with fewer than 20 characters 
    if number_of_characters < number_of_support_characters:
        number_of_support_characters = number_of_characters
        
    support_characters_indexes = random.sample(range(0, number_of_characters -1), number_of_support_characters - 1)
    
    for index in support_characters_indexes:
        current_character = different_characters[index]
        available_images = (dictionary[current_alphabet])[current_character]
        image_path = os.path.join(dataset_path, image_folder_name, current_alphabet, current_character)
        image_indexes = random.sample(range(0, 19), 1)
        image = os.path.join(image_path, available_images[image_indexes[0]])
        batch_images_path.append(image)
        
    images, labels = __convert_path_list_to_images_and_labels(batch_images_path, is_one_shot_task = True)
    
    return images, labels


def one_shot_test(model, support_set_size, number_of_tasks_per_alphabet, is_validation):
    
    """
    one shot task performance evaluation
    input:
    support_set_size: number of **characters** to use in the support set for one-shot tasks
    number_of_tasks_per_alphabet:
    return:
    mean_accuracy

    """
    # variables dependent on the data 
    if is_validation:
        alphabets = __validation_alphabets
        print ('\none shot on validation alphabets:')
    
    else:    
        alphabets = __evaluation_alphabets
        print ('\none shot on evaluation alphabets:')
        
    mean_global_accuracy = 0
    
    for alphabet in alphabets:
        mean_alphabet_accuracy = 0
        for _ in range(number_of_tasks_per_alphabet):
    
    
