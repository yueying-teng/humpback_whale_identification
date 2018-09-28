## siamese network with omniglot data 



### omniglot data structure - alphabets, characters, examples 

train_dict
  - alphabet1_dict
    - character1_dict: list of image names belongs to character1
  - alphabet2_dcit
    - character2_dict: ...
    ...

same for testing data:

test_dict
  - alphabet1_dict 
    - character1_dict: list of image names belong to character1
  - ...
    - ...

If the features learned by the model are sufficient to confirm or deny the identity of characters from one set of alphabets, then they ought to be sufficient for other alphabets, provided that the model has been exposed to a variety of alphabets to encourage variance amongst the learned features.



### training
from the 30 alphabets background set, 80% (24) are used for training and 20% (6) are using for validation one-shot tasks

- training consists of classifying pairs of images to be the same or different characters. there are equal number of same and different character pairs for each alphabet
- in validation, the testing image is paired with each one of the support set images. the pair with the highest probability output is considered the class for the testing image.

weighted L1 distance function is learned between the embeddings. this is done by applying L1 distance on the output embeddings then adding one fully connected layer to learn the weighted distance. the loss function used is a regularized cross entropy, where the main aim is to drive similar samples to predict 1, and 0 otherwise. 



### testing
image_evaluation set: 20 alphabets, which is unseen to training data

for each alphabet, one randomly selected testing image is presented 20 images representing the potential unseen classes that can be chosen for the testing image, and these 20 images are the only known examples of each of those classes. 



### the way dataset is loaded from images_background and images_evaluation folders: load_dataset()
```python
train_dictionary = {}

for alphabet in [i for i in os.listdir(train_path) if '.DS_Store' not in i]:
    alphabet_path = os.path.join(train_path, alphabet)
    
    current_alphabet_dictionary = {}
    # character folders under each language 
    for character in [i for i in os.listdir(alphabet_path) if '.DS_Store' not in i]:
        character_path = os.path.join(alphabet_path, character)
        current_alphabet_dictionary[character] = os.listdir(character_path)

    train_dictionary[alphabet] = current_alphabet_dictionary
```



### generate batches of training data: get_train_batch()

each batch of training data is made of pairs of images. half of the pairs are made of the same character, while the other half are made of pairs from different characters. **characters in each batch are from the same alphabet**

get current alphabet, randomly select batch_size/2 characters under this alphabet. 

for each of the slected characters under this alphabet:
 - randomly select three images from this character, among which two images will be used as the same character pair.
 - randomly select one character from the remaining batch_size/2 characters under this alphabet, and select one image from this character. it will be used with the third image from the previous selection as the different character pair.
selection are done by randomly selecting indices and the stored images are actaully image paths, which will be returned as pairs of np arrays with corresponding labels 
even pairs are pairs of the same characters, while odd pairs are of different characters 



### generate batches of validation or testing data: get_one_shot_batch(support_set_size, is_validation)

each batch is made of support_set_size pairs images. for both validation and testing data, only the first pair is made of the same characters, while all the other images are of different characteres. **characters in each batch are still from the same alphabet**

- first select two images from one character under the current alphabet
- second select one image from charcters different from the one used in pair before. this number of images is decided by support_set_size
- reorder the images so that the pairs have the following fashion: (A, A), (A, B), (A, C), (A, D)...



### evaluating validation or testing accuracy: one_shot_test(model, support_set_size, number_of_tasks_per_alphabet, is_validation)

 - for each alphabet in the testing set of alphabets, generate batches of character classes that are made of differnt character pairs apart from the first pair and return these unseen pairs of characters as np arrays and corresponding labels 
 - test the model on the images and labels just generated

```python
for alphabet in alphabets:
    mean_alphabet_accuracy = 0
    for _ in range(number_of_tasks_per_alphabet):
        images, _ = get_one_shot_batch(support_set_size, is_validation = is_validation)
        probabilities = model.predict_on_batch(images)
        # print mean_alphabet_accuracy and mean_global_accuracy
```



### train netwrok with validation during training: train_siamese_network(number_of_iterations, support_set_size, final_momentum, momentum_slope, evaluate_each, model_name)

```python
for iteration in range(number_of_iterations):
    images, labels = data_loader.get_train_batch()
    train_loss, train_accuracy = model.train_on_batch(images, labels)
    # learning rate decay
    # print training loss and accuracy

    # validation after evaluate_each number of iterations
    if (iteration + 1) % evaluate_each == 0:
        validation_accuracy = data_loader.one_shot_test(model, support_set_size, number_of_runs_per_alphabet, is_validation = True)
        # save the model has the best validation accuracy so far
```



### test the trained model on testing data: one_shot_test(model, support_set_size, number_of_tasks_per_alphabet, is_validation = False)



### references 
- https://github.com/Goldesel23/Siamese-Networks-for-One-Shot-Learning

- https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

