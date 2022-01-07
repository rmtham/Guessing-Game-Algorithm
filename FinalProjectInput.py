import pandas as pd
import numpy as np

# Import the attributes of the corresponding animals
animal_matrix = pd.read_csv('zoodata_small_matrix.csv')
# Import the names of the animals from file
animal_names = pd.read_csv('zoodata_small_concepts.csv', names=['animal'])
# Import the prior probabilities of the animals
prior_probabilities = pd.read_csv('zoodata_small_priorprobs.csv', names=['probabilities'])
# Import animal class of animal
animal_class = pd.read_csv('zoodata_small_classtypes.csv', names=['class'])
# Add animal_names and prior probabilities to animal_matrix
animal_matrix['animal'] = animal_names
animal_matrix['probabilities'] = prior_probabilities
animal_matrix['class'] = animal_class
# Initialize vector that will store user's responses
user_responses = []
all_animal_types = list(set(animal_matrix['class'].tolist()))
iteration_num = 0
continue_playing = True

# Iterate through all of the animals in the animal matrix to determine
# the number of questions it takes to "find" an animal
while continue_playing:
    # Reset filtered_animal matrix, animal_not_found, and num_questions for the new animal
    filtered_animal_matrix = animal_matrix.copy()
    animal_not_found = True
    num_questions = 0
    iteration_num += 1

    # Ask the user to input the animal they are thinking of
    print()
    print(animal_matrix['animal'])
    animal = int(input("Please input the number corresponding to the animal you are thinking of: "))

    if iteration_num > 30:
        for animal_class in all_animal_types:
            # Calculate and assign the updated probabilities for each animal
            # class, based on the user's responses once the 30th iteration is reached
            animal_type_rows = filtered_animal_matrix[filtered_animal_matrix['class'] == animal_class]
            animal_type_indices = animal_type_rows.index.values.tolist()
            animal_instances_user = len([i for i in user_responses[0:iteration_num] if (i in animal_type_indices)])
            user_probability = animal_instances_user / len(user_responses[0:iteration_num])
            animal_instances_matrix = len(animal_type_rows)
            final_animal_probability = user_probability / animal_instances_matrix
            filtered_animal_matrix.loc[filtered_animal_matrix['class'] == animal_class, 'probabilities'] = final_animal_probability

    while animal_not_found:
        # While the animal has not been found yet, reset questions_IG, which stores the IG
        # for each question, probabilities_yes_list, which stores the probabilities if the animal
        # has the feature, and probabilities_no_list, which stores the probabilities  if the animal doesn't
        questions_IG = []
        probabilities_yes_list = []
        probabilities_no_list = []

        # Find the question with the greatest Information Gain. This for-loop iterates through
        # all of the questions and calculates the Information Gain for each of them
        for col_number in range(len(filtered_animal_matrix.columns) - 3):
            question_string = filtered_animal_matrix.columns[col_number]

            # (Part 1): Find the entropy value
            entropy_column = -(filtered_animal_matrix['probabilities'] * np.log2(filtered_animal_matrix['probabilities'],
                               where=filtered_animal_matrix['probabilities'] > 0))
            entropy = entropy_column.sum()

            # (Part 2): Find the probabilities that the animal has or does not have the feature
            # (i.e., the probabilities that a yes or no is the answer to the question, respectively)
            prob_a_yes = np.nansum(filtered_animal_matrix[question_string] * filtered_animal_matrix['probabilities'])
            prob_a_no = np.nansum((1 - filtered_animal_matrix[question_string]) * filtered_animal_matrix['probabilities'])

            # (Part 3): Find the new uncertainties after learning the answer to the question is
            # yes or no, respectively

            # Step 1 of Part 3: Calculate the new probabilities of the animals, based on whether
            # the question is yes or no
            new_probability_yes = (filtered_animal_matrix[question_string] * filtered_animal_matrix['probabilities']) / \
                                   np.nansum(filtered_animal_matrix[question_string] * filtered_animal_matrix['probabilities'])
            new_probability_yes.columns = ['new probability yes']
            new_probability_no = ((1 - filtered_animal_matrix[question_string]) * filtered_animal_matrix['probabilities']) /\
                                   np.nansum((1 - filtered_animal_matrix[question_string]) * filtered_animal_matrix['probabilities'])
            new_probability_no.columns = ['new probability no']

            probabilities_yes_list.append(new_probability_yes)
            probabilities_no_list.append(new_probability_no)

            # Step 2 of Part 3: Calculate P(x)log2P(x) for probabilities that are greater than zero
            logarithm_yes = -(new_probability_yes * np.log2(new_probability_yes, where=new_probability_yes > 0))
            logarithm_no = -(new_probability_no * np.log2(new_probability_no, where=new_probability_no > 0))

            # Step 3 of Part 3: Find the new uncertainties after learning the answer to the question is
            # yes or no, respectively
            uncertainty_q_yes = logarithm_yes.sum()
            uncertainty_q_no = logarithm_no.sum()

            # (Part 4): Find the Information Gain for this particular question
            question_IG = entropy - ((prob_a_yes * uncertainty_q_yes) + (prob_a_no * uncertainty_q_no))
            questions_IG.append(question_IG)

        # Determine the index corresponding to the question with the maximum Information Gain
        index = questions_IG.index(max(questions_IG))
        # Determine the column name corresponding to the question with the maximum Information Gain
        col_name = filtered_animal_matrix.columns[index]
        animal_col_index = 8
        animal_col = filtered_animal_matrix['animal']
        selected_animal_name = animal_col[animal]

        features_col = filtered_animal_matrix[col_name].copy()
        selected_animal_feature = features_col.loc[animal]

        if selected_animal_feature == 1:
            # If the animal has the feature for the selected question, then update the
            # probabilities to the probability_yes values and drop animals that have a
            # probability of 0
            probability_yes = probabilities_yes_list[index].copy()
            filtered_animal_matrix.loc[:, 'probabilities'] = probability_yes.values
            condition = filtered_animal_matrix[col_name] == 1
            filtered_animal_matrix = filtered_animal_matrix[condition]
            filtered_animal_matrix.drop(filtered_animal_matrix[filtered_animal_matrix['probabilities'] == 0].index,
                                        inplace=True)
        elif selected_animal_feature == 0:
            # If the animal does not have the feature for the selected question, then update
            # the probabilities to the probability_no values and drop animals that have a
            # probability of 0
            probability_no = probabilities_no_list[index].copy()
            filtered_animal_matrix.loc[:, 'probabilities'] = probability_no.values
            condition = filtered_animal_matrix[col_name] == 0
            filtered_animal_matrix = filtered_animal_matrix[condition]
            filtered_animal_matrix.drop(filtered_animal_matrix[filtered_animal_matrix['probabilities'] == 0].index,
                                        inplace=True)

        num_questions += 1
        print(f"This is question number {num_questions}: ", col_name)

        if len(filtered_animal_matrix.index) == 1:
            # If the number of rows in the filtered_animal_matrix is one, then
            # we break out of the while loop, since we have found the animal
            animal_not_found = False

    print("This is the total number of questions it took to identify the animal: ", num_questions)
    animal_col_index = 8
    print("The animal is a: ", animal_matrix.iloc[animal][animal_col_index])
    print(f"This this is animal #{iteration_num}")
    print()
    # Ask the user if they would like to play another round. If yes, continue
    # if no, break out of the while loop
    response = int(input("Would you like to play another round? (Yes = 1, No = 0): "))
    if response == 0:
        continue_playing = False
    elif response == 1:
        continue_playing = True
