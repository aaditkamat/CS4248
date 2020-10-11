# POS Tagging with Viterbi Algorithm

1. Generate the POS tags as well as the Transition Probabilities and Observation 
   Likelihoods from the training data and store them in the model file
   by running the `buildtagger.py` as follows:
   ```bash
   python3 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>
   ```

2. Run the Viterbi algorithm to tag the words in the test file as follows:
   ```bash
   python3 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>
   ```