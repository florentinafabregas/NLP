# About

This repository contains the source code for our final exam in the course Natural Language Processing and Deep Learning, at IT University of Copenhagen. We analyse the resilience against typographical errors between models for Named Entity Recognition task in NLP. The comparison is between BERT and BiLSTM, with both word and character-level embeddings. By injecting different types of noise in the test dataset, at different levels, we want to compare the drop in performance among NLP models to define how robust they are against typos. 

#### Reproducibilty of results
##### Training
The training of the three different models can be found in the three main jupyter notebooks bilstm_word.ipynb bilstm_char.ipynb and bert.ipynb respectively. For reproducibility you can use the train-dev-test dataset in the folder data from the repository. You also need to download the glove embeddings glove.gB.100d.txt to the same folder. 

###### Testing
For testing performance in noisy test datasets you need an input file with the format <word><TAB><nertag>. You can use 'data/gold.txt'. Choose a type of noise (capitalization_swap, character_swap, character_removal, character_replacement) and a rate. 
Example input file = 'data/gold.txt' , type = 'character_swap', rate = 0.175

##### Noise injection
 Run the noisee.py script with input file, type and rate. It will output a modified text file in folder out/noisy_file

 python3 scripts/noise.py data/gold.txt character_swap 0.175

##### Predict and evaluate
Run any of the model scripts for prediction of previously generated noisy data. It will output a prediction file in the corresponding folder predictions/model, and it will return the span-F1 score. 

python3 scripts/predict_bilstm_word.py character_swap 0.175

------ return <Span-F1 score:  0.4649981181783967>
