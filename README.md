# CS-598-Ryan-Kupiec
CS598 DL for healthcare final project - Ryan Kupiec
To run the model you must follow these steps:
- Download the NOTEVENTS.csv and ADMISSIONS.csv from MIMIC-III
- Use the Data_Creation.ipynb notebook to merge the datasets, remove patients, and save to a merged mortatlity.csv
- Use the CS598_Final_Ryan_Kupiec.ipynb notebook. Load in the required packages and run preprocessing.py on the merged datset to chunk and split the dataset
- Pass the preprocessed dataset through split_into_chunk.py to split the data in chunks
- Pass the chunked data to FTL.py (FTL-Trans model) or run_clbert_lstm.py (LSTM model) to run the model. Testing chunks can be found at the bottom of the script to test the models. 
