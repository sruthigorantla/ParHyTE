# ParHyTE
Parallelized implementation of HyTE - Temporally Aware Knowledge Graph Embeddings
## split data
Use split_data.py to split the training data into groups based on time steps 

`python split_data.py <time_step>`

## Train and Test
1. Compile ParHyTE.cpp `g++ -fopenmp ParHyTE.cpp -o ParHyTE`
2. Compile ParHyTE.cpp `g++ -fopenmp ParTest_HyTE.cpp -o ParTest_HyTE`
3. Run the script file with suitable arguments `./train_test_HyTE.sh`
