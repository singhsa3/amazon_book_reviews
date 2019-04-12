Objective:
Based on a given set of sentences , paragraph or document, identify other similar documents from a set of documents using doc2vec

The data used here is historical Amazon Book Reviews. 

Files:
You will need the following files to run the program
1. Data_import.ipynb : This is used to import raw data , do some processing and then store in a pickle binary format
2. Pre_processing and Train.ipynb : This is used to do data clean up, tokenize the words, converting them into vectors and finally training on our dataset.
3. Conclusion.ipynb is where we load the trained model and test it out against some unseen data. Here we also compare against tf-idf methodology
4. prep2.py : This contains various function. I would have prefered to write these functions in the jupyter notebook. However, it was giving error during multiprocessing

Other considerations:
It took almost 3 hours to run the codes. I am using Windows 10, 64 bit, HP Z820 , 32 logical cores (Intel Xeon 2 Ghz) and 128 GB RAM. This is not what you normally would have, so consider using AWS. The steps remain the same

Since it’s a long running process, turn the logging on. At least you will be aware what is going on. Love Python for this !

Make sure you have a C compiler before installing Gensim, to use the optimized Doc2vec routines
