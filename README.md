# Final-Project-Group-7
Shared repo for Machine Learning II Group 7

This project used the Street View House Number (SVHN) database, available at this link:

http://ufldl.stanford.edu/housenumbers

This project uses **Format 1** of the data at these links:

http://ufldl.stanford.edu/housenumbers/train.tar.gz   (training data)  
http://ufldl.stanford.edu/housenumbers/test.tar.gz    (test data)

There is a convenience bash script in the /data folder called "get_data". After checking out the repo, it may be necessary to set the executable bit on this file before running. Cd to the data directory, then  
  
    chmod +x get_data  
    ./get_data
    
This will download the data and unzip it into the /data directory. Then go up a level and cd into the /code directory, 
and run:  
  
    python3 make_pickles.py
  
to create the pickle files used by the code.
    
