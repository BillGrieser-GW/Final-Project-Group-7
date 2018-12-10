These files generate data in the pickle format required by some project pieces. Start in the /data directory and run "get_data". You may need to make it executable. Then cd to this folder. Run:

python3 make_pickles.py

That will create the pickle files. To test this, run:

python3 read_pickles.py

To display some data from the pickles. 

The other files are included by other modules and should not be run from here.

Code for the three frameworks are in the pytorch, caffe, and tensorflow folders. CD to each one to run the code in there.
