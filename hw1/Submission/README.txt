Dominic Ventimiglia 
3035572695

How to Duplicate Results:


1. In featurize.py change the BASE_DIR to match whatever the correct file path is for your computer. Then run featurize.py that way the correct feature vectors will be created for the spam dataset.

2.  Do the same thing as #1 for load.py

3. In the submitted ipython notebook, you will also need to change the file paths as well. Near the top of nearly every function there will be an np.load(*). Make sure you change the file paths for these loads that way you can properly load the data file. 

4. In the submission notebook, make sure to run the top two cells. Nearly every other cell is dependent on them. 

5. Each cell is labeled the question it corresponds to. At this point, in order to reproduce results, all you have to do is run the cell for the corresponding question you want to test. (Beware that for #5 it takes over an hour to run, you can delete some of the c values out of the c value array, especially the larger ones in order for it to run faster)