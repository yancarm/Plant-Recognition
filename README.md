1. Run Anaconda Shell
2. " conda create -n tensorflow(you can write any name you want here) pip ptyhon=3.5 "
3. conda activate tensorflow
4. pip install tensorflow==1.15.0
		keras==2.1.2
		numpy
		pandas
		matplotlib
		opencv-python
		scikitplot
		seaborn
   (If there are missing libraries, it will give an error. The error output indicates which library it is.)
5. Switch to the folder of the code we want to test. for example; " (tensorflow) PS C:\Users\yancar\Desktop\plant> "
6. python extract_features.py
7. python train.py
6. python test.py

NOTE: Don't forget to change the conf.json file according to the folder where your images are located.



























#from sklearn.metrics import multilabel_confusion_matrix