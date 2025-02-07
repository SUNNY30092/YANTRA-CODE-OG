All the above files run using few python libraries and other dependencies The below is the command line instructions using python installation package(pip) for downloading and running the above files:
#checking for python
python --version
//if python not installed please follow any tutorial to install it.

#Creating vertual environment(optional but recommended)
cd /path/to/your/project
python -m venv venv

#installing dependencies
pip install fastapi uvicorn pandas python-multipart
pip install numpy
pip install scikit-learn

#setting the server properly
python -m http.server 8080

#running main.py folder
python main.py

#run your html file
you can open document using any browser or use live server if you are testing using VScode.


Thats it you can now smoothly run the application.
