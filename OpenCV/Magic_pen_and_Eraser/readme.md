# Magic Pen and Eraser
This directory implements a magic pen and eraser functionality which can be used by anyone with a good 
quality face camera. The program locates and detects the user's pen from the camera input and traces 
the tip of the pen as the user continues to write on the screen normally. 

## Setup
To get started with, your local machine must have all dependencies installed. It is recommended to create
a virtual environment before proceeding to install all dependencies. All commands given below are tried 
and tested on Windows 10. 

### Setting up a virtual environment
```
$ python -m venv venv   # to create the environment
$ .\venv\Scripts\activate   # activates the virtual environment
```

### Installing dependencies
```
$ pip install -r requirements.txt
```

### Calibrate program with your pen
Currently, I have calibrated the program with a blue ball point pen. Before you start, it is highly 
recommended that you calibrate the scripts with your pen so that detection and tracing is smooth and realistic. 

To calibrate, you will need to run `penval.py` script which will open up a window with various trackbars.
Once this script is running, you will need to adjust the trackbars such that on the image screen, only the tip/cap
of your pen appears **white**. After the trackbars are adjusted and you are satisfied with the calibration,
you can save the values by pressing **'S'** on the keyboard.
```
$ python penval.py  # run the penval script
```

## Usage
Run the `main.py` script to start the program. 
```
$ python main.py
```
A window will appear on which you can draw and erase using the pen which you had used earlier to
calibrate the program. To switch between the eraser and the pen, a button has been provided on the top 
left of the window. You need to bring your hand towards that location in your camera to switch 
between the eraser and the pen.

### Saving the canvas
You might want to save your writing somewhere sometimes. To do that, press **'s'** on the keyboard. You
will then be prompted to enter the name of the saved document. Note that if the document with a similar name
exists then it would be overwritten without any alerts

### Loading canvas
To load a pre-existing canvas which you saved earlier, press **'l'** on the keyboard. You will
then be prompted to enter the name of the document which you wished to save. Note that if no such document
exists then it would not be loaded.

### Changing color of the Pen
If you wish to change the color of the pen during your presentation, press **'c'** on the keyboard. 
A window will popup in which you will be required to enter the color code of the color you desire to write 
with. An example has been provided for the same in the window itself.

### Pause writing
To pause writing and erasing from the document, press **'p'** on the keyboard. The camera will no longer
detect your pen and you will have the freedom to explain whatever you wrote without the pen overwriting
on your written canvas. To start detection once again, press **'p'** once again on the keyboard. The program
will now start detecting the pen normally and you can continue writing.