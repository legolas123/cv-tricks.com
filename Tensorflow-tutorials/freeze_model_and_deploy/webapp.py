import sys
import os
import flask
from flask import render_template, send_from_directory, request, redirect,url_for
from werkzeug import secure_filename
from flask import jsonify
import base64
import StringIO
import tensorflow as tf 
import numpy as np
import cv2
# Obtain the flask app object
app = flask.Flask(__name__)

UPLOAD_FOLDER='static'
def load_graph(trained_model):   
    with tf.gfile.GFile(trained_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=""
            )
    return graph

@app.route('/')
def index():
    return "Webserver is running"

@app.route('/demo',methods=['POST','GET'])
def demo():
    if request.method == 'POST':
        upload_file = request.files['file']
        filename = secure_filename(upload_file.filename)
        upload_file.save(os.path.join(UPLOAD_FOLDER, filename))
        image_size=128
        num_channels=3
        images = []
        # Reading the image using OpenCV
        image = cv2.imread(os.path.join(UPLOAD_FOLDER, filename))
        # Resizing the image to our desired size and preprocessing will be done exactly as done during training
        image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
        images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0/255.0)
        #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
        x_batch = images.reshape(1, image_size,image_size,num_channels)
        graph =app.graph
        y_pred = graph.get_tensor_by_name("y_pred:0")
          ## Let's feed the images to the input placeholders
        x= graph.get_tensor_by_name("x:0")
        #y_true = graph.get_tensor_by_name("y_true:0") 
        y_test_images = np.zeros((1, 2))    
        sess= tf.Session(graph=graph)

        ### Creating the feed_dict that is required to be fed to calculate y_pred 
        feed_dict_testing = {x: x_batch}
        result=sess.run(y_pred, feed_dict=feed_dict_testing)
        # result is of this format [probabiliy_of_cats probability_of_dogs]
        #print()
        #pred=str(result[0][0]).split(" ")
        #print(pred)
        out={"cat":str(result[0][0]),"dog":str(result[0][1])}
        return jsonify(out)
        #return redirect(url_for('just_upload',pic=filename))

    return  '''
    <!doctype html>
    <html lang="en">
    <head>
      <title>Running my first AI Demo</title>
    </head>
    <body>
    <div class="site-wrapper">
        <div class="cover-container">
            <nav id="main">
                <a href="http://localhost:5000/demo" >HOME</a>
            </nav>
          <div class="inner cover">

          </div>
          <div class="mastfoot">
          <hr />
            <div class="container">
              <div style="margin-top:5%">
		            <h1 style="color:black">Dogs Cats Classification Demo</h1>
		            <h4 style="color:black">Upload new Image </h4>
		            <form method=post enctype=multipart/form-data>
	                 <p><input type=file name=file>
        	        <input type=submit style="color:black;" value=Upload>
		            </form>
	            </div>	
            </div>
        	</div>
     </div>
   </div>
</body>
</html>

    '''




app.graph=load_graph('./dogs-cats-model.pb')  
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("5000"), debug=True, use_reloader=False)
    
