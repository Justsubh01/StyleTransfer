try:

    import imghdr
    import os
    import uuid

    from flask import Flask, render_template, request, redirect, url_for, abort, \
        send_from_directory , flash, jsonify, make_response
    from werkzeug.utils import secure_filename

    from functions_module import *

    import requests
   
  
    print("All Modules Loaded in main block ..... ")
except:
    print(" Some Module are missing  in main block..... ")

#################################################Flask####################


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/content'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg', '.gif','.webp']

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    content_file = request.files['content-file']
    style_file = request.files['style-file']
    files = [content_file, style_file]
    content_name = str(uuid.uuid4()) + ".jpg"
    style_name = str(uuid.uuid4()) + ".jpg"
    file_names = [content_name, style_name]
    for i, file in enumerate(files):
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file.filename != '':
            file.filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_names[i]))
    result_filename = parameter_gen(file_names[0], file_names[1])
    
    params = {
        'content': "static/uploads/content/" + file_names[0],
        'style' : "static/uploads/content/" + file_names[1],
        'result' : "static/uploads/content/" + result_filename
    }
    return render_template("result.html", **params)



##########################################################################

if __name__=='__main__':
    
    app.run(debug=True,port=5000)