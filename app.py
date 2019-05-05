from flask import Flask, render_template, request
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
cors  = CORS(app)

#### Processing arguments and predicting back the result
@app.route("/")
def main_page():
    return "YOU HAVE COME TO A HIDDEN PLACE! PLEASE LEAVE!!! :'("

@app.route("/predict")
def index():
    
    #### Getting all the arguments
    # cat = "Art" #request.args.get('category')
    # month = 12 #request.args.get('month')
    # backers = 1 #request.args.get('backers')
    # country = 1 #request.args.get('country')

    #### Getting all the arguments
    cat = request.args.get('category').strip()
    month = int(request.args.get('month').strip())
    backers = int(request.args.get('backers').strip())
    country = int(request.args.get('country').strip())

    #### Converting cat into oneHot Model to feed it into model
    cat  = "_" + cat
    cat_list = ['_Art', '_Comics', '_Crafts', '_Dance', '_Design', '_Fashion', '_Film & Video', '_Food', '_Games', 
       '_Journalism', '_Music', '_Photography', '_Publishing', '_Technology',
       '_Theater']
    cat_index = cat_list.index(cat)
    cat_feed = [0]*len(cat_list)
    cat_feed[cat_index] = 1

    #### Creating the input for model
    input_ = [month, backers, country]
    input_.extend(cat_feed)
    input_ = np.array(input_).reshape(1, -1)

    print(input_)
    #### Loading the model 
    lr = pickle.load(open( "model.pkl", "rb" ))
    return str(lr.predict(input_))



if __name__ == '__main__':
    app.run(debug=True)