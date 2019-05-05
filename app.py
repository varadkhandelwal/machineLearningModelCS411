from flask import Flask, render_template
from load import loadData

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)