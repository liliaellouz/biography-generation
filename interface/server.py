from flask import Flask, render_template, request, json
import random

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')



@app.route('/', methods=['POST'])
def get_bio():
	"""Given name, occupation and a threshhold of realness, generates a biography"""
	if request.method == "POST":
		name = request.form['name']
		occupation = request.form["occupation"]
		fidelity = request.form["fidelity"]

		return f"{name} was a Venetian {occupation}. The fidelity level of this bio is {fidelity}."


@app.route('/random', methods=['GET'])
def randomize():
	"""Generates random name & occupation combination"""
	if request.method == "GET":
		forenames = ["Giovanni", "Giovanna"]
		surnames = ["Padova", "Belluno"]
		occupations = ["merchant", "artist"]
		result = {'name': random.choice(forenames) +" "+ random.choice(surnames),
				'occupation' : random.choice(occupations)}
		return json.dumps(result)


if __name__ == "__main__":
	app.run()
