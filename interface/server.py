from flask import Flask, render_template, request, json
# from flask import request, jsonify


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')


@app.route('/', methods=['POST'])
def get_bio():
	if request.method == "POST":

		_json = json.dumps(request.form)

		return str(json.dumps(request.form))
	else: return "not a POST request", 500
	# occupation = request.form['occupation']

	# return "Json is : " + str(request.get_json()['name']), 201
	# return render_template('index.html')

if __name__ == "__main__":
	app.run()
