from flask import Flask, render_template, request, json
import random
from enhanced_generative_main import generate_bio

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

		bio, bio_score = generate_bio(name, occupation, float(fidelity))
		return bio


@app.route('/random', methods=['GET'])
def randomize():
	"""Generates random name & occupation combination"""
	if request.method == "GET":
		forenames = ["Giovanni", "Giovanna"]
		surnames = ["Padova", "Belluno"]
		occupations = ["merchant", "artist"]

		first_names = ['Achille', 'Adamo', 'Adelmo', 'Adriano', 'Agnolo', 'Agostino', 'Alberico', 'Alberto', 'Alderano', 'Aldo', 'Alessandro', 'Alessio', 'Alfio', 'Alfredo', 'Alphons', 'Amadeo', 'Amedeo', 'Amico', 'Amleto', 'Angelo', 'Annibale', 'Ansaldo', 'Antonello', 'Antonino', 'Antonio', 'Armando', 'Arnaldo', 'Arnulfo', 'Arsenio', 'Arturo', 'Atenolfo', 'Atenulfo', 'Augusto', 'Azeglio', 'Baccio', 'Baldassare', 'Bartolomeo', 'Benedetto', 'Benito', 'Benvenuto', 'Beppe', 'Bernardo', 'Biagio', 'Bruno', 'Calcedonio', 'Calogero', 'Camillo', 'Carlo', 'Carlo', 'Carmelo', 'Carmine', 'Cesare', 'Cipriano', 'Cirillo', 'Ciro', 'Claudio', 'Coluccio', 'Constanzo', 'Coriolano', 'Corrado', 'Costantino', 'Costanzo', 'Danilo', 'Damiano', 'Daniele', 'Daniello', 'Dante', 'Dario', 'Davide', 'Delfino', 'Dino', 'Dionigi', 'Domenico', 'Donatello', 'Donato', 'Durante', 'Edgardo', 'Edoardo', 'Elladio', 'Elmo', 'Emilio', 'Ennio', 'Enrico', 'Enzio', 'Enzo', 'Eraldo', 'Ermanno', 'Ermenegildo', 'Ermes', 'Ernesto', 'Ettore', 'Ezio', 'Fabio', 'Fabrizio', 'Fausto', 'Fedele', 'Federico', 'Federigo', 'Ferdinando', 'Filippo', 'Fiorenzo', 'Fiorino', 'Flavio', 'Francesco', 'Franco', 'Fredo', 'Fulvio', 'Gabriele', 'Gaetano', 'Galasso', 'Gaspare', 'Gastone', 'Gavino', 'Gennaro', 'Geppetto', 'Geronimo', 'Giacinto', 'Giacobbe', 'Giacomo', 'Giampaolo', 'Giampiero', 'Gian', 'Gian Carlo', 'Gianantonio', 'Giancarlo', 'Gianfrancesco', 'Gianfranco', 'Gianluca', 'Gianluigi', 'Gianmarco', 'Giannantonio', 'Gianni', 'Gianpaolo', 'Gianpietro', 'Gilberto', 'Gino', 'Gioacchino', 'Gioachino', 'Gioele', 'Gioffre', 'Gionata', 'Giordano', 'Giorgio', 'Giosuè', 'Giovanni', 'Giovanni Battista', 'Girolamo', 'Giuliano', 'Giulio', 'Giuseppe', 'Giustino', 'Goffredo', 'Graziano', 'Greco', 'Guarino', 'Guglielmo', 'Guido', 'Gustavo', 'Hugo', 'Ignazio', 'Ippazio', 'Ivan', 'Ivo', 'Jacopo', 'Lamberto', 'Lando', 'Laureano', 'Lazzaro', 'Leonardo', 'Leone', 'Leopoldo', 'Liberato', 'Liberto', 'Livio', 'Lodovico', 'Lorenzo', 'Lotario', 'Luca', 'Luchino', 'Luciano', 'Lucio', 'Ludovico', 'Luigi', 'Manuel', 'Marcantonio', 'Marcello', 'Marco', 'Mariano', 'Mario', 'Martino', 'Martino', 'Massimiliano', 'Massimo', 'Matteo', 'Mattia', 'Maurilio', 'Maurizio', 'Mauro', 'Michelangelo', 'Michele', 'Micheletto', 'Michelotto', 'Milo', 'Mirco', 'Mirko', 'Moreno', 'Nanni', 'Napoleone', 'Niccolò', 'Nico', 'Nicola', 'Nicolò', 'Nino', 'Nunzio', 'Omero', 'Orazio', 'Oreste', 'Orlando', 'Osvaldo', 'Ottavio', 'Ottone', 'Pandulf', 'Panfilo', 'Paolo', 'Paride', 'Pasqual', 'Pasquale', 'Patrizio', 'Pellegrino', 'Peppino', 'Pier', 'Pierangelo', 'Piergiorgio', 'Piergiuseppe', 'Pierluigi', 'Piermaria', 'Piero', 'Pierpaolo', 'Piersanti', 'Pietro', 'Pompeo', 'Pomponio', 'Puccio', 'Raffaele', 'Raffaellino', 'Raffaello', 'Raimondo', 'Ranieri', 'Rembrandt', 'Renzo', 'Riccardo', 'Ricciotti', 'Rino', 'Roberto', 'Rocco', 'Rodolfo', 'Rolando', 'Roman', 'Romeo', 'Romolo', 'Ronaldo', 'Rosario', 'Ruggero', 'Ruggiero', 'Sabatino', 'Salvatore', 'Salvi', 'Sandro', 'Sante', 'Santino', 'Saverio', 'Sebastiano', 'Sergius', 'Severino', 'Silvestro', 'Silvio', 'Simone', 'Stefano', 'Tazio', 'Telemaco', 'Temistocle', 'Tiziano', 'Tommaso', 'Toni', 'Tonino', 'Tonio', 'Torquato', 'Tullio', 'Ubaldo', 'Uberto', 'Ugo', 'Ugolino', 'Umberto', 'Valerio', 'Venancio', 'Vincentio', 'Vincenzo', 'Virgilio', 'Vito', 'Vittorio', 'Vladimiro', 'Wladimiro', 'Zanobi']
		last_names = ['Rossi', 'Russo', 'Ferrari', 'Esposito', 'Bianchi', 'Romano', 'Regio', 'Ricci', 'Marino', 'Lupo', 'Lastra', 'Bruno', 'Gallo', 'Conti', 'De Luca', 'Mancini', 'Costa', 'Giordano', 'Rizzo', 'Lombardi', 'Moretti']
		occupations = ['Acrobat', 'Alchemist', 'Apothecarist', 'Architect', 'Astrologer', 'Armorer', 'Artist', 'Baker', 'Barrister', 'Bookbinder', 'Bowyer', 'Basket Weaver', 'Blacksmith', 'Brewer', 'Brick Layer', 'Butcher', 'Calligrapher', 'Candlemaker', 'Carpenter', 'Cartographer', 'Charcoal Burner', 'Clerk', 'Clothier', 'Cook', 'Coppersmith', 'Cooper', 'Diplomat', 'Dyer', 'Engineer', 'Engraver', 'Falconer', 'Farmer', 'Fisherman', 'Fishmonger', 'Forester', 'Fortune-Teller', 'Fruitier', 'Fuller', 'Furrier', 'Glassblower', 'Goldsmith', 'Grocer', 'Gardener', 'Grain Merchant', 'Grave Digger', 'Haberdasher', 'Herald', 'Herbalist', 'Hunter', 'Inkeeper', 'Interpreter', 'Jester', 'Jeweler', 'Lacemaker', 'Leatherworker', 'Locksmith', 'Mason', 'Mercer', 'Miller', 'Minstrel', 'Messenger', 'Miner', 'Moneylender', 'Navigator', 'Needleworker', 'Painter', 'Pardoner', 'Peddler', 'Priest', 'Physician', 'Playwright', 'Politician', 'Potter', 'Rat Catcher', 'Sailor', 'Scribe', 'Servant', 'Shipwright', 'Shoemaker', 'Silversmith', 'Solicitor', 'Soapmaker', 'Stoncarver', 'Storyteller', 'Spy', 'Tanner', 'Towne Crier', 'Vintner', 'Washer Woman', 'Waterman', 'Weaver', 'Wet Nurse', 'Wheelwright Wood Carver', 'Woodworker']


		result = {'name': random.choice(first_names) +" "+ random.choice(last_names),
				'occupation' : random.choice(occupations)}
		return json.dumps(result)


if __name__ == "__main__":
	app.run()
