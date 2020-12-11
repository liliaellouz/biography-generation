# %tensorflow_version 1.x
import tensorflow as tf
import tensorflow_hub as hub
import gpt_2_simple as gpt2
tf.logging.set_verbosity(tf.logging.ERROR)

import csv
import sys
csv.field_size_limit(100000000)

# define global variables
model_run_name='old_dates'
first_names = ['Achille', 'Adamo', 'Adelmo', 'Adriano', 'Agnolo', 'Agostino', 'Alberico', 'Alberto', 'Alderano', 'Aldo', 'Alessandro', 'Alessio', 'Alfio', 'Alfredo', 'Alphons', 'Amadeo', 'Amedeo', 'Amico', 'Amleto', 'Angelo', 'Annibale', 'Ansaldo', 'Antonello', 'Antonino', 'Antonio', 'Armando', 'Arnaldo', 'Arnulfo', 'Arsenio', 'Arturo', 'Atenolfo', 'Atenulfo', 'Augusto', 'Azeglio', 'Baccio', 'Baldassare', 'Bartolomeo', 'Benedetto', 'Benito', 'Benvenuto', 'Beppe', 'Bernardo', 'Biagio', 'Bruno', 'Calcedonio', 'Calogero', 'Camillo', 'Carlo', 'Carlo', 'Carmelo', 'Carmine', 'Cesare', 'Cipriano', 'Cirillo', 'Ciro', 'Claudio', 'Coluccio', 'Constanzo', 'Coriolano', 'Corrado', 'Costantino', 'Costanzo', 'Danilo', 'Damiano', 'Daniele', 'Daniello', 'Dante', 'Dario', 'Davide', 'Delfino', 'Dino', 'Dionigi', 'Domenico', 'Donatello', 'Donato', 'Durante', 'Edgardo', 'Edoardo', 'Elladio', 'Elmo', 'Emilio', 'Ennio', 'Enrico', 'Enzio', 'Enzo', 'Eraldo', 'Ermanno', 'Ermenegildo', 'Ermes', 'Ernesto', 'Ettore', 'Ezio', 'Fabio', 'Fabrizio', 'Fausto', 'Fedele', 'Federico', 'Federigo', 'Ferdinando', 'Filippo', 'Fiorenzo', 'Fiorino', 'Flavio', 'Francesco', 'Franco', 'Fredo', 'Fulvio', 'Gabriele', 'Gaetano', 'Galasso', 'Gaspare', 'Gastone', 'Gavino', 'Gennaro', 'Geppetto', 'Geronimo', 'Giacinto', 'Giacobbe', 'Giacomo', 'Giampaolo', 'Giampiero', 'Gian', 'Gian Carlo', 'Gianantonio', 'Giancarlo', 'Gianfrancesco', 'Gianfranco', 'Gianluca', 'Gianluigi', 'Gianmarco', 'Giannantonio', 'Gianni', 'Gianpaolo', 'Gianpietro', 'Gilberto', 'Gino', 'Gioacchino', 'Gioachino', 'Gioele', 'Gioffre', 'Gionata', 'Giordano', 'Giorgio', 'Giosuè', 'Giovanni', 'Giovanni Battista', 'Girolamo', 'Giuliano', 'Giulio', 'Giuseppe', 'Giustino', 'Goffredo', 'Graziano', 'Greco', 'Guarino', 'Guglielmo', 'Guido', 'Gustavo', 'Hugo', 'Ignazio', 'Ippazio', 'Ivan', 'Ivo', 'Jacopo', 'Lamberto', 'Lando', 'Laureano', 'Lazzaro', 'Leonardo', 'Leone', 'Leopoldo', 'Liberato', 'Liberto', 'Livio', 'Lodovico', 'Lorenzo', 'Lotario', 'Luca', 'Luchino', 'Luciano', 'Lucio', 'Ludovico', 'Luigi', 'Manuel', 'Marcantonio', 'Marcello', 'Marco', 'Mariano', 'Mario', 'Martino', 'Martino', 'Massimiliano', 'Massimo', 'Matteo', 'Mattia', 'Maurilio', 'Maurizio', 'Mauro', 'Michelangelo', 'Michele', 'Micheletto', 'Michelotto', 'Milo', 'Mirco', 'Mirko', 'Moreno', 'Nanni', 'Napoleone', 'Niccolò', 'Nico', 'Nicola', 'Nicolò', 'Nino', 'Nunzio', 'Omero', 'Orazio', 'Oreste', 'Orlando', 'Osvaldo', 'Ottavio', 'Ottone', 'Pandulf', 'Panfilo', 'Paolo', 'Paride', 'Pasqual', 'Pasquale', 'Patrizio', 'Pellegrino', 'Peppino', 'Pier', 'Pierangelo', 'Piergiorgio', 'Piergiuseppe', 'Pierluigi', 'Piermaria', 'Piero', 'Pierpaolo', 'Piersanti', 'Pietro', 'Pompeo', 'Pomponio', 'Puccio', 'Raffaele', 'Raffaellino', 'Raffaello', 'Raimondo', 'Ranieri', 'Rembrandt', 'Renzo', 'Riccardo', 'Ricciotti', 'Rino', 'Roberto', 'Rocco', 'Rodolfo', 'Rolando', 'Roman', 'Romeo', 'Romolo', 'Ronaldo', 'Rosario', 'Ruggero', 'Ruggiero', 'Sabatino', 'Salvatore', 'Salvi', 'Sandro', 'Sante', 'Santino', 'Saverio', 'Sebastiano', 'Sergius', 'Severino', 'Silvestro', 'Silvio', 'Simone', 'Stefano', 'Tazio', 'Telemaco', 'Temistocle', 'Tiziano', 'Tommaso', 'Toni', 'Tonino', 'Tonio', 'Torquato', 'Tullio', 'Ubaldo', 'Uberto', 'Ugo', 'Ugolino', 'Umberto', 'Valerio', 'Venancio', 'Vincentio', 'Vincenzo', 'Virgilio', 'Vito', 'Vittorio', 'Vladimiro', 'Wladimiro', 'Zanobi']
last_names = ['Rossi', 'Russo', 'Ferrari', 'Esposito', 'Bianchi', 'Romano', 'Regio', 'Ricci', 'Marino', 'Lupo', 'Lastra', 'Bruno', 'Gallo', 'Conti', 'De Luca', 'Mancini', 'Costa', 'Giordano', 'Rizzo', 'Lombardi', 'Moretti']
occupations = ['Acrobat', 'Alchemist', 'Apothecarist', 'Architect', 'Astrologer', 'Armorer', 'Artist', 'Baker', 'Barrister', 'Bookbinder', 'Bowyer', 'Basket Weaver', 'Blacksmith', 'Brewer', 'Brick Layer', 'Butcher', 'Calligrapher', 'Candlemaker', 'Carpenter', 'Cartographer', 'Charcoal Burner', 'Clerk', 'Clothier', 'Cook', 'Coppersmith', 'Cooper', 'Diplomat', 'Dyer', 'Engineer', 'Engraver', 'Falconer', 'Farmer', 'Fisherman', 'Fishmonger', 'Forester', 'Fortune-Teller', 'Fruitier', 'Fuller', 'Furrier', 'Glassblower', 'Goldsmith', 'Grocer', 'Gardener', 'Grain Merchant', 'Grave Digger', 'Haberdasher', 'Herald', 'Herbalist', 'Hunter', 'Inkeeper', 'Interpreter', 'Jester', 'Jeweler', 'Lacemaker', 'Leatherworker', 'Locksmith', 'Mason', 'Mercer', 'Miller', 'Minstrel', 'Messenger', 'Miner', 'Moneylender', 'Navigator', 'Needleworker', 'Painter', 'Pardoner', 'Peddler', 'Priest', 'Physician', 'Playwright', 'Politician', 'Potter', 'Rat Catcher', 'Sailor', 'Scribe', 'Servant', 'Shipwright', 'Shoemaker', 'Silversmith', 'Solicitor', 'Soapmaker', 'Stoncarver', 'Storyteller', 'Spy', 'Tanner', 'Towne Crier', 'Vintner', 'Washer Woman', 'Waterman', 'Weaver', 'Wet Nurse', 'Wheelwright Wood Carver', 'Woodworker']
prompt_middle = ' was a Venetian '
prompt_end = '. [SEP] '

def init(load_model=True):
	"""Initializes tf session and loads model."""
	tf.reset_default_graph()
	tf.initialize_all_variables()
	sess = gpt2.start_tf_sess()

	if load_model:
		gpt2.load_gpt2(sess, run_name='old_dates')

	return sess

def finetune_model(steps=2000, model=None, sess=None):
	"""Finetunes a GPT-2 model (by default its 355M variant)."""

	# guarantee tf session is running and base model has been
	# loaded for fine-tuning
	if not model:
		model = '355M'
		gpt2.download_gpt2(model_name='355M')

	if not sess:
		tf.reset_default_graph()
		sess = init(load_model=False)

	# finetune model
	gpt2.finetune(sess,
				dataset=file_name,
				model_name=model,
				steps=steps,
				restore_from='latest',
				run_name=model_run_name,
				print_every=50,
				sample_every=250,
				save_every=500)

def generate_fake_bio(prompt, sess=None):
	"""Generates a fake biography using the given prompt.
	This prompt does not any hard structure requirements, though
	we designed a semi-strutured prompt that was always employed
	by us to generate biographies was (without the quotes): 
	"{FirstName LastName} was a venetian {occupation}. [SEP ] "
	"""
	# if not sess:
	# 	sess = init()

	split_proposed_bio = ''

	# ensure minimum bio length (GPT-2 can be lead into outputing nothing)
	while len(split_proposed_bio) < 100:
		proposed_bio = str(gpt2.generate(sess, 
										run_name=model_run_name, 
										return_as_list=True, 
										prefix=prompt, 
										length=4096)[0])

		# if model started generating a second bio, discard it
		split_proposed_bio = proposed_bio.split('<|')[0]
		if split_proposed_bio.count('[SEP]') > 1:
			# remove 2nd bio's [SEP] token
			split_proposed_bio = ''.join(
				''.join(split_proposed_bio.split('[SEP]')[:2])
				.split('.')[:-2])
			# remove 2nd bio's prompt before [SEP] token
			split_proposed_bio = '.'.join(
				split_proposed_bio.split('.')[:-1]) + '.'

	# return generated biography
	split_proposed_bio = ' '.join(split_proposed_bio.split())
	return '.'.join(split_proposed_bio.split('.')[:-1]) + '.'


def generate_random_fake_bio(sess=None):
	"""Generates a biography for a random Italian name and a
	random occupation."""
	import random

	prompt = (random.choice(first_names) + ' ' + 
				random.choice(last_names) + prompt_middle + 
				random.choice(occupations).lower() + prompt_end)

	return generate_fake_bio(prompt, sess)