from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField
from wtforms.validators import NumberRange

import tensorflow as tf
import numpy as np  
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy

def generate_text(model, start_seed, gen_size=100 ,temp=1.0):
  '''
  model: Trained Model to Generate Text
  start_seed: Intial Seed text in string form
  gen_size: Number of characters to generate

  Basic idea behind this function is to take in some seed text, format it so
  that it is in the correct shape for our network, then loop the sequence as
  we keep adding our own predicted characters.
  '''
    
  # Number of characters to generate
  num_generate = gen_size

  # Vectorizing starting seed text
  input_eval = [char_to_ind[s] for s in start_seed]

  # Expand to match batch format shape
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty list to hold resulting generated text
  text_generated = []

  # Temperature effects randomness in our resulting text
  # The term is derived from entropy/thermodynamics.
  # The temperature is used to effect probability of next characters.
  # Higher probability == lesss surprising/ more expected
  # Lower temperature == more surprising / less expected
 
  temperature = temp

  # Here batch size == 1
  model.reset_states()

  for i in range(num_generate):

      # Generate Predictions
      predictions = model(input_eval)

      # Remove the batch shape dimension
      predictions = tf.squeeze(predictions, 0)

      # Use a cateogircal disitribution to select the next character
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # Pass the predicted charracter for the next input
      input_eval = tf.expand_dims([predicted_id], 0)

      # Transform back to character letter
      text_generated.append(ind_to_char[predicted_id])

  return (start_seed + ''.join(text_generated))


app = Flask(__name__)
# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'mysecretkey'

path_to_file = 'shakespeare.txt'
text = open(path_to_file, 'r').read()
vocab = sorted(set(text))
char_to_ind = {u:i for i, u in enumerate(vocab)}
ind_to_char = np.array(vocab)

def sparse_cat_loss(y_true,y_pred):
  return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
  model = Sequential()
  model.add(Embedding(vocab_size, embed_dim,batch_input_shape=[batch_size, None]))
  model.add(GRU(rnn_neurons,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))
  # Final Dense Layer to Predict
  model.add(Dense(vocab_size))
  model.compile(optimizer='adam', loss=sparse_cat_loss) 
  return model

# Length of the vocabulary in chars
vocab_size = len(vocab)
# The embedding dimension
embed_dim = 64
# Number of RNN units
rnn_neurons = 1026

# LOAD THE MODEL
nlp_model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1) # Model is trained with different batch size

nlp_model.load_weights('shakespeare_gen.h5')

nlp_model.build(tf.TensorShape([1, None]))


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class TextForm(FlaskForm):
    start_text = StringField('Start Text')

    submit = SubmitField('Generate')


@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = TextForm()
    # If the form is valid on submission
    if form.validate_on_submit():
        # Grab the data

        session['start_text'] = form.start_text.data

        return redirect(url_for("generation"))

    return render_template('home.html', form=form)


@app.route('/generation')
def generation():

    Start_Text = session['start_text']

    results = generate_text(model=nlp_model,start_seed=Start_Text)

    return render_template('generation.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)