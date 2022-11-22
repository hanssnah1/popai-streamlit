import io

import collections
import tensorflow as tf
import glob
import pathlib
import numpy as np
import pandas as pd
import pretty_midi
import requests
import seaborn as sns #statistical graphics
import streamlit as st
from bs4 import BeautifulSoup
from scipy.io import wavfile

@st.cache(allow_output_mutation=True)
def load_session():
    return requests.Session()


def has_download_attr(tag):
    return tag.has_attr("download")


@st.cache(
    hash_funcs={requests.Session: id},
    allow_output_mutation=True,
    suppress_st_warning=True,
)
def download_from_bitmidi(url: str, sess: requests.Session) -> bytes:
    user_agent = {"User-agent": "bot"}
    r_page = sess.get(url, headers=user_agent)
    soup = BeautifulSoup(r_page.content, "html.parser")
    link = soup.find(lambda tag: tag.name == "a" and tag.has_attr("download"))
    if link is None:
        st.error(f"No MIDI file found on page '{url}'")
        raise ValueError(f"No MIDI file found on page '{url}'")

    url_midi_file = "https://bitmidi.com" + link["href"]
    r_midi_file = sess.get(url_midi_file, headers=user_agent)
    return r_midi_file.content

################## Tensorflow Code

model = tf.keras.models.load_model("Finalmodel.h5")
model.summary()

def duration(tempDur):
  K = tempDur
  lst = [0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5]
  value = lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]
  return value

def midi_to_notes(midi_file: str) -> pd.DataFrame:
  pm = pretty_midi.PrettyMIDI(midi_file)
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  # Sort the notes by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start - prev_start)
    notes['duration'].append(end - start)
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str,
  instrument_name: str,
  velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = end

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

def create_sequences(
    dataset: tf.data.Dataset,
    seq_length: int,
    vocab_size = 22,
) -> tf.data.Dataset:
  """Returns TF Dataset of sequence and label examples."""
  seq_length = seq_length+1

  # Take 1 extra for the labels
  windows = dataset.window(seq_length, shift=1, stride=1,
                              drop_remainder=True)

  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)

  # Normalize note pitch
  def scale_pitch(x):
    x = x/[vocab_size,1.0,1.0]
    return x

  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

    return scale_pitch(inputs), labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

def mse(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  return tf.reduce_mean(mse)

def predict_next_note(
    notes: np.ndarray,
    keras_model: tf.keras.Model,
    temperature: float = 1.0) -> int:
  """Generates a note IDs using a trained sequence model."""

  assert temperature > 0

  # Add batch dimension
  inputs = tf.expand_dims(notes, 0)

  predictions = model.predict(inputs)
  pitch_logits = predictions['pitch']
  step = predictions['step']
  duration = predictions['duration']

  pitch_logits /= temperature
  pitch = tf.random.categorical(pitch_logits, num_samples=1)
  pitch = tf.squeeze(pitch, axis=-1)
  duration = tf.squeeze(duration, axis=-1)
  step = tf.squeeze(step, axis=-1)

  # `step` and `duration` values should be non-negative
  step = tf.maximum(0, step)
  duration = tf.maximum(0, duration)

  return int(pitch), float(step), float(duration)

def run(file):
  #for randomization
  seed = 42
  tf.random.set_seed(seed)
  np.random.seed(seed)

  # Sampling rate for audio playback
  _SAMPLING_RATE = 16000

  data_dir = pathlib.Path('FinalDataset')
  filenames = glob.glob(str(data_dir/'*.mid*'))
  print('Number of files:', len(filenames))


  all_notes = []
  for f in filenames:
    notes = midi_to_notes(f)
    all_notes.append(notes)

  sample_file = filenames[-1]

  pm = pretty_midi.PrettyMIDI(sample_file)

  instrument = pm.instruments[0]
  instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

  all_notes = pd.concat(all_notes)
  all_notes["pitch"].max()

  raw_notes = midi_to_notes(sample_file)
  raw_notes.head()

  key_order = ['pitch', 'step', 'duration']
  train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

  notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
  notes_ds.element_spec

  seq_length = 25
  vocab_size = 22
  seq_ds = create_sequences(notes_ds, seq_length, vocab_size)
  seq_ds.element_spec

  temperature = 4
  num_predictions = 15
  vocab_size = 22

  sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

  # The initial sequence of notes; pitch is normalized similar to training
  # sequences
  input_notes = (
      sample_notes[-seq_length:] / np.array([vocab_size, 1, 1]))

  generated_notes = []
  prev_start = 0
  prev_end = 0

  for _ in range(num_predictions):
    pitch, step, duration = predict_next_note(input_notes, model, temperature)
    start = prev_start + step
    end = start + duration
    input_note = (pitch, step, duration)
    generated_notes.append((*input_note, start, end))
    input_notes = np.delete(input_notes, 0, axis=0)
    input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
    prev_start = end

  generated_notes = pd.DataFrame(
      generated_notes, columns=(*key_order, 'start', 'end'))

  generated_notes.head(20)

  out_file = 'output.mid'
  out_pm = notes_to_midi(
      generated_notes, out_file=out_file, instrument_name=instrument_name)
  return(out_pm)

################## Tensorflow Code

def main():

    st.title("MIDI Output GUI")
    st.header("Input a midi file of 25 notes.")
    sess = load_session()

    uploaded_file = st.file_uploader("Upload MIDI file", type=["mid"])

    midi_file = None

    if uploaded_file is not None:
        midi_file = uploaded_file
        run(midi_file)
    else:
        st.error("Input MIDI file")
        st.stop()

    st.markdown("---")

    with st.spinner(f"Transcribing to FluidSynth"):
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        audio_data = midi_data.fluidsynth()
        audio_data = np.int16(
            audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
        )  # -- Normalize for 16 bit audio https://github.com/jkanner/streamlit-audio/blob/main/helper.py

        virtualfile = io.BytesIO()
        wavfile.write(virtualfile, 44100, audio_data)

    st.audio(virtualfile)
    st.markdown("Download the audio by right-clicking on the media player")


if __name__ == "__main__":
    st.set_page_config(
        page_title="MIDI to WAV",
        page_icon="musical_note",
        initial_sidebar_state="collapsed",
    )
    main()
    with st.sidebar:
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://twitter.com/andfanilo">@andfanilo</a></h6>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="margin: 0.75em 0;"><a href="https://www.buymeacoffee.com/andfanilo" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a></div>',
            unsafe_allow_html=True,
        )
