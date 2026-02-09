"""
Select utility functions taken from FMA repo utils.py
https://github.com/mdeff/fma/blob/master/utils.py
"""
import os
import ast
import pandas as pd


NB_AUDIO_SAMPLES = 1321967
SAMPLING_RATE = 44100

def get_audio_path(audio_dir, track_id):
    """
    Return the path to the mp3 given the directory where the audio is stored
    and the track ID.

    Examples
    --------
    >>> import utils
    >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
    >>> utils.get_audio_path(AUDIO_DIR, 2)
    '../data/fma_small/000/000002.mp3'

    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

class Loader:
	def load(self, filepath):
		raise NotImplementedError()

class RawAudioLoader(Loader):
	def __init__(self, sampling_rate=SAMPLING_RATE):
		self.sampling_rate = sampling_rate
		self.shape = (NB_AUDIO_SAMPLES * sampling_rate // SAMPLING_RATE, )

	def load(self, filepath):
		return self._load(filepath)[:self.shape[0]]

class LibrosaLoader(RawAudioLoader):
    def _load(self, filepath):
        import librosa
        sr = self.sampling_rate if self.sampling_rate != SAMPLING_RATE else None
        # kaiser_fast is 3x faster than kaiser_best
        # x, sr = librosa.load(filepath, sr=sr, res_type='kaiser_fast')
        x, sr = librosa.load(filepath, sr=sr)
        return x
		
def load(filepath) -> pd.DataFrame:

	filename = os.path.basename(filepath)

	if 'features' in filename:
		return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

	if 'echonest' in filename:
		return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

	if 'genres' in filename:
		return pd.read_csv(filepath, index_col=0)

	if 'tracks' in filename:
		tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

		COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
								('track', 'genres'), ('track', 'genres_all')]
		for column in COLUMNS:
			tracks[column] = tracks[column].map(ast.literal_eval)

		COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
								('album', 'date_created'), ('album', 'date_released'),
								('artist', 'date_created'), ('artist', 'active_year_begin'),
								('artist', 'active_year_end')]
		for column in COLUMNS:
			tracks[column] = pd.to_datetime(tracks[column], errors='coerce')

		SUBSETS = ('small', 'medium', 'large')
		try:
			tracks['set', 'subset'] = tracks['set', 'subset'].astype(
							'category', categories=SUBSETS, ordered=True)
		except (ValueError, TypeError):
			# the categories and ordered arguments were removed in pandas 0.25
			tracks['set', 'subset'] = tracks['set', 'subset'].astype(
								pd.CategoricalDtype(categories=SUBSETS, ordered=True))

		COLUMNS = [('track', 'genre_top'), ('track', 'license'),
								('album', 'type'), ('album', 'information'),
								('artist', 'bio')]
		for column in COLUMNS:
			tracks[column] = tracks[column].astype('category')

		return tracks
    
	return pd.read_csv(filepath)
