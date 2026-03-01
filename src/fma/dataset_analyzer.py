from pathlib import Path
import shutil
import logging
from typing import List
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import numpy as np


class DatasetAnalyzer:
	def __init__(self, 
		name: str,
		idstr: str,
		sampling_rate: int, 
		track_genres: List[int],
		genre_encoder: LabelEncoder,
		audio_min_sec: int,
		audio_max_sec: int,
		audio_start_pos: List[int],
		audio_end_pos: List[int]
	):
		self.ds_name_ = name
		self.sampling_rate_ = sampling_rate
		self.track_genres = track_genres
		self.genre_encoder = genre_encoder
		self.audio_min_sec = audio_min_sec
		self.audio_max_sec = audio_max_sec
		self.audio_start_pos = audio_start_pos
		self.audio_end_pos = audio_end_pos
		self.idstr = idstr

		self.dir_ = Path('analysis') / 'dataset' / idstr
		if self.dir_.exists():
			shutil.rmtree(self.dir_)
		self.dir_.mkdir(parents=True, exist_ok=True)
		self.lengths_sec = [(end-st)/self.sampling_rate_ for st, end in zip(self.audio_start_pos, self.audio_end_pos)]
		self.start_sec = [st/self.sampling_rate_ for st in self.audio_start_pos]
		self.mean_length_sec = sum(self.lengths_sec)/len(self.lengths_sec) if self.lengths_sec else 0.0
		self.genre_names = self.genre_encoder.classes_
		self.per_genre_stats = self._compute_per_genre()

	def _compute_per_genre(self):
		stats = defaultdict(dict)
		for g_idx, g_name in enumerate(self.genre_names):
			indices = [i for i, x in enumerate(self.track_genres) if x==g_idx]
			if not indices:
				continue
			stats[g_name]['count'] = len(indices)
			stats[g_name]['lengths_sec'] = [self.lengths_sec[i] for i in indices]
			stats[g_name]['start_sec'] = [self.start_sec[i] for i in indices]
			stats[g_name]['mean_length'] = sum(stats[g_name]['lengths_sec'])/len(stats[g_name]['lengths_sec'])
			stats[g_name]['mean_start'] = sum(stats[g_name]['start_sec'])/len(stats[g_name]['start_sec'])
			stats[g_name]['std_length'] = np.std(stats[g_name]['lengths_sec'])
			stats[g_name]['std_start'] = np.std(stats[g_name]['start_sec'])
		return stats

	def simple(self):
		print(
			f'Dataset: {self.ds_name_}'
			f'\nSampling rate: {self.sampling_rate_}'
			f'\nTotal audio files: {len(self.audio_start_pos)}'
			f'\nMean audio length: {self.mean_length_sec:.2f}s'
		)
		print("Counts per genre:")
		for g, stat in self.per_genre_stats.items():
			print(f"{g}: count={stat['count']}, mean_length={stat['mean_length']:.2f}s, mean_start={stat['mean_start']:.2f}s, std_length={stat['std_length']:.2f}, std_start={stat['std_start']:.2f}")

	def visual(self):
		if not self.audio_start_pos:
			logging.warning("[ANALYSIS] No data to visualize.")
			return
		plt.style.use("ggplot")
		fig, ax = plt.subplots(figsize=(12,6))
		labels = list(self.genre_names)
		counts = [self.per_genre_stats[g]['count'] if g in self.per_genre_stats else 0 for g in labels]
		bars = ax.bar(labels, counts, alpha=0.85, edgecolor="black", linewidth=0.8)
		ax.set_title(f"{self.ds_name_} — Counts by Genre", fontsize=14, weight="bold")
		ax.set_xlabel("Genre")
		ax.set_ylabel("Sample Count")
		ax.grid(axis="y", linestyle="--", alpha=0.6)
		plt.xticks(rotation=45)
		for bar in bars:
			h = bar.get_height()
			ax.annotate(f"{int(h)}", xy=(bar.get_x()+bar.get_width()/2,h), xytext=(0,4), textcoords="offset points", ha="center", fontsize=9)
		plt.tight_layout()
		plt.savefig(self.dir_/ f"{self.ds_name_}_counts_by_genre.png", dpi=200)
		plt.close(fig)
		global_bins = range(self.audio_min_sec, self.audio_max_sec + 1)
		start_max = max(int(np.ceil(max(self.start_sec))) if self.start_sec else self.audio_max_sec, 1)
		start_bins = range(0, start_max + 1)
		self._plot_hist(self.lengths_sec, global_bins, "Audio Length (seconds)", "Frequency", f"{self.ds_name_}_length_histogram.png", f"{self.ds_name_} — Audio Length Histogram")
		self._plot_hist(self.start_sec, start_bins, "Start Position (seconds)", "Frequency", f"{self.ds_name_}_start_histogram.png", f"{self.ds_name_} — Start Position Histogram")
		for g, stat in self.per_genre_stats.items():
			self._plot_hist(stat['lengths_sec'], global_bins, "Audio Length (seconds)", "Frequency", f"{g}_length_histogram.png", f"{self.ds_name_} — {g} Length Histogram")
			self._plot_hist(stat['start_sec'], start_bins, "Start Position (seconds)", "Frequency", f"{g}_start_histogram.png", f"{self.ds_name_} — {g} Start Histogram")
		self._plot_bar_mean_std('length')
		self._plot_bar_mean_std('start')
		logging.info(f"[ANALYSIS] Saved visualizations to {self.dir_}")

	def _plot_hist(self, values, bins, xlabel, ylabel, filename, title):
		fig, ax = plt.subplots(figsize=(10,6))
		ax.hist(values, bins=bins, alpha=0.85, edgecolor="black", linewidth=0.7)
		ax.set_title(title, fontsize=14, weight="bold")
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.grid(axis="y", linestyle="--", alpha=0.6)
		plt.tight_layout()
		plt.savefig(self.dir_/filename, dpi=200)
		plt.close(fig)

	def _plot_bar_mean_std(self, mode):
		labels = []
		means = []
		stds = []
		for g, stat in self.per_genre_stats.items():
			labels.append(g)
			if mode=='length':
				means.append(stat['mean_length'])
				stds.append(stat['std_length'])
			else:
				means.append(stat['mean_start'])
				stds.append(stat['std_start'])
		fig, ax = plt.subplots(figsize=(12,6))
		bars = ax.bar(labels, means, yerr=stds, capsize=5, alpha=0.85, edgecolor="black", linewidth=0.8)
		ax.set_title(f"{self.ds_name_} — Avg {mode.capitalize()} by Genre ± Stddev", fontsize=14, weight="bold")
		ax.set_xlabel("Genre")
		ax.set_ylabel(f"{mode.capitalize()} (seconds)")
		ax.grid(axis="y", linestyle="--", alpha=0.6)
		plt.xticks(rotation=45)
		plt.tight_layout()
		plt.savefig(self.dir_/ f"{self.ds_name_}_avg_{mode}_by_genre.png", dpi=200)
		plt.close(fig)

def compare_splits(*analyzers: DatasetAnalyzer):
	if not analyzers:
		return

	split_labels = []
	for a in analyzers:
		s = [part for part in a.idstr.split("_") if part.startswith("split-")]
		split_labels.append(s[0].replace("split-", "") if s else a.ds_name_)

	mean_lengths = [a.mean_length_sec for a in analyzers]
	mean_starts = [sum(a.start_sec)/len(a.start_sec) if a.start_sec else 0.0 for a in analyzers]

	fig, ax = plt.subplots(figsize=(8, 6))
	ax.bar(split_labels, mean_lengths, color="skyblue", edgecolor="black")
	ax.set_title("CompareSplits_MeanAudioLength")
	ax.set_ylabel("Mean Length (s)")
	plt.tight_layout()
	plt.savefig(analyzers[0].dir_ / "CompareSplits_MeanAudioLength.png", dpi=200)
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(8, 6))
	ax.bar(split_labels, mean_starts, color="lightgreen", edgecolor="black")
	ax.set_title("CompareSplits_MeanStartPosition")
	ax.set_ylabel("Mean Start (s)")
	plt.tight_layout()
	plt.savefig(analyzers[0].dir_ / "CompareSplits_MeanStartPosition.png", dpi=200)
	plt.close(fig)

	genre_names = analyzers[0].genre_encoder.classes_
	genre_percentages = []
	for a in analyzers:
		counts = [sum(1 for x in a.track_genres if x == i) for i in range(len(genre_names))]
		total = sum(counts)
		if total > 0:
			percent = [c/total*100 for c in counts]
		else:
			percent = [0 for _ in counts]
		genre_percentages.append(percent)

	fig, ax = plt.subplots(figsize=(12, 6))
	width = 0.2
	x = np.arange(len(genre_names))
	for i, percent in enumerate(genre_percentages):
		ax.bar(x + i*width, percent, width=width, edgecolor="black", label=split_labels[i])
	ax.set_xticks(x + width)
	ax.set_xticklabels(genre_names, rotation=45)
	ax.set_ylabel("Genre Percentage (%)")
	ax.set_title("CompareSplits_GenrePercent")
	ax.legend()
	plt.tight_layout()
	plt.savefig(analyzers[0].dir_ / "CompareSplits_GenrePercent.png", dpi=200)
	plt.close(fig)
