import numpy as np
import pandas as pd
from scipy import stats, sparse
from statsmodels.stats.multitest import multipletests

np.random.seed(42)


# Function to compute hypergeometric p-values for two arrays
def hypergeom_p_values(data, selected):

	"""
	Calculates p_values using hypergeometric distribution for two numpy arrays.
	Works on a matrices containing zeros and ones. All other values are truncated to zeros and ones.
	Args:
		data (numpy.array): all documents in rows, their term features in columns.
		selected (numpy.array): selected documents in rows, their term features in columns.
		callback: callback function used for printing progress.
	Returns: p-values for features
	"""

	def col_sum(x):
		if sparse.issparse(x):
			return np.squeeze(np.asarray(x.sum(axis=0)))
		else:
			return np.sum(x, axis=0)

	if data.shape[1] != selected.shape[1]:
		raise ValueError("Number of columns does not match.")

	# Clip values to a binary variables
	data = data > 0
	selected = selected > 0

	num_features = selected.shape[1]
	pop_size = data.shape[0] # Population size = number of all data examples
	sam_size = selected.shape[0] # Sample size = number of selected examples
	pop_counts = col_sum(data) # Number of observations in population = occurrences of words all data
	sam_counts = col_sum(selected) # Number of observations in sample = occurrences of words in selected data
	step = 250
	p_vals = []

	for i, (pc, sc) in enumerate(zip(pop_counts, sam_counts)):
		hyper = stats.hypergeom(pop_size, pc, sam_size)
		# Since p-value is probability of equal to or "more extreme" than what was actually observed
		# we calculate it as 1 - cdf(sc-1). sf is survival function defined as 1-cdf.
		p_vals.append(hyper.sf(sc-1))
	return p_vals


def compute_bias_fdr(df, domains, top_ids, filename):

	from collections import OrderedDict
	from statsmodels.stats.multitest import multipletests

	p_vals = hypergeom_p_values(df, df.loc[top_ids])
	neg_logs = -1 * np.log10(p_vals)
	fdrs = multipletests(p_vals, method="fdr_bh")[1]

	results = pd.DataFrame({"TERM": df.columns, "DOMAIN": domains, 
							"P": p_vals, "-log10(P)": neg_logs, "FDR": fdrs})
	
	results["ORDER"] = 1
	for i, dom in enumerate(OrderedDict.fromkeys(results["DOMAIN"])):
	    results.loc[results["DOMAIN"] == dom, "ORDER"] = i
	results = results.sort_values(["ORDER", "FDR"])

	results.to_csv("data/bias/{}.csv".format(filename))

	return results


def tfidf(df):

	# Documents along index, terms along columns
	docs = df.index
	terms = df.columns

	# Inverse document frequencies
	doccount = float(df.shape[0])
	freqs = df.astype(bool).sum(axis=0)
	idfs = np.log(doccount / freqs)
	idfs[np.isinf(idfs)] = 0.0  # log(0) = 0
	idfs = idfs.values.reshape(len(idfs), 1)
	
	# Term frequencies
	doc_totals = df.sum(axis=1)
	tfs = (df.values.T / doc_totals.values.T)
	
	# Reweighting of the matrix
	mat_tfidf = tfs * idfs
	df_tfidf = pd.DataFrame(mat_tfidf.T, index=docs, columns=terms)

	return df_tfidf


def plot_dots(df, filename, xrange=list(range(0, 100, 20)), width=3.5, height=14, font_size=8):
	
	import matplotlib.pyplot as plt
	from matplotlib import font_manager, rcParams
	from style import style
	
	font = style.font
	
	prop_x = font_manager.FontProperties(fname=font, size=14)
	prop_xlab = font_manager.FontProperties(fname=font, size=16)
	prop_y = font_manager.FontProperties(fname=font, size=font_size)

	rcParams["axes.linewidth"] = 1.5
	
	dom2col = {dom: style.c[col] for dom, col in zip(style.order["data-driven"], style.fw2c["data-driven"])}
	colors = [dom2col[dom] for dom in df["DOMAIN"]]
	terms = [term.replace("_", " ") for term in df["TERM"]]
	
	fig, ax = plt.subplots(figsize=(width, height))
	
	plt.scatter(df["-log10(P)"], range(len(df)), color=colors, s=40)
	
	for i, p in enumerate(df["-log10(P)"]):
		plt.plot([0, p], [i, i], color=colors[i], linewidth=2.5)

	plt.xticks(xrange)
	plt.yticks(np.arange(0, len(df), step=1), terms, ha="right")
	
	ax.set_xticklabels(xrange, fontproperties=prop_x)
	ax.set_yticklabels(terms, fontproperties=prop_y)
	
	plt.xlim([xrange[0], xrange[-1]])
	plt.ylim([-0.25, len(terms)-0.25])
	
	plt.gca().invert_yaxis()
	
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
		
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)
	
	ax.set_xlabel("-log$_{10}$(${p}$)", fontproperties=prop_xlab)
	
	plt.savefig("figures/{}.png".format(filename), dpi=250, bbox_inches="tight")
	plt.show()


def load_atlas(path="data", cerebellum="combo"):

	import numpy as np
	from nilearn import image

	cer = "{}/brain/atlases/Cerebellum-MNIfnirt-maxprob-thr25-1mm.nii.gz".format(path)
	cor = "{}/brain/atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz".format(path)
	sub = "{}/brain/atlases/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz".format(path)

	sub_del_dic = {1:0, 2:0, 3:0, 12:0, 13:0, 14:0}
	sub_lab_dic_L = {4:1, 5:2, 6:3, 7:4, 9:5, 10:6, 11:7, 8:8}
	sub_lab_dic_R = {15:1, 16:2, 17:3, 18:4, 19:5, 20:6, 21:7, 7:8}

	sub_mat_L = image.load_img(sub).get_data()[91:,:,:]
	sub_mat_R = image.load_img(sub).get_data()[:91,:,:]

	for old, new in sub_del_dic.items():
		sub_mat_L[sub_mat_L == old] = new
	for old, new in sub_lab_dic_L.items():
		sub_mat_L[sub_mat_L == old] = new
	sub_mat_L = sub_mat_L + 48
	sub_mat_L[sub_mat_L == 48] = 0

	for old, new in sub_del_dic.items():
		sub_mat_R[sub_mat_R == old] = new
	for old, new in sub_lab_dic_R.items():
		sub_mat_R[sub_mat_R == old] = new
	sub_mat_R = sub_mat_R + 48
	sub_mat_R[sub_mat_R == 48] = 0

	cor_mat_L = image.load_img(cor).get_data()[91:,:,:]
	cor_mat_R = image.load_img(cor).get_data()[:91,:,:]

	mat_L = np.add(sub_mat_L, cor_mat_L)
	mat_L[mat_L > 56] = 0
	mat_R = np.add(sub_mat_R, cor_mat_R)
	mat_R[mat_R > 56] = 0

	if cerebellum == "combo":
		mat_R = mat_R + 59
		mat_R[mat_R > 118] = 0
		mat_R[mat_R < 60] = 0

	elif cerebellum == "seg":
		mat_R = mat_R + 74
		mat_R[mat_R > 148] = 0
		mat_R[mat_R < 75] = 0

	cer_mat_L = image.load_img(cer).get_data()[91:,:,:]
	cer_mat_R = image.load_img(cer).get_data()[:91,:,:]
	
	if cerebellum == "combo":
		cer_mat_L[np.isin(cer_mat_L,[1,3,5,14,17,20,23,26])] = 57
		cer_mat_L[np.isin(cer_mat_L,[8,11])] = 58
		cer_mat_L[np.isin(cer_mat_L,[6,9,12,15,18,21,24,27])] = 59
		cer_mat_R[np.isin(cer_mat_R,[2,4,7,16,19,22,25,28])] = 116
		cer_mat_R[np.isin(cer_mat_R,[10,13])] = 117
		cer_mat_R[np.isin(cer_mat_R,[6,9,12,15,18,21,24,27])] = 118
		
		mat_L = np.add(mat_L, cer_mat_L)
		mat_L[mat_L > 59] = 0
		mat_R = np.add(mat_R, cer_mat_R)
		mat_R[mat_R > 118] = 0

	elif cerebellum == "seg":
		cer_mat_L[cer_mat_L == 1] = 57
		cer_mat_L[cer_mat_L == 3] = 58
		cer_mat_L[cer_mat_L == 5] = 59
		cer_mat_L[cer_mat_L == 6] = 69
		cer_mat_L[cer_mat_L == 8] = 65
		cer_mat_L[cer_mat_L == 9] = 67
		cer_mat_L[cer_mat_L == 11] = 66
		cer_mat_L[cer_mat_L == 12] = 68
		cer_mat_L[cer_mat_L == 14] = 60
		cer_mat_L[cer_mat_L == 15] = 70
		cer_mat_L[cer_mat_L == 17] = 61
		cer_mat_L[cer_mat_L == 18] = 71
		cer_mat_L[cer_mat_L == 20] = 62
		cer_mat_L[cer_mat_L == 21] = 72
		cer_mat_L[cer_mat_L == 23] = 63
		cer_mat_L[cer_mat_L == 24] = 73
		cer_mat_L[cer_mat_L == 26] = 64
		cer_mat_L[cer_mat_L == 27] = 74
		
		cer_mat_R[cer_mat_R == 2] = 131
		cer_mat_R[cer_mat_R == 4] = 132
		cer_mat_R[cer_mat_R == 6] = 143
		cer_mat_R[cer_mat_R == 7] = 133
		cer_mat_R[cer_mat_R == 9] = 141
		cer_mat_R[cer_mat_R == 10] = 139
		cer_mat_R[cer_mat_R == 12] = 142
		cer_mat_R[cer_mat_R == 13] = 140
		cer_mat_R[cer_mat_R == 15] = 144
		cer_mat_R[cer_mat_R == 16] = 134
		cer_mat_R[cer_mat_R == 18] = 145
		cer_mat_R[cer_mat_R == 19] = 135
		cer_mat_R[cer_mat_R == 21] = 146
		cer_mat_R[cer_mat_R == 22] = 136
		cer_mat_R[cer_mat_R == 24] = 147
		cer_mat_R[cer_mat_R == 25] = 137
		cer_mat_R[cer_mat_R == 27] = 148
		cer_mat_R[cer_mat_R == 28] = 138

		mat_L = np.add(mat_L, cer_mat_L)
		mat_L[mat_L > 75] = 0
		mat_R = np.add(mat_R, cer_mat_R)
		mat_R[mat_R > 148] = 0

	mat = np.concatenate((mat_R, mat_L), axis=0)
	atlas_image = image.new_img_like(sub, mat)

	return atlas_image


def transparent_background(file_name):
	
	from PIL import Image
	
	img = Image.open(file_name)
	img = img.convert("RGBA")
	data = img.getdata()
	
	newData = []
	for item in data:
		if item[0] == 255 and item[1] == 255 and item[2] == 255:
			newData.append((255, 255, 255, 0))
		else:
			newData.append(item)
	
	img.putdata(newData)
	img.save(file_name, "PNG")


def map_plane(estimates, atlas, path, suffix="", plane="z", cut_coords=1, cbar=False,
			  vmin=0.0, vmaxs=[], cmaps=[], print_fig=True, verbose=False):
	
	from nilearn import image, plotting

	if len(vmaxs) < len(estimates.columns):
		vmaxs = [round(v, 2) for v in estimates.max()]
	
	for f, feature in enumerate(estimates.columns):
		
		stat_map = image.copy_img(atlas).get_data()
		data = estimates[feature]
		
		if verbose:
			print("{:20s} Min: {:6.4f}  Mean: {:6.4f}  Max: {:6.4f}".format(
				  feature, min(data), np.mean(data), max(data)))
		if not verbose and print_fig:
			print("\n{}".format(feature))
		
		for i, value in enumerate(data):
			stat_map[stat_map == i+1] = value
		stat_map = image.new_img_like(atlas, stat_map)
		
		if plane == "ortho":
			cut_coords = None
		
		display = plotting.plot_stat_map(stat_map,
										 display_mode=plane, cut_coords=cut_coords,
										 symmetric_cbar=False, colorbar=cbar,
										 cmap=cmaps[f], threshold=vmin, 
										 vmax=vmaxs[f], alpha=0.5,
										 annotate=False, draw_cross=False)
		
		file_name = "{}/{}{}.png".format(path, feature, suffix)
		display.savefig(file_name, dpi=250)
		transparent_background(file_name)
		
		if print_fig:
			display.close()
			display = plotting.plot_stat_map(stat_map,
										 display_mode=plane, cut_coords=cut_coords,
										 symmetric_cbar=False, colorbar=cbar,
										 cmap=cmaps[f], threshold=vmin, 
										 vmax=vmaxs[f], alpha=0.5,
										 annotate=True, draw_cross=False)
			plotting.show()
		display.close()
