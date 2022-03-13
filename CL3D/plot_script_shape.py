import numpy as np
import matplotlib.pyplot as plt
import matplotlib

TITLE_SIZE = 28
AXIS_SIZE = 24
matplotlib.rcParams['font.family'] = ['serif']
matplotlib.rc('axes', titlesize=TITLE_SIZE)
matplotlib.rc('axes', labelsize=AXIS_SIZE)
dotted_line_width = 2.5
fig, ax1 = plt.subplots()
ax1.tick_params(labelsize=20)

###### File names
file_list = ['test/sdfnet_55_single/eval/sdfnet_55_single/out.npz',
			 'test/occnet_55_single/eval/occnet_55_single/out.npz']

###### Curve labels
labels = ['Algo 1', 'Algo 2']
###### Curve colors
colors = ['r', 'orange']

fig = plt.gcf()

###### Change this for different runs
total_classes = 55
n_exposures = 11
n_cls_per_exposure = 5
################################################

x_axis = np.arange(n_exposures)
seen_classes = (np.arange(total_classes)+1)*n_cls_per_exposure
future_classes = total_classes-seen_classes

###### False: For single exposure, True for repeated exposures
rep = False

mean_acc = []
for i,f in enumerate(file_list):
	print(f)

	acc_matrr = np.zeros((n_exposures, total_classes))
	first_exp = {}

	file = np.load(f, allow_pickle=True)
	if 'fscore' in file.files:
		arr_acc = np.asarray(file['fscore'])
		acc = []
		for j,exp in enumerate(arr_acc):
			if type(exp) is np.ndarray or isinstance(exp, list):
				exp_acc = []
				for k,cl in enumerate(exp):
					exp_acc.append(np.mean(cl,axis=0)[1])
					acc_matrr[j,k] = np.mean(cl,axis=0)[1]

					if rep:
						if k not in first_exp:
							first_exp[k] = j
				acc.append(np.mean(exp_acc))
			else:
				acc = arr_acc
				break
	else:
		raise Exception("fscore not in numpy file")

	plt.plot(x_axis,acc,'o-',label=labels[i],markersize=5,color=colors[i],linewidth=3)

####### Batch
gt = ax1.plot([n_exposures-1], 0.50, 'x', color='r', mew=3, label='2.5D Inp. Batch',markersize=15,zorder=10)[0]
gt.set_clip_on(False)

plt.ylim(0,0.55)

plt.title('Single Exposure Shape Reconstruction on ShapeNetCore.v2')

plt.xlabel('Learning Exposures')
plt.ylabel('Fscore@1')
ax1.legend(ncol=2, numpoints=1, borderaxespad=0., fancybox=True, framealpha=0.7, fontsize=20)
plt.grid()
fig.set_size_inches(16, 6)
plt.savefig('shape.pdf',dpi=300, bbox_inches='tight', pad_inches=0.01 ,transparent=True)
plt.show()