
def rmOuttliers(dataset, target):
	dataset[target]=dataset[target].apply(pd.to_numeric, errors='raise')
	target_avg = dataset[target].mean()
	target_std = dataset[target].std()
	dataset = dataset.where((dataset[target] < target_avg - 2 * target_std) &
	                      (dataset[target] > target_avg + 2 * target_std)
	                       )
	return dataset.dropna(axis=0, how='all', thresh=None, subset=None, inplace=True)


def rmAllOutliers(dataset):
	for name in dataset.columns.names:
		dataset=rmOuttliers(dataset, name)
	return dataset



