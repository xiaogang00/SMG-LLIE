from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ours_encode': {
		'transforms': transforms_config.OursEncodeTransforms,
		'train_source_root': dataset_paths['ours_train_lq'],
		'train_target_root': dataset_paths['ours_train_hq'],
		'test_source_root': dataset_paths['ours_test_lq'],
		'test_target_root': dataset_paths['ours_test_hq'],
	},
}
