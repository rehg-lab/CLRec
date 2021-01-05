path = dict(
		src_dataset_path = '/data/DevLearning/SDFNet_data/ShapeNet55_3DOF-VC_LRBg',
		input_image_path = None,
		input_depth_path = 'depth_NPZ',
		input_normal_path = 'normal_output',
		input_seg_path = 'segmentation',

		src_pt_path = '/data/DevLearning/SDFNet_data/ShapeNet55_sdf',	
		data_split_json_path = '/data/DevLearning/SDFNet_data/json_files/data_split_55.json'
		)
data_setting = dict(
		input_size = 224,
		img_extension = 'png',
		random_view = True,
		seq_len = 25,
		categories = None
		)
training = dict(
		out_dir = '/data/DevLearning/SDFNet_model_output/test_release',
		batch_size = 128,
		batch_size_eval = 16,
		num_epochs = 500,

		save_model_step = 50,
		# Evaluated on val data of all seen classes after each exposure
		eval_step = 500, 
		verbose_step = 10,
		num_points = 2048,
		# Example of a valid cont
		# cont = 'model-0-500.pth.tar',
		cont = None,
		shape_rep = 'sdf',
		model = None,
		coord_system = '3dvc',
		num_rep = 1,
		nclass = 5,
		)
logging = dict(
		log_dir = '/data/DevLearning/SDFNet_model_output/log',
		exp_name = 'test'
		)
testing = dict(
		eval_task_name = 'test',
		box_size = 1.7,
		# Always 1 if generating mesh on the fly
		batch_size_test = 1,
		# Eval up to "split_counter" learning exposure
		split_counter = 10
		) 

