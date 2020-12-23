path = dict(
		# src_dataset_path = '/media/ant/5072f36a-8c4f-445d-89f5-577b5b1e05d9/ShapeNet_13_RGB_D+N',
		src_dataset_path = '/data/DevLearning/SDFNet_data/ShapeNet55_3DOF-VC_LRBg',
		# src_dataset_path = '/data/DevLearning/ModelNet40_3DOF_renders',

		# src_dataset_path = '/scr/devlearning/ShapeNet55_EVC_LR',
		# src_dataset_path = '/data/devlearning/ShapeNet55_EVC_LR_PRED_UNSEEN/',
		# src_dataset_path = '/data/devlearning/ShapeNet55_HVC_LR_PRED/',
		# src_dataset_path = '/data/devlearning/ShapeNet55_HVC_LR_PRED_UNSEEN/',
		# src_dataset_path = '/data/devlearning/ShapeNet55_HVC_LR_PRED_SEEN_TEST',

		
		# src_dataset_path = '/data/devlearning/ShapeNet55_EVC_LR_PRED_SEEN_TEST/',
		# src_dataset_path = '/data/devlearning/ShapeNet55_HVC_basic/',
		# src_dataset_path = '/data/devlearning/ShapeNet55_HVC_basic_PRED_SEEN/',
		# src_dataset_path = '/data/devlearning/LRBG_study_PRED/HVC_basic_100_samples_LRB',



		########### Variability
		# src_dataset_path = '/data/devlearning/data_for_LRBg_variability_study/ShapeNet55_HVC_light_VAL_100_samples',
		# src_dataset_path = '/data/devlearning/data_for_LRBg_variability_study/ShapeNet55_HVC_reflectance+light_VAL_100_samples',
		# src_dataset_path = '/data/devlearning/data_for_LRBg_variability_study/ShapeNet55_HVC_reflectance+light+background_VAL_100_samples',




		# src_dataset_path = '/data/devlearning/ShapeNet55_EVC_LR_PRED_2ND_backup',


		# input_image_path = 'image_output',
		input_image_path = None,

		input_depth_path = 'depth_NPZ',
		# input_depth_path = 'depth_pred_NPZ',

		# input_depth_path = None,


		input_normal_path = 'normal_output',
		# input_normal_path = 'normals_pred_output',

		# input_normal_path = None,

		input_seg_path = 'segmentation',
		# input_seg_path = 'segmentation_output',

		# input_seg_path = None,


		# src_pt_path = '/media/ant/50b6d91a-50d7-45a7-b77c-92b79a62e56a/data/ShapeNet.build',
		src_pt_path = '/data/DevLearning/SDFNet_data/ShapeNet55_sdf',	
		# src_pt_path = '/data/DevLearning/SDFNet_data/ModelNet_sdf',	


		input_points_path = '',
		input_pointcloud_path = '',
		input_metadata_path = '',
		# data_split_json_path = '/data/DevLearning/SDFNet_data/json_files/modelnet_split.json',
		# data_split_json_path = '/data/devlearning/gensdf_data/json_files/data_13_42_split_unseen.json',
		# data_split_json_path = '/data/DevLearning/SDFNet_data/json_files/data_split.json',
		# data_split_json_path = '/data/DevLearning/SDFNet_data/json_files/sample.json',
		# data_split_json_path = '/data/DevLearning/SDFNet_data/json_files/data_42_50_each.json',

		# data_split_json_path = '/data/devlearning/sample_test_data_incr/sample.json',
		# data_split_json_path = '/data/devlearning/sample_test_data_incr/data_42_50_each.json',	

		# data_split_json_path = '/data/DevLearning/SDFNet_data/json_files/data_split_55.json',
		data_split_json_path = '/data/DevLearning/SDFNet_data/json_files/sample_30obj_55.json',


		######### For LRBg
		# data_split_json_path = '/data/devlearning/gensdf_data/json_files/val_LRBg_study_updated.json',
		src_mesh_path = '/data/devlearning/gensdf_data/ShapeNet_sdf_55'
			)
data_setting = dict(
		input_size = 224,
		img_extension = 'png',
		random_view = True,
		seq_len = 25,
		# categories = ['03001627', '02691156', '02958343', '03636649', '04090263']
		# categories = ['03636649']

		# categories = ['02691156','02828884','02933112','02958343','03001627','03211117','03636649','03691459','04090263','04256520','04379243','04401088','04530566']
		categories = None
		)
training = dict(
		# out_dir = '/data/devlearning/model_output_bmvc/sdf_gt_dn_hvc',
		# out_dir = '/data/devlearning/model_output_bmvc/img_hvc',
		# out_dir = '/data/devlearning/model_output_bmvc/sdf_pred_dn_evc',
		# out_dir = '/data/devlearning/model_output_bmvc/occnet_evc',
		# out_dir = '/data/devlearning/model_output_bmvc/sdf_gt_dn_oc',
		# out_dir = '/data/devlearning/model_output_bmvc/sdf_pred_dn_hvc',

		# out_dir = '/data/devlearning/model_output_bmvc/sdf_img_hvc_basic',
		# out_dir = '/data/devlearning/model_output_bmvc/sdf_pred_dn_hvc_basic',

		# out_dir = '/data/DevLearning/SDFNet_model_output/incr_13_hvc_rep_5',
		# out_dir = '/data/devlearning/model_output_incr/incr_13_oc',
		# out_dir = '/data/DevLearning/SDFNet_model_output/incr_13_hvc_rep_10_cls',
		# out_dir = '/data/DevLearning/SDFNet_model_output/incr_13_hvc_rep_10_2',
		# out_dir = '/data/DevLearning/SDFNet_model_output/incr_5_hvc_rep_4',
		# out_dir = '/data/DevLearning/SDFNet_model_output/incr_5_hvc_rep_4_img',
		# out_dir = '/data/DevLearning/SDFNet_model_output/gdumb_5_rep_4',
		# out_dir = '/data/DevLearning/SDFNet_model_output/gdumb_5_rep_4_2',


		# out_dir = '/data/DevLearning/SDFNet_model_output/incr_5_hvc_rep_4_clb',

		# out_dir = '/data/DevLearning/SDFNet_model_output/incr_5_hvc_rep_4_occnet',

		# out_dir = '/data/DevLearning/SDFNet_model_output/gdumb_13_single',
		# out_dir = '/data/DevLearning/SDFNet_model_output/gdumbplus_13_single',

		# out_dir = '/data/DevLearning/SDFNet_model_output/occnet_13_single',
		# out_dir = '/data/DevLearning/SDFNet_model_output/sdfnet_13_single_img',

		# out_dir = '/data/DevLearning/SDFNet_model_output/sdfnet_13_rep_img',

		# out_dir = '/data/DevLearning/SDFNet_model_output/sdfnet_13_single',
		# out_dir = '/data/DevLearning/SDFNet_model_output/modelnet_incr_single',
		# out_dir = '/data/DevLearning/SDFNet_model_output/sdfnet_13_2_single',
		# out_dir = '/data/DevLearning/SDFNet_model_output/modelnet_single_40_4',
		# out_dir = '/data/DevLearning/SDFNet_model_output/occnet_13_2_single',
		# out_dir = '/data/DevLearning/SDFNet_model_output/test',
		# out_dir = '/data/DevLearning/SDFNet_model_output/occnet_modelnet_single_40_4',
		# out_dir = '/data/DevLearning/SDFNet_model_output/sdfnet_55_single',
		out_dir = '/data/DevLearning/SDFNet_model_output/occnet_55_single',





























		# batch_size = 128,
		batch_size = 320,

		# batch_size = 50,

		# batch_size = 256,

		# batch_size_eval = 16,
		batch_size_eval = 32,

		# num_epochs = None,
		num_epochs = 500,

		save_model_step = 50,
		eval_step = 5000, #### After each exposure
		verbose_step = 10,
		num_points = 2048,
		# cont = None,
		# cont = 'model-5-500.pth.tar',
		# cont = 'model-7-500.pth.tar',
		cont = 'model-9-500.pth.tar',



		# cont = 'model-29-500.pth.tar',

		# cont = 'model-3-500.pth.tar',
		# cont = 'model-4-500.pth.tar',



		# cont = 'model-900.pth.tar', # For GTDN HVC
		# cont = 'model-38-500.pth.tar', # For GTDN HVC
		#################################################
		# cont = 'model-64-500.pth.tar', # For GTDN HVC
		# cont = 'model-80-500.pth.tar', # For GTDN HVC
		# cont = 'model-97-500.pth.tar', # For GTDN HVC
		# cont = 'model-114-500.pth.tar', # For GTDN HVC





		algo = 'occnet',

		# algo = 'disn',
		model = None,
		# coord_system = 'vc',
		coord_system = 'hvc',
		# coord_system = 'oc',
		rep = True,
		num_rep = 1,
		# num_rep = 10,
		# joint = True,
		joint = False,
		clb = False, ### continual batch

		# nclass = 1,
		# nclass = 2,
		# nclass = 4,
		nclass = 5,



		# gdumb = True
		# gdumb = False




		)
logging = dict(
		log_dir = '/data/DevLearning/SDFNet_model_output/log',
		# exp_name = 'sdf_gt_dn_hvc' # Change
		# exp_name = 'img_hvc' # Change
		# exp_name = 'sdf_pred_dn_evc' # Change
		# exp_name = 'sdf_img_hvc_basic' # Change


		# exp_name = 'occnet_evc' # Change
		# exp_name = 'sdf_gt_dn_oc' # Change

		# exp_name = 'sdf_pred_dn_hvc' # Change

		# exp_name = 'sdf_pred_dn_hvc_basic' # Change

		# exp_name = 'incr_13_oc' # Change
		# exp_name = 'incr_13_update'
		# exp_name = 'incr_13_hvc_rep_5'
		# exp_name = 'incr_13_hvc_rep_10_cls'
		# exp_name = 'incr_13_hvc_rep_10_2'
		# exp_name = 'incr_5_hvc_rep_4'
		# exp_name = 'incr_5_hvc_rep_4_img'
		# exp_name = 'gdumb_5_rep_4'
		# exp_name = 'gdumb_5_rep_4_2'



		# exp_name = 'incr_5_hvc_rep_4_clb'
		# exp_name = 'incr_5_hvc_rep_4_occnet'

		# exp_name = 'gdumb_13_single'
		# exp_name = 'gdumbplus_13_single'
		# exp_name = 'occnet_13_single'
		# exp_name = 'sdfnet_13_single_img'
		# exp_name = 'modelnet_incr_single'

		# exp_name = 'sdfnet_13_rep_img'
		# exp_name = 'sdfnet_13_2_single'
		# exp_name = 'modelnet_single_40_4'
		# exp_name = 'occnet_13_2_single'
		# exp_name = 'occnet_modelnet_single_40_4'
		# exp_name = 'sdfnet_55_single'
		exp_name = 'occnet_55_single'



















		)
testing = dict(
		# eval_task_name = 'sdf_gt_dn_hvc_unseen_3deg', #Change
		# eval_task_name = 'sdf_gt_dn_hvc_unseen_2deg', #Change
		# eval_task_name = 'sdf_gt_dn_hvc_seen_2deg', #Change
		# eval_task_name = 'sdf_gt_dn_hvc_seen_3deg', #Change
		# eval_task_name = 'sdf_gt_dn_hvc_basic_val', #Change

		# eval_task_name = 'sdf_gt_dn_hvc_seen_3deg_2', #Change

		

		# eval_task_name = 'sdf_gt_dn_oc_unseen_3deg', #Change




		# eval_task_name = 'occnet_evc', #Change
		# eval_task_name = 'occnet_evc_unseen_2deg', #Change
		# eval_task_name = 'occnet_evc_seen_2deg', #Change




		# eval_task_name = 'img_hvc', #Change
		# eval_task_name = 'img_hvc_unseen_3deg', #Change
		# eval_task_name = 'img_hvc_seen_3deg', #Change


		# eval_task_name = 'sdf_pred_dn_evc', #Change
		# eval_task_name = 'sdf_pred_dn_evc_unseen_2deg', #Change
		# eval_task_name = 'sdf_pred_dn_evc_seen_2deg', #Change


		# eval_task_name = 'sdf_pred_dn_hvc', #Change
		# eval_task_name = 'sdf_pred_dn_hvc_unseen_3deg', #Change
		# eval_task_name = 'sdf_pred_dn_hvc_seen_3deg', #Change



		# eval_task_name = 'img_hvc_basic', #Change
		# eval_task_name = 'img_hvc_basic_val', #Change

		# eval_task_name = 'img_hvc_basic_light_val', #Change
		# eval_task_name = 'img_hvc_basic_lr_val', #Change
		# eval_task_name = 'img_hvc_basic_lrbg_val', #Change

		# eval_task_name = 'sdf_pred_dn_hvc_basic', #Change
		# eval_task_name = 'sdf_pred_dn_hvc_basic_val', #Change
		# eval_task_name = 'sdf_pred_dn_hvc_basic_light_val', #Change
		# eval_task_name = 'sdf_pred_dn_hvc_basic_LRB_val', #Change






		# eval_task_name = 'incr_13_oc', #Change
		# eval_task_name = 'incr_13_oc_gen', #Change
		# eval_task_name = 'incr_13_hvc_gen',
		# eval_task_name = 'incr_13_hvc_rep_5',
		# eval_task_name = 'incr_13_hvc_rep_10_cls',
		# eval_task_name = 'incr_13_hvc_rep_10_2',
		# eval_task_name = 'incr_5_hvc_rep_4',
		# eval_task_name = 'incr_5_hvc_rep_4_img',
		# eval_task_name = 'gdumb_5_rep_4',
		# eval_task_name = 'gdumb_5_rep_4_2',



		# eval_task_name = 'incr_5_hvc_rep_4_clb',

		# eval_task_name = 'gdumb_13_single',
		# eval_task_name = 'gdumbplus_13_single',

		# eval_task_name = 'occnet_13_single',
		# eval_task_name = 'sdfnet_13_single_img',
		# eval_task_name = 'sdfnet_13_rep_img',
		# eval_task_name = 'modelnet_incr_single',

		# eval_task_name = 'sdfnet_13_2_single',
		# eval_task_name = 'modelnet_single_40_4',
		# eval_task_name = 'occnet_13_2_single',
		# eval_task_name = 'occnet_modelnet_single_40_4',
		# eval_task_name = 'sdfnet_55_single',
		eval_task_name = 'occnet_55_single',















		box_size = 1.7,
		# box_size = 1.01,

		batch_size_test = 1,
		model_selection_path = None,
		# split_counter = 80
		# split_counter = 97
		# split_counter = 114
		# split_counter = 129

		# split_counter = 64
		# split_counter = 39

		# split_counter = 6
		# split_counter = 9

		# split_counter = 12

		# split_counter = 5
		# split_counter = 7
		# split_counter = 9
		split_counter = 10








		) 

