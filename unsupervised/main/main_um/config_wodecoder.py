config = {
	'date': '12172019_fine_clustering_resnet50_wodecoder_slide_ALL_lr-3_epoch40_ncluster50_clusterdim64_batch128_imgsize128_gpu2',
	'log_dir':'../log',
	'input_dir':'../data/generated_data/coarse_extraction_128',
	'out_dir':'../data/result/fine_extraction/',
	'model_name': 'resnet50',
	'DCN_out_channel': 512,
	'cluster_vector_dim': 64,
	'fixed_feature': False,
	'n_cluster': 50,

	'resume': False,
	'pretrain_path': '../checkpoint/11282019_fine_clustering_vae_wodecoder_slide_ALL_lr-3_50_30_ncluster50_batch256_gpu4_finetune_1128.pkl',

	'num_epoch':20,
	'learning_rate': 1e-3,
	'batch_size':128,
	'cluster_update_multi': 0.1,
	'if_train':True,
	'if_valid':True,
	'valid_epoch_interval':1,
	'switch_learning_rate_interval': 3,

	'ae_loss_multi_ori': 0.0,
	'cluster_loss_multi_ori': 1.0,
	'ae_loss_multi_new': 0.0,
	'cluster_loss_multi_new':1.0,
	'loss_multi_epoch_switch':-1
}
