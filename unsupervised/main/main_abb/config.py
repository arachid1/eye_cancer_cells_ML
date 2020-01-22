config = {
	'date': '12302019_fine_clustering_resnet50_wodecoder_slide_ALL_lr-3_epoch20_ncluster100_clusterdim64_batch128_imgsize128_gpu2_largestSTD',
	'log_dir':'../log',
	'input_dir':'../data/generated_data/coarse_extraction_128',
	'out_dir':'../data/result/fine_extraction/',
	'model_name': 'resnet50',
	'fixed_feature': False,

	'pretrain_path': '../checkpoint/11282019_fine_clustering_vae_wodecoder_slide_ALL_lr-3_50_30_ncluster50_batch256_gpu4_finetune_1128.pkl',

	'num_epoch':20,
	'learning_rate': 1e-3,
	'batch_size':32,
	'if_train':True,
	'if_valid':True,
	'valid_epoch_interval':1,
	'switch_learning_rate_interval': 3,
}
