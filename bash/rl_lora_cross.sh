
CUDA_VISIBLE_DEVICES=1 python mainRL.py group_name=StableMoFusion_crossData experiment_name=10Step_Guidance_LORA_128x4_t2m_to_kit lora=True guidance_weight=2.5 train_batch_size=22 real_dataset_name="kit"
