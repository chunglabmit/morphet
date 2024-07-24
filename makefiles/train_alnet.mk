#-----------------------
# train_microglia.mk
#
# created on 03/18/2019
#-----------------------

TIMESTMP := $(shell /bin/date "+%Y%m%d-%H%M%S")
guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Requires [ $* ] to be set"; \
		exit 1; \
	fi

train.alnet.microglia.x: guard-gid guard-bs guard-epoch guard-ce guard-nc guard-desc guard-date
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs $(bs) -e $(epoch) -ts $(TIMESTMP) -ce $(ce) -nc $(nc) -de $(desc) -dp /media/share5/MYK/training_data/$(date) -mt ALTR --debug

train.alnet.microglia.AE: guard-desc
	CUDA_VISIBLE_DEVICES=1 python src/train/main.py -ds microglia_labeled -ph train_val -bs 8 -e 100 -ts $(TIMESTMP) -ce 100 -nc 3 -de "DSET:031219_DESC:$(desc)" -dp /media/share5/MYK/training_data/031219 -mt ALTR -en --train_AE --debug

train.alnet.microglia.with.pretrained: guard-desc guard-aw
	CUDA_VISIBLE_DEVICES=1 python src/train/main.py -ds microglia_labeled -ph train_val -bs 16 -e 100 -ts $(TIMESTMP) -ce 100 -nc 3 -de "DSET:031219_DESC:$(desc)" -dp /media/share5/MYK/training_data/031219 -mt ALTR -aw $(aw) --debug


train.alnet.microglia.2024: guard-gid guard-bs guard-epoch guard-ce guard-nc guard-desc guard-date
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs $(bs) -e $(epoch) -ts $(TIMESTMP) -ce $(ce) -nc $(nc) -de $(desc) -dp /data_ssd3/data/training_data/cc/$(date) -mt ALTR --debug

train.alnet.microglia.human.2024: guard-gid guard-bs guard-epoch guard-ce guard-nc guard-desc
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs $(bs) -e $(epoch) -ts $(TIMESTMP) -ce $(ce) -nc $(nc) -de $(desc) -dp /media/share12/MYK/data/training_data/microglia/human/seowoo/020124_128x128x16 -mt ALTR --debug

train.alnet.microglia.with.pretrained_03_2024: guard-desc guard-aw
	CUDA_VISIBLE_DEVICES=1 python src/train/main.py -ds microglia_labeled -ph train_val -bs 16 -e 100 -ts $(TIMESTMP) -ce 100 -nc 3 -de "DSET:031219_DESC:$(desc)" -dp /media/share5/MYK/training_data/031219 -mt ALTR -aw $(aw) --debug


train.alnet.microglia.031219.no_zproj: guard-gid guard-desc
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 100 -ts $(TIMESTMP) -ce 100 -nc 3 -de $(desc) -dp /media/share5/MYK/training_data/031219 -mt ALTR -n2d --debug

train.alnet.microglia.031219.with_zproj: guard-gid guard-desc
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 100 -ts $(TIMESTMP) -ce 100 -nc 3 -de $(desc) -dp /media/share5/MYK/training_data/031219 -mt ALTR --debug

train.alnet.microglia.031219.with_zproj.pretrained.freeze: guard-gid guard-desc guard-aw guard-nt guard-cl
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 100 -ts $(TIMESTMP) -ce 100 -nc 3 -de $(desc) -dp /media/share5/MYK/training_data/031219 -mt ALTR -aw $(aw) -awf -nt $(nt) -cl $(cl) --debug

train.alnet.microglia.031219.with_zproj.pretrained: guard-gid guard-desc guard-aw guard-nt guard-cl
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 100 -ts $(TIMESTMP) -ce 100 -nc 3 -de $(desc) -dp /media/share5/MYK/training_data/031219 -mt ALTR -aw $(aw) -nt $(nt) -cl $(cl) --debug

train.alnet.microglia.031724: guard-gid guard-desc
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 100 -ts $(TIMESTMP) -ce 100 -nc 3 -de $(desc) -dp /data_ssd3/morphet/train_data/FINAL/031724 -mt ALTR --debug


train.alnet.042624.1: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 100 -ts $(TIMESTMP) -ce 100 -nc 3 -de "ZeroMean_Clip_PRT_ALTR20210914-234831_FREEZE" -dp /media/share5/MYK/training_data/031219 -mt ALTR -aw /media/share12/MYK/models/microglia/20210914-234831-bumblebee/ALTR_AE_00063.pth -nt zero_mean -cl True --debug


train.alnet.042624.2: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 100 -ts $(TIMESTMP) -ce 100 -nc 3 -de "ZeroMean_NoClip_PRT_ALTR20210914-234831_FREEZE" -dp /media/share5/MYK/training_data/031219 -mt ALTR -aw /media/share12/MYK/models/microglia/20210914-234831-bumblebee/ALTR_AE_00063.pth -nt zero_mean -cl False --debug

train.alnet.042624.3: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 100 -ts $(TIMESTMP) -ce 100 -nc 3 -de "MinusOneNOne_NoClip_PRT_ALTR20210914-234831_FREEZE" -dp /media/share5/MYK/training_data/031219 -mt ALTR -aw /media/share12/MYK/models/microglia/20210914-234831-bumblebee/ALTR_AE_00063.pth -nt minus_one_and_one -cl False --debug

train.alnet.042624.4: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 100 -ts $(TIMESTMP) -ce 100 -nc 3 -de "ZeroNOne_NoClip_PRT_ALTR20210914-234831_FREEZE" -dp /media/share5/MYK/training_data/031219 -mt ALTR -aw /media/share12/MYK/models/microglia/20210914-234831-bumblebee/ALTR_AE_00063.pth -nt zero_and_one -cl False --debug

train.alnet.042624.5: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 64 -e 100 -ts $(TIMESTMP) -ce 100 -nc 3 -de "ZeroMean_from_Scratch_NoClip_BS64" -dp /media/share5/MYK/training_data/031219 -mt ALTR -nt zero_mean -cl False -lr 0.01 --debug


train.alnet.reproduce: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 128 -e 100 -ts $(TIMESTMP) -ce 100 -nc 3 -de "ZeroNOne_CLIP_PRT_20210914-234831-bumblebee-00063_FREEZE" -dp /data_ssd3/data/training_data/cc/031219 -mt ALTR -aw /home/mykim/cbm/src/train/weights/models/20210914-234831-bumblebee/ALTR_AE_00063.pth -nt zero_and_one -cl True -awf --debug

train.alnet.finetune2dNet: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 128 -e 100 -ts $(TIMESTMP) -ce 100 -nc 3 -de "ZeroNOne_CLIP_PRT_20210914-234831-bumblebee-00063_UNLOCK" -dp /data_ssd3/data/training_data/cc/031219 -mt ALTR -aw /home/mykim/cbm/src/train/weights/models/20210914-234831-bumblebee/ALTR_AE_00063.pth -nt zero_and_one -cl True --debug

train.alnet.finetune2dNet_smallerLR.AdamOpt: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 128 -e 100 -ts $(TIMESTMP) \
	 -ce 100 -nc 3 -de "ZeroNOne_CLIP_PRT_20210914-234831-bumblebee-00063_UNLOCK_SmallerLR_AdamOptimizer_BiggerReconLoss" \
	 -dp /data_ssd3/data/training_data/cc/031219 -mt ALTR \
	 -aw /home/mykim/cbm/src/train/weights/models/20210914-234831-bumblebee/ALTR_AE_00063.pth -nt zero_and_one -cl True -lr 0.00004 -al 0.7 --debug

train.alnet.reproduce.pretrained.FREEZE: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 64 -e 100 -ts $(TIMESTMP) \
	-ce 100 -nc 3 -de "Reproduce result from latest ALTR, pretrained from the best BMTR, FREEZE resnet" \
	-dp /data_ssd3/data/training_data/cc/031219 -mt ALTR -nt zero_and_one -cl True \
	-lr 0.0001 -al 0.95 -dw 32 -dh 32 -dd 16 \
	-aw /home/mykim/cbm/src/train/weights/models/20240426-181730-bumblebee/ALTR_AE_00099.pth \
	-awf --debug

train.alnet.reproduce.pretrained.UNLOCK: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 200 -ts $(TIMESTMP) \
	-ce 200 -nc 3 -de "pretrained from the best ALTR(202404280-131926), UNLOCK resnet" \
	-dp /media/share5/MYK/training_data/042824 -mt ALTR -nt zero_and_one -cl True \
	-aw /home/mykim/cbm/src/train/weights/models/20240428-131926-bumblebee/ALTR_AE_00099.pth \
	-lr 0.0001 -al 0.5 -dw 32 -dh 32 -dd 16 --debug

train.alnet.reproduce.pretrained.UNLOCK.no_alpha: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 200 -ts $(TIMESTMP) \
	-ce 200 -nc 3 -de "pretrained from the best ALTR(202404280-131926), UNLOCK resnet, NO ALPHA" \
	-dp /media/share5/MYK/training_data/042824 -mt ALTR -nt zero_and_one -cl True \
	-aw /home/mykim/cbm/src/train/weights/models/20240428-131926-bumblebee/ALTR_AE_00099.pth \
	-lr 0.0001 -dw 32 -dh 32 -dd 16 -na --debug

train.alnet.reproduce.scratch.DS042824: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 100 -ts $(TIMESTMP) \
	-ce 100 -nc 3 -de "Reproduce result with the new cleaned dataset. Train from scratch" \
	-dp /media/share5/MYK/training_data/042824 -mt ALTR -nt zero_and_one -cl True \
	-lr 0.001 -al 0.5 -dw 32 -dh 32 -dd 16 --debug



# FULL COMPARISON
train.alnet.scratch.DS042824.alpha_p25: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 300 -ts $(TIMESTMP) \
	-ce 300 -nc 3 -de "Final Benchmarking alpha=0.25" \
	-dp /media/share5/MYK/training_data/042824 -mt ALTR -nt zero_and_one -cl True \
	-lr 0.0001 -al 0.25 -dw 32 -dh 32 -dd 16 --debug

train.alnet.scratch.DS042824.alpha_p5: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 300 -ts $(TIMESTMP) \
	-ce 300 -nc 3 -de "Final Benchmarking alpha=0.5" \
	-dp /media/share5/MYK/training_data/042824 -mt ALTR -nt zero_and_one -cl True \
	-lr 0.0001 -al 0.5 -dw 32 -dh 32 -dd 16 --debug

train.alnet.scratch.DS042824.alpha_p75: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 300 -ts $(TIMESTMP) \
	-ce 300 -nc 3 -de "Final Benchmarking alpha=0.75" \
	-dp /media/share5/MYK/training_data/042824 -mt ALTR -nt zero_and_one -cl True \
	-lr 0.0001 -al 0.75 -dw 32 -dh 32 -dd 16 --debug

train.alnet.scratch.DS042824.noALPHA: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 300 -ts $(TIMESTMP) \
	-ce 300 -nc 3 -de "Final Benchmarking No Alpha" \
	-dp /media/share5/MYK/training_data/042824 -mt ALTR -nt zero_and_one -cl True \
	-lr 0.0001 -dw 32 -dh 32 -dd 16 -na --debug

train.alnet.scratch.DS042824.no2d: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 300 -ts $(TIMESTMP) \
	-ce 300 -nc 3 -de "Final Benchmarking No 2D Data" \
	-dp /media/share5/MYK/training_data/042824 -mt ALTR -nt zero_and_one -cl True \
	-lr 0.0001 -al 0.0 -dw 32 -dh 32 -dd 16 -n2d --debug

train.alnet.reproduce.scratch.DS042824.zero_mean: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 100 -ts $(TIMESTMP) \
	-ce 100 -nc 3 -de "trian.al.net.reproduce.scratch.DS042824.zero_mean, alpha=0.5, No Clip" \
	-dp /media/share5/MYK/training_data/042824 -mt ALTR -nt zero_mean -cl False \
	-lr 0.001 -al 0.5 -dw 32 -dh 32 -dd 16 --debug

train.alnet.reproduce.scratch.DS042824.noALPHA.zero_mean: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 100 -ts $(TIMESTMP) \
	-ce 100 -nc 3 -de "Train from Scratch. Zero Mean, No Alpha, No Clip" \
	-dp /media/share5/MYK/training_data/042824 -mt ALTR -nt zero_mean -cl False \
	-lr 0.001 -dw 32 -dh 32 -dd 16 -na --debug

train.alnet.reproduce.scratch.DS042824.noALPHA.zero_mean.clip: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 32 -e 100 -ts $(TIMESTMP) \
	-ce 100 -nc 3 -de "Train from Scratch. Zero Mean, No Alpha, Clip" \
	-dp /media/share5/MYK/training_data/042824 -mt ALTR -nt zero_mean -cl True \
	-lr 0.001 -dw 32 -dh 32 -dd 16 -na --debug


# New DATASET
train.alnet.042624dataset: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 64 -e 100 -ts $(TIMESTMP) -ce 100 -nc 4 -de "ZeroNOne_NO_CLIP_PRT_20210914-234831-bumblebee-00063_UNLOCK_NewDataset_042624" -dp /media/share7/MYK/MorPheT/data/training_data/042624 -mt ALTR -aw /home/mykim/cbm/src/train/weights/models/20210914-234831-bumblebee/ALTR_AE_00063.pth -nt zero_and_one -cl False -lr 0.001 -al 0.5 -dw 64 -dh 64 -dd 16 --debug

train.alnet.042624dataset.downsample: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 64 -e 100 -ts $(TIMESTMP) -ce 100 -nc 4 -de "ZeroNOne_NO_CLIP_PRT_20210914-234831-bumblebee-00063_UNLOCK_NewDataset_042624_DOWNSAMPLE" -dp /media/share7/MYK/MorPheT/data/training_data/042624 -mt ALTR -aw /home/mykim/cbm/src/train/weights/models/20210914-234831-bumblebee/ALTR_AE_00063.pth -nt zero_and_one -cl False -lr 0.001 -al 0.5 -dw 32 -dh 32 -dd 16 --debug

train.alnet.042624dataset.downsample_zero_mean: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 64 -e 100 -ts $(TIMESTMP) -ce 100 -nc 4 -de "ZeroNOne_NO_CLIP_NO_PRT_SGDOpt_NoAverageLoss" -dp /media/share7/MYK/MorPheT/data/training_data/042624 -mt ALTR -nt zero_mean -cl False -lr 0.0005 -al 0.5 -dw 32 -dh 32 -dd 16 --debug

train.alnet.042624dataset.downsample_zero_mean.SumReduction: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 64 -e 100 -ts $(TIMESTMP) -ce 100 -nc 4 -de "ZeroNOne_NO_CLIP_NO_PRT_SGDOpt_ReductionSum_alpha0.001" -dp /media/share7/MYK/MorPheT/data/training_data/042624 -mt ALTR -nt zero_mean -cl False -lr 0.001 -al 0.001 -dw 32 -dh 32 -dd 16 --debug

train.alnet.042624dataset.downsample_ZAO.MeanReduction: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 64 -e 100 -ts $(TIMESTMP) -ce 100 -nc 4 -de "ZeroNOne_NO_CLIP_NO_PRT_SGDOpt_ReductionMean_alpha0.001" -dp /media/share7/MYK/MorPheT/data/training_data/042624 -mt ALTR -nt zero_and_one -cl False -lr 0.001 -al 0.001 -dw 32 -dh 32 -dd 16 --debug

train.alnet.042624dataset.downsample_ZAO.MeanReduction.pretrained: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 64 -e 100 -ts $(TIMESTMP) -ce 100 -nc 4 -de "pretrain from ALNET" -dp /media/share7/MYK/MorPheT/data/training_data/042624 -mt ALTR -nt zero_and_one -cl True -lr 0.001 -al 0.0001 -dw 32 -dh 32 -dd 16 -aw /home/mykim/cbm/src/train/weights/models/20210914-234831-bumblebee/ALTR_AE_00063.pth --debug

train.alnet.DS042624_downsample.MeanReduction.pretrained.FREEZE: guard-gid
	CUDA_VISIBLE_DEVICES=$(gid) python src/train/main.py -ds microglia_labeled -ph train_val -bs 64 -e 100 -ts $(TIMESTMP) \
	-ce 100 -nc 4 -de "pretrain from BMTR" \
	-dp /media/share7/MYK/MorPheT/data/training_data/042624 -mt ALTR -nt zero_and_one -cl True \
	-lr 0.0001 -al 0.5 -dw 32 -dh 32 -dd 16 \
	-aw /media/share7/MYK/MorPheT/data/train_models/microglia/20190312-170708-bumblebee/BMTR_AE_00200.pth -awf --debug
