#-----------------------
# data.mk
#
# created on 03/29/2019
#-----------------------
TIMESTMP := $(shell /bin/date "+%Y%m%d-%H%M%S")

guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Requires [ $* ] to be set"; \
		exit 1; \
	fi

convert.tif2zarr.x: guard-data_root guard-channel
	python src/utils/data/tif2zarr.py -dr $(data_root) -ch $(channel)

convert.tif2zarr.big: guard-data_root guard-channel guard-chunksize guard-ext
	python src/utils/data/tif2zarr.py -dr $(data_root) -ch $(channel) -cs $(chunksize) -ext $(ext) --batchwise

generate_data.x: guard-data_root guard-drp guard-save_path guard-dw guard-dh guard-dd guard-ac guard-csz_half guard-csz_quart guard-ns guard-num_cpu
	python src/utils/data/generate_data.py -dr $(data_root) -dp $(drp) -dw $(dw) -dh $(dh) -dd $(dd) -ac $(ac) -ts $(TIMESTMP) -ns $(ns) -ch $(csz_half) -cq $(csz_quart) -sp $(save_path) -nc $(num_cpu)

generate_data.Microglia.human.02012024:
	python src/utils/data/generate_data.py -dr /media/share7/MYK/SeoWoo/human_IBA1 -ts 020124 -sp /media/share12/MYK/data/training_data/microglia/human/seowoo/ -dt 128x128x16 -nc 10 -ns 40 -ch 64 -cq 8 -mc '["IBA1"]'

generate_data.Microglia.dev.02022024:
	python src/utils/data/generate_data.py -dr /mnt/cephfs/MYK/2023/April-May/20230505_18_30_54_167_P0_hemi-1_aGFP_none_CD206 -ts 020124 -sp /data_ssd3/data/training_data/cc/020224 -dt 32x32x16 -nc 25 -ns 200 -ch 32 -cq 8 -mc '["Ex_488_Em_0_destriped_stitched_downsampled2x"]'

generate_data.Microglia.dev.02022024.2:
	python src/utils/data/generate_data.py -dr /mnt/cephfs/MYK/2023/20230217_13_20_42_172_E15.5_aGFP_none_CD206 -ts 020224 -sp /data_ssd3/data/training_data/cc/020224_172 -dt 32x32x16 -nc 25 -ns 200 -ch 32 -cq 8 -mc '["Ex_488_Em_0_destriped_stitched_master_downsampled2x"]'

generate_data.Microglia.dev.02022024.3:
	python src/utils/data/generate_data.py -dr /mnt/cephfs/MYK/2023/20230219_18_36_48_197_E13.5_aGFP_none_CD206 -ts 020224 -sp /data_ssd3/data/training_data/cc/020224_197 -dt 32x32x16 -nc 25 -ns 200 -ch 32 -cq 8 -mc '["Ex_488_Em_0_destriped_stitched_master_downsampled2x"]'

generate_data.Microglia.dev.02022024.4:
	python src/utils/data/generate_data.py -dr /media/share7/MYK/MorPheT/data/images/E10.5/20220416_17_26_08_10E_E10.5_GFP_CD206_aGFP -ts 020224_10E -sp /data_ssd3/data/training_data/cc/020224 -dt 32x32x16 -nc 25 -ns 200 -ch 32 -cq 8 -mc '["Ex_642_Em_2_destriped_stitched_master_downsampled2x"]'

