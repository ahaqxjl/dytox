# imagenet100训练脚本，ImageNet数据没有准备好，因此暂时无法运行
nohup bash train.sh 0 --options options/data/imagenet100_10-10.yaml options/data/imagenet100_order1.yaml options/model/imagenet_dytox.yaml --name dytox --data-path MY_PATH_TO_DATASET --output-basedir PATH_TO_SAVE_CHECKPOINTS > output_imagenet_100_dytox_tome.20230418.log 2>&1 &

# meal_50
nohup bash train.sh 0 --options options/data/meal_2-2.yaml options/model/meal_dytox.yaml  --name dytox --data-path MY_PATH_TO_DATASET/meal300 --output-basedir PATH_TO_SAVE_CHECKPOINTS > output_meal_50_dytox_tome.`date +'%Y%m%d'`.log 2>&1 &