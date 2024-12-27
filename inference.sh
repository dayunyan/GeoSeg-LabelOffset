# python inference.py \
# -i ../data/xBD/test/images \
# -c config/xBD/unetformer.py \
# -o fig_results/xbd/unetformer \
# -t 'lr' -ph 512 -pw 512 -b 4 -d "building"

python inference_advent.py \
-i ../data/segmentation/Turkey/Islahiye/pre/test/images \
-c config/xBD_Teq/unetformer.py \
-o fig_results/xbd_teq/advent \
-t 'lr' -ph 512 -pw 512 -b 4 -d "building"