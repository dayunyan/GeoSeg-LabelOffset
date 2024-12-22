python inference.py \
-i ../data/xBD/test/images \
-c config/xBD/unetformer.py \
-o fig_results/xbd/unetformer \
-t 'lr' -ph 512 -pw 512 -b 4 -d "building"