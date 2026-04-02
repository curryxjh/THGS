for sc in figurines; do
    python ../test_lerf.py -s <dataset path>/$sc -m <model path>/$sc --path_pred <mask saving path>
done

python ../scripts/eval_seg.py \
    --dataset lerf \
    --scene_list figurines \
    --path_pred output/render/lerf \
    --path_gt <gt mask path>

# Example:
# for sc in figurines ramen teatime waldo_kitchen; do python test_lerf.py -s data/lerf/$sc -m output/lerf/$sc --path_pred output/render/lerf; done
# python scripts/eval_seg.py --dataset lerf --scene_list figurines ramen teatime waldo_kitchen --path_pred output/render/lerf --path_gt data/lerf/label