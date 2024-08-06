# # Full Dataset, no compression

# # SEED 123
# # Marina + Clipping
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --clip-update --clip-mult 1.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --clip-update --clip-mult 1.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --clip-update --clip-mult 1.

# # Marina + Clipping (.1) + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1

# # Marina + Clipping (1.) + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 1.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 1.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 1.

# # Marina + Clipping (10.) + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 10.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 10.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 10.

# # Marina + Clipping (100.) + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 100.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 100.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 100.

# # Marina + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2

# # Mom_Sgd + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model mom_sgd --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model mom_sgd --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model mom_sgd --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2

# # SEED 124
# # Marina + Clipping
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --clip-update --clip-mult 1.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --clip-update --clip-mult 1.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --clip-update --clip-mult 1.

# # Marina + Clipping (.1) + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1

# # Marina + Clipping (1.) + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 1.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 1.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 1.

# # Marina + Clipping (10.) + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 10.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 10.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 10.

# # Marina + Clipping (100.) + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 100.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 100.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 100.

# # Marina + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2

# # Mom_Sgd + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model mom_sgd --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model mom_sgd --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model mom_sgd --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2


# # SEED 125
# # Marina + Clipping
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --clip-update --clip-mult 1.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --clip-update --clip-mult 1.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --clip-update --clip-mult 1.

# # Marina + Clipping (.1) + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1

# # Marina + Clipping (1.) + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 1.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 1.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 1.

# # Marina + Clipping (10.) + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 10.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 10.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 10.

# # Marina + Clipping (100.) + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 100.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 100.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 100.

# # Marina + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2

# # Mom_Sgd + Partial Participation
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model mom_sgd --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model mom_sgd --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model mom_sgd --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2


# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.1 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --clip-update --clip-mult 1.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.01 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --clip-update --clip-mult 1.
# CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack SHB --lr 0.001 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --clip-update --clip-mult 1.

CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack BF --lr 0.01 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1
CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack ALIE --lr 0.1 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1
CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack LF --lr 0.01 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1
CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack IPM --lr 0.1 --seed 123 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1

CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack BF --lr 0.01 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1
CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack ALIE --lr 0.1 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1
CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack LF --lr 0.01 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1
CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack IPM --lr 0.1 --seed 124 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1

CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack BF --lr 0.01 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1
CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack ALIE --lr 0.1 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1
CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack LF --lr 0.01 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1
CUDA_VISIBLE_DEVICES=0 python run_a9a.py --model marina --agg cm --bucketing 2 --attack IPM --lr 0.1 --seed 125 -b 32 --test-batch-size 32561 -e 10 -n 20 -f 5 --use-cuda --eval-every 1 --compression none --full-dataset --partial-participation --partial-participation-ratio 0.2 --clip-update --clip-mult 0.1