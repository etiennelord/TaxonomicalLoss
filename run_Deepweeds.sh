#!/bin/bash
k="5 10 20 50 100 200"
# Start at seed 10
iseed=10


# Function to generate a random number between 1 and 10
get_random_increment() {
    echo $((RANDOM % 10 + 1))
}

for i in {1..10}; 
do
    increment=$(get_random_increment)
    
        # Increase the counter
        iseed=$((iseed + increment))
    for ik in $k
    do
        echo "i=$i /10 k=$ik seed=$iseed"
        rm -rf DeepWeeds_*
        python setup_k_dataset.py --data DeepWeeds --output DeepWeeds_$ik --seed=$iseed --k=$ik --complete --max_test=250
        CUDA_VISIBLE_DEVICES=0 python generic_mobilenet3.py --epoch=50 --data_dir DeepWeeds_$ik --output output_PlantSeedling_generic_mobilenet_$ik --batch_size=128 --imgsz=256 --seed=$iseed
        CUDA_VISIBLE_DEVICES=0 python generic_resnet50.py --epoch=50 --data_dir DeepWeeds_$ik --output output_PlantSeedling_generic_resnet50_$ik --batch_size=128 --imgsz=256 --seed=$iseed
        CUDA_VISIBLE_DEVICES=0 python train.py --epoch=50 --data DeepWeeds_$ik --tree DeepWeeds/taxonomy.txt --output output_DeepWeeds_mobilenet_taxonomic_$ik --batch_size=128 --imgsz=256 --lr=0.001 --model "mobilenet" --seed=$iseed --embeddings_size=64 --loss "taxonomic" 
        CUDA_VISIBLE_DEVICES=0 python train.py --epoch=50 --data DeepWeeds_$ik --tree DeepWeeds/taxonomy.txt --output output_DeepWeeds_resnet_taxonomic_$ik --batch_size=128 --imgsz=256 --lr=0.001 --model "resnet50" --seed=$iseed --embeddings_size=64 --loss "taxonomic" 
        CUDA_VISIBLE_DEVICES=0 python train.py --epoch=50 --data DeepWeeds_$ik --tree DeepWeeds/taxonomy.txt --output output_DeepWeeds_mobilenet_triplet_$ik --batch_size=128 --imgsz=256 --lr=0.001 --model "mobilenet" --seed=$iseed --embeddings_size=64 --loss "triplet" 
        CUDA_VISIBLE_DEVICES=0 python train.py --epoch=50 --data DeepWeeds_$ik --tree DeepWeeds/taxonomy.txt --output output_DeepWeeds_resnet_triplet_$ik --batch_size=128 --imgsz=256 --lr=0.001 --model "resnet50" --seed=$iseed --embeddings_size=64 --loss "triplet" 
        CUDA_VISIBLE_DEVICES=0 python train.py --epoch=50 --data DeepWeeds_$ik --tree DeepWeeds/taxonomy.txt --output output_DeepWeeds_mobilenet_htl_$ik --batch_size=128 --imgsz=256 --lr=0.001 --model "mobilenet" --seed=$iseed --embeddings_size=64 --loss "htl" 
        CUDA_VISIBLE_DEVICES=0 python train.py --epoch=50 --data DeepWeeds_$ik --tree DeepWeeds/taxonomy.txt --output output_DeepWeeds_resnet_htl_$ik --batch_size=128 --imgsz=256  --lr=0.001 --model "resnet50" --seed=$iseed --embeddings_size=64 --loss "htl" 
    done
done


