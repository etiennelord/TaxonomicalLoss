k="20 50 100"
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
        echo "i=$i / 100 k=$ik seed=$iseed BBCH10"
        rm -rf WPD_BBCH10_$ik
        python setup_k_20sept2024.py --data BBCH10 --output WPD_BBCH10_$ik --seed=$iseed --k=$ik
        CUDA_VISIBLE_DEVICES=0 python generic_mobilenet3.py --epoch=50 --data_dir WPD_BBCH10_$ik --output output_WPD_generic_mobilenet_BBCH10_$ik --batch_size=32 --imgsz=512 --seed=$iseed
        CUDA_VISIBLE_DEVICES=0 python generic_resnet50.py --epoch=50 --data_dir WPD_BBCH10_$ik --output output_WPD_generic_resnet50_BBCH10_$ik --batch_size=32 --imgsz=512 --seed=$iseed
        CUDA_VISIBLE_DEVICES=0 python train.py --epoch=50 --data WPD_BBCH10_$ik --tree taxonomy.txt --output output_WPD_mobilenet_triplet_BBCH10_$ik --batch_size=16 --imgsz=512 --lr=0.001 --model "mobilenet" --seed=$iseed --embeddings_size=64 --loss "triplet" --no-val
        CUDA_VISIBLE_DEVICES=0 python train.py --epoch=50 --data WPD_BBCH10_$ik --tree taxonomy.txt --output output_WPD_mobilenet_taxonomic_BBCH10_$ik --batch_size=16 --imgsz=512 --lr=0.001 --model "mobilenet" --seed=$iseed --embeddings_size=64 --loss "taxonomic" --no-val
        
        echo "i=$i / 100 k=$ik seed=$iseed BBCH11"
        rm -rf WPD_BBCH11_$ik
        python setup_k_20sept2024.py --data BBCH11 --output WPD_BBCH11_$ik --seed=$iseed --k=$ik
        CUDA_VISIBLE_DEVICES=0 python generic_mobilenet3.py --epoch=50 --data_dir WPD_BBCH11_$ik --output output_WPD_generic_mobilenet_BBCH11_$ik --batch_size=32 --imgsz=512 --seed=$iseed
        CUDA_VISIBLE_DEVICES=0 python generic_resnet50.py --epoch=50 --data_dir WPD_BBCH11_$ik --output output_WPD_generic_resnet50_BBCH11_$ik --batch_size=32 --imgsz=512 --seed=$iseed
        CUDA_VISIBLE_DEVICES=0 python train.py --epoch=50 --data WPD_BBCH11_$ik --tree taxonomy.txt --output output_WPD_mobilenet_triplet_BBCH11_$ik --batch_size=16 --imgsz=512 --lr=0.001 --model "mobilenet" --seed=$iseed --embeddings_size=64 --loss "triplet" --no-val
        CUDA_VISIBLE_DEVICES=0 python train.py --epoch=50 --data WPD_BBCH11_$ik --tree taxonomy.txt --output output_WPD_mobilenet_taxonomic_BBCH11_$ik --batch_size=16 --imgsz=512 --lr=0.001 --model "mobilenet" --seed=$iseed --embeddings_size=64 --loss "taxonomic" --no-val
        
        echo "i=$i / 100 k=$ik seed=$iseed BBCH12"
        rm -rf WPD_BBCH12_$ik
        python setup_k_20sept2024.py --data BBCH12 --output WPD_BBCH12_$ik --seed=$iseed --k=$ik
        CUDA_VISIBLE_DEVICES=0 python generic_mobilenet3.py --epoch=50 --data_dir WPD_BBCH12_$ik --output output_WPD_generic_mobilenet_BBCH12_$ik --batch_size=32 --imgsz=512 --seed=$iseed
        CUDA_VISIBLE_DEVICES=0 python generic_resnet50.py --epoch=50 --data_dir WPD_BBCH12_$ik --output output_WPD_generic_resnet50_BBCH12_$ik --batch_size=32 --imgsz=512 --seed=$iseed
        CUDA_VISIBLE_DEVICES=0 python train.py --epoch=50 --data WPD_BBCH12_$ik --tree taxonomy.txt --output output_WPD_mobilenet_triplet_BBCH12_$ik --batch_size=16 --imgsz=512 --lr=0.001 --model "mobilenet" --seed=$iseed --embeddings_size=64 --loss "triplet" --no-val
        CUDA_VISIBLE_DEVICES=0 python train.py --epoch=50 --data WPD_BBCH12_$ik --tree taxonomy.txt --output output_WPD_mobilenet_taxonomic_BBCH12_$ik --batch_size=16 --imgsz=512 --lr=0.001 --model "mobilenet" --seed=$iseed --embeddings_size=64 --loss "taxonomic" --no-val
        
	done
done


