
declare epochs=50
declare -a models=("MLP" "CNN" "LSTM" "Attention" "cnn-lstm-paper-experiments" )
declare -a layers=(1 2)
declare -a neurons=(64 128 256)

## now loop through the above array
for model in "${models[@]}"
do
  for layer in "${layers[@]}"
  do
    for neuron in "${neurons[@]}"
    do
      echo " Training ---------------------->  $model $layer $neuron   <----------------------------------------"

      export CUDA_VISIBLE_DEVICES=0 && source activate gpu && python /content/WSYCUHK_FDIA/Code/all_models.py  --model "$model" --n_epoch "$epochs" \
      --data_dir "/content/drive/MyDrive/openUAE/locational detection/data_ieee14_118_locational_detection/data118_1.mat" \
      --output_dir "/content/drive/MyDrive/openUAE/locational detection/tradeoff_analysis/IEEE118/" \
      --layers "$layer"\
      --neurons "$neuron"\
      --shape 180
   # or do whatever with individual element of the array
    done
  done
done
