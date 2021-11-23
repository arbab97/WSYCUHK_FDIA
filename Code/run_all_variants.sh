
#To change the bus: data_dir, output_dir, shape
declare epochs=50
declare -a variants=(1 2 3 4 5 )
declare -a models=("MLP" "CNN" "LSTM" "Attention" "cnn-lstm-paper-experiments" )
declare -a layers=(2)
declare -a neurons=(128)

## now loop through the above array
for variant in "${variants[@]}"
do
  for model in "${models[@]}"
  do
    for layer in "${layers[@]}"
    do
      for neuron in "${neurons[@]}"
      do
        echo " Training ---------------------->  $model $layer $neuron   <----------------------------------------"
        
        export CUDA_VISIBLE_DEVICES=0 && source activate gpu && python /content/WSYCUHK_FDIA/Code/all_models.py  --model "$model" --n_epoch "$epochs" \
        --data_dir "/content/drive/MyDrive/openUAE/locational detection/data_ieee14_118_locational_detection/data14_${variant}.mat" \
        --output_dir "/content/drive/MyDrive/openUAE/locational detection/variant_analysis/IEEE14/${variant}/" \
        --layers "$layer"\
        --neurons "$neuron"\
        --shape 19
    # or do whatever with individual element of the array
      done
    done
  done
done