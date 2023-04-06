# CycleGAN implementation in Pytorch

This is a Pytorch implementation of CycleGAN to transfer Van Gogh's art style onto pictures.
To download the dataset, run ./download.sh in your terminal before executing main.py. Please install the required packages using 
```pip3 install -r requirements.txt```

The model has now been deployed on [Streamlit](https://bear96-cyclegan-vangogh-app-kimcyr.streamlit.app/). Please try it out and let me know if you find any bugs, or if it is crashing for you after multiple tries. Also, if it is performing badly on some images and well on others, please let me know!

## Results of training

Loss graph coming

## Example Images during training

### After 1 epoch.
![After 1 epoch](/outputs/900.png)

### After 3 epochs. 

![After 3 epochs](/outputs/4000.png)

### After 5 epochs. 
![After 5 epochs](/outputs/5300.png)

Final model is not yet ready. Model is being trained on other kinds of data (such as faces dataset from kaggle, etc) to improve robustness as the model performs poorly on people and objects such as cars. Model performs well on pictures of scenery or nature.
