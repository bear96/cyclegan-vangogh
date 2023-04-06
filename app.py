import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import gc


t = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

@st.cache_resource
def load_model():
    model = torch.jit.load('checkpoint/CycleGAN_Generator.pt')
    return model

Gen_BA = load_model()


def predict(im,Gen_BA):
    w,h = im.size
    if h or w >=500:
        scale_factor = 500/max(h,w)
        height = int(h * scale_factor)
        width = int(w * scale_factor)
        im = transforms.Resize((height,width))(im)

    input = t(im)
    Gen_BA.eval()
    output = Gen_BA(input.unsqueeze(0))
    output = output/2 +0.5
    return (transforms.ToPILImage()(output.squeeze(0)))

st.title(body = "Van Gogh style transfer using CycleGAN.")
st.markdown("""This app has been built using a machine learning model called CycleGAN. For more information on CycleGAN, 
            you can read this [paper](https://arxiv.org/abs/1703.10593). The dataset used has been sourced from 
            [UC Berkeley](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/) and other open sourced datasets on Kaggle.
            If you're interested in the code behind the project, the project is hosted on my [GitHub](https://github.com/bear96/cyclegan-vangogh).
            Alternatively, you can choose to clone into the repository using  

            git clone https://github.com/bear96/cyclegan-vangogh.git""")
st.markdown("""The model hasn't yet been trained to it's limits. It performs poorly on people and objects such as cars. 
However, the model performs reasonably well on pictures of scenery or nature. The dataset that was used to train the model mostly
contained pictures of scenery, hence this handicap.""")
st.warning("""Please note that the app may crash if you provide it with an image that's too large with very high resolution. \
    Typically, the model seems to perform well on images which are less than or equal to 1MB in size. If you're using this from your phone,
    it is possible that the model may not work if you provide an image with a .PNG extension. I'm unaware of why this error happens, since
    if you try to provide the same image via laptop, it works perfectly fine. I'm working on solving this issue.""")
vg = Image.open("style/Self-Portrait_(1889)_by_Vincent_van_Gogh.jpg")
vg = vg.resize([int(vg.size[0]*0.2),int(vg.size[1]*0.2)])
with st.sidebar:
    st.image(vg,caption="Self portrait by Van Gogh, 1889")

st.sidebar.title("Who doesn't know about the legendary post-impressionist painter, Vincent Van Gogh?")
with st.sidebar:
    st.markdown("""
    *Vincent Van Gogh, the red-haired Dutch master of the paintbrush, was a genius whose artistic 
    legacy continues to inspire people around the world. He is best known for his breathtakingly beautiful paintings, 
    which were so ahead of their time that many people thought he was just a little bit crazy. But don't worry, 
    it wasn't his fault - he was just misunderstood! From his famous Sunflowers series to his unforgettable Starry Night, 
    Van Gogh's work is a testament to the power of imagination, creativity, and a healthy dose of eccentricity.*
    """)

file = st.file_uploader(label="Upload your image",type=['.png','.jpg','jpeg'])

if not file:
    st.warning("Please upload an image")
    gc.collect()
    st.stop()
else:
    image = file.read()
    st.caption("Your image.")
    st.image(image, use_column_width=True)
    img = Image.open(io.BytesIO(image))
    pred_button = st.button("Generate")

if pred_button:
    with st.spinner("Generating. Please wait..."):
        gen_image = predict(img,Gen_BA)
        del img
        st.caption("Generated image.")
        st.image(gen_image)
    del gen_image
    del image
