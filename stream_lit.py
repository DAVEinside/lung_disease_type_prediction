import streamlit as st
import torch
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np
import pandas as pd
from model import CNN

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Applying Machine Learning to Diagnose\
 			Lung Disease using Chest X-rays')
st.header('SEM-6 Mini Project')
st.subheader('KJ Somaiya College of Engineering')

st.markdown("This Model is a CNN model that, when provided an chest X-ray can predict with an 90+% accuray if the patient has pneumonia and if yes then its type ")



def layout(*args):
    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(position="fixed", left=0, bottom=0, margin=px(0, 0, 0, 0), width=percent(100), color="white", text_align="center", height=percent(3.8), opacity=1)
    style_hr = styles(display="block", margin=px(0, 0, 0, 0), border_style="inset", border_width=px(2))
    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Made by Nimit Dave "
    ]
    layout(*myargs)


footer()

def img_to_torch(pil_image):
	img = pil_image.convert('L')
	x = torchvision.transforms.functional.to_tensor(img)
	x = torchvision.transforms.functional.resize(x, [150, 150])
	x.unsqueeze_(0)
	return x

def predict(image, model):
	x = img_to_torch(image)
	pred = model(x)
	pred = pred.detach().numpy()

	df = pd.DataFrame(data=pred[0], index=['Bacterial', 'Normal', 'Viral'], columns=['confidence'])

	st.write(f'''### Bacterial Probability :  **{np.round(pred[0][0]*100, 3)}%**''')
	st.write(f'''### Viral Probability : **{np.round(pred[0][2]*100, 3)}%**''')
	st.write(f'''### Normal Probability: **{np.round(pred[0][1]*100, 3)}%**''')
	st.write('')
	st.bar_chart(df)

PATH_TO_MODEL = './0.906_loss0.081.pt'
model = torch.load(PATH_TO_MODEL)
model.eval()

uploaded_file = st.file_uploader('Upload X-ray image to be Diagnosed ', type=['jpeg', 'jpg', 'png'])

if uploaded_file is not None:
	image = Image.open(uploaded_file)
	st.image(image, use_column_width=True)

	if st.button('Run Analysis'):
		predict(image, model)