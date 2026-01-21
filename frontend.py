import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import CNN, predict

def load_model(path= 'CNN_model.pth'):
    model = CNN()
    load_model = torch.load(path,map_location=torch.device('cpu'))

    model.load_state_dict(load_model)
    model.eval()
    return model

transform = transforms.Compose(
                    [transforms.Resize((128,128)),
                    transforms.ToTensor()]
                    )

classes = ['building', 'forest', 'glacier', 'mountain', 'sea', 'street']


st.title("Image Prediction")

st.header("making the prediciton model via CNN")

st.write("this gubrish time jnr")



file = st.file_uploader("upload Image",type=['jpg','jpeg','png'])
if file is not None:
    img = Image.open(file)
    st.image(img, caption="actual Image", width='stretch')
    
    if st.button("Predict"):
        label, confidence, all_probs = predict(img)

        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{confidence*100:.2f}%**")

        # Show all class probabilities
        st.subheader("Class Probabilities")
        for i, cls in enumerate(classes):
            st.write(f"{cls}: {(all_probs[0][i].detach().cpu().item() * 100):.2f}")
        