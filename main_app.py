import streamlit as st
import pandas as pd
import torch
import torchvision
from PIL import Image
from transformers import ViTForImageClassification

emotion_mapping = {0: 'Angry 😠', 1: 'Disgust 😣', 2: 'Fear 😱', 3: 'Happy 😀', 4: 'Sad 😢', 5: 'Surprise 😲', 6: 'Neutral 😐'}


def predict(image, model):
            
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = transform(image)
    image = image.unsqueeze(0)

    model.eval()
    logits = model(image)

    y_prob = torch.nn.functional.softmax(logits[0], dim=-1)

    df_result = pd.DataFrame(
        y_prob.T.detach().numpy(), 
        columns=['Probability'],
        index=emotion_mapping.values()
    )

    return df_result
    

def main():
    # load model with cpu
    model = ViTForImageClassification.from_pretrained('yosshstd/vit-fer2013', device_map='cpu')
    
    st.title('Face2Emoji App')
    st.sidebar.write('# Face2Emoji App')
    st.sidebar.write('')

    img_source = st.sidebar.radio('Please select the source of the facial expression image.',
                                  ('Upload the image', 'Capture the image', 'Select a sample image'))
    
    if img_source == 'Upload the image':
        img_file = st.sidebar.file_uploader('Please upload the facial expression image.', type=['jpg', 'png', 'jpeg'])
    elif img_source == 'Capture the image':
        img_file = st.camera_input('Please capture the facial expression image.')
    else:
        img_file = st.sidebar.radio(
            'Please elect a sample image. (Free to use under the Unsplash License)',
            ('Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6')
        )

        if img_file == 'Sample 1':
            img_file = 'image/sample1.jpg'
        elif img_file == 'Sample 2':
            img_file = 'image/sample2.jpg'
        elif img_file == 'Sample 3':
            img_file = 'image/sample3.jpg'
        elif img_file == 'Sample 4':
            img_file = 'image/sample4.jpg'
        elif img_file == 'Sample 5':
            img_file = 'image/sample5.jpg'
        elif img_file == 'Sample 6':
            img_file = 'image/sample6.jpg'
        else:
            img_file = None
        


    if img_file is not None:
        with st.spinner('loading・・・'):
            img = Image.open(img_file)
            st.image(img, caption='Facial expression image', use_column_width=True)
            st.write('')

            results = predict(img, model)

            st.subheader('Probs of each emotion:')

            # Display bar chart
            st.bar_chart(data=results)

            # Display emotion
            emotion = results.idxmax()[0]
            st.write('This image is classified as:')

            # Display Big Emoji
            st.write(f'## {emotion}')
            


    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')

    github = 'https://github.com/yosshstd/Face2Emoji'
    st.sidebar.caption('This app is powered by PyTorch and Hugging Face 🤗.  \n \
                        The model was fine-tuned on FER2013 dataset.  \n \
                        The sample images are free to use under the Unsplash License.  \n \
                        The source code is available on [GitHub](%s).' % github)


if __name__ == '__main__':
    main()
