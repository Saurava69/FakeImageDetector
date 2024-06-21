import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Hide Streamlit's default icons and fork option
st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model
model_path = 'resnet50_v3_model.pth'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Instantiate the model (assuming it's a ResNet-50)
model = models.resnet50(pretrained=False)
num_classes = 2  # Assuming 2 classes: Real Image and AI-generated Image
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Modify fully connected layer

# Load model state dictionary directly from checkpoint
model.load_state_dict(checkpoint)

# Set model to evaluation mode
model.eval()

# Define transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0][class_idx]
        target.backward()

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor, image

def show_cam_on_image(image, mask):
    image = np.array(image)
    image = cv2.resize(image, (mask.shape[1], mask.shape[0]))  # Resize image to match CAM size
    image = image / 255.0  # Normalize to [0, 1] for overlay

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def classify_image(image_path):
    # Preprocess the input image
    input_tensor, image = preprocess_image(image_path)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)  # Get class probabilities
        confidence_score, predicted_class = torch.max(probabilities, 1)

    # Generate Grad-CAM
    grad_cam = GradCAM(model, model.layer4[-1])
    cam = grad_cam.generate_cam(input_tensor, predicted_class.item())

    # Show the CAM on the image
    cam_image = show_cam_on_image(image, cam)

    return image, cam_image, predicted_class.item(), confidence_score.item()

st.title("Image Classification and Grad-CAM Visualization")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f'Classifying {uploaded_file.name}...'):
            image, cam_image, predicted_class, confidence_score = classify_image(uploaded_file)
            prediction_label = "Real Image" if predicted_class == 0 else "AI-generated Image"

            st.write(f'**Prediction for {uploaded_file.name}:** {prediction_label} (Confidence: {confidence_score:.3f})')
            st.image(image, caption=f'Original Image - {uploaded_file.name}', use_column_width=True)
            st.image(cam_image, caption='Grad-CAM', use_column_width=True)
            st.write("---")
