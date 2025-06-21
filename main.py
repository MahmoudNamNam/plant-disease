import io
import torch
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from fastapi.middleware.cors import CORSMiddleware

# ========= Labels =========
index_label = {
    0: "Pepper__bell___Bacterial_spot",
    1: "Pepper__bell___healthy",
    2: "Potato___Early_blight",
    3: "Potato___healthy",
    4: "Potato___Late_blight",
    5: "Tomato___Target_Spot",
    6: "Tomato___Tomato_mosaic_virus",
    7: "Tomato___Tomato_YellowLeaf_Curl_Virus",
    8: "Tomato_Bacterial_spot",
    9: "Tomato_Early_blight",
    10: "Tomato_healthy",
    11: "Tomato_Late_blight",
    12: "Tomato_Leaf_Mold",
    13: "Tomato_Septoria_leaf_spot",
    14: "Tomato_Spider_mites_Two_spotted_spider_mite"
}

# ========= Recommendations =========
recommendations = {
    'Pepper__bell___Bacterial_spot': (
        "• Remove infected leaves immediately.\n"
        "• Use copper-based fungicides weekly.\n"
        "• Avoid overhead irrigation.\n"
        "• Practice crop rotation."
    ),
    'Potato___Late_blight': (
        "• Use certified seed potatoes.\n"
        "• Apply fungicides during wet weather.\n"
        "• Improve air circulation.\n"
        "• Destroy infected plants."
    ),
    'Tomato_Leaf_Mold': (
        "• Improve ventilation.\n"
        "• Apply protective fungicides.\n"
        "• Remove infected leaves.\n"
        "• Avoid overhead watering."
    ),
    'Tomato___Tomato_YellowLeaf_Curl_Virus': (
        "• Control whiteflies.\n"
        "• Remove infected plants.\n"
        "• Use resistant varieties.\n"
        "• Sanitize tools."
    ),
    'Tomato_Bacterial_spot': (
        "• Use disease-free seeds.\n"
        "• Apply copper-based bactericides.\n"
        "• Avoid working with wet plants.\n"
        "• Rotate crops."
    ),
    'Tomato_Septoria_leaf_spot': (
        "• Apply protectant fungicides.\n"
        "• Remove infected leaves early.\n"
        "• Mulch soil.\n"
        "• Rotate crops."
    ),
    'Tomato_Spider_mites_Two_spotted_spider_mite': (
        "• Spray plants with water.\n"
        "• Introduce predatory mites.\n"
        "• Apply miticides if needed.\n"
        "• Avoid drought stress."
    ),
    'Tomato_Early_blight': (
        "• Apply fungicides at flowering.\n"
        "• Remove infected leaves.\n"
        "• Rotate crops.\n"
        "• Mulch and avoid overhead watering."
    ),
    'Tomato___Target_Spot': (
        "• Apply broad-spectrum fungicides.\n"
        "• Remove infected leaves.\n"
        "• Ensure proper spacing.\n"
        "• Sanitize field."
    ),
    'Potato___Early_blight': (
        "• Apply protectant fungicides.\n"
        "• Destroy plant debris.\n"
        "• Use disease-free seeds.\n"
        "• Ensure good nutrition."
    ),
    'Tomato_Late_blight': (
        "• Apply systemic fungicides.\n"
        "• Remove infected leaves quickly.\n"
        "• Avoid working when plants are wet.\n"
        "• Use resistant varieties."
    ),
    'Tomato___Tomato_mosaic_virus': (
        "• Disinfect tools.\n"
        "• Control aphids and thrips.\n"
        "• Use resistant varieties.\n"
        "• Avoid smoking near plants."
    )
}

# ========= Model Preparation =========
OUT_CLASSES = 15
model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classifier layers (only train head)
for param in model.classifier.parameters():
    param.requires_grad = True


num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, OUT_CLASSES)

model_path = "./model/best_model.pth"
device = torch.device("cpu") 
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ========= Image Preprocessing =========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========= Helper Functions =========
def clean_label(raw_label):
    label = raw_label.replace('__', ' ').replace('_', ' ').strip()
    label = label.replace('  ', ' ')
    return label

# ========= FastAPI =========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    raw_label = index_label[pred_class]
    clean = clean_label(raw_label)

    response = {
        "class_name_raw": raw_label,
        "class_name_clean": clean,
    }

    if "healthy" in raw_label.lower():
        response["status"] = "Plant is healthy"
    else:
        response["status"] = "Disease detected"
        recommendation = recommendations.get(raw_label, "No recommendation available.")
        response["recommendation"] = recommendation

    return response
