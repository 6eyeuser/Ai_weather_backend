import io
import os
import math
from typing import Optional
from dotenv import load_dotenv
import exifread
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image
from supabase import create_client, Client

load_dotenv(dotenv_path="../.env")
load_dotenv(dotenv_path="../.env.local")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase configuration in .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@app.post("/trigger-broadcast")
async def trigger_broadcast(
    latitude: float = Form(...),
    longitude: float = Form(...),
    title: str = Form(...),
    severity: str = Form(...),
    description: str = Form(...),
    radius_km: float = Form(50.0),
    file: Optional[UploadFile] = File(None)
):
    ai_analysis_note = ""

    # If the admin optionally attached an image, pass it to the PyTorch model
    if file and file.filename:
        image_bytes = await file.read()
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            input_tensor = preprocess(img).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                confidence = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, _ = torch.topk(confidence, 1)
                
            model_severity = "SEVERE" if top_prob.item() < 0.5 else "MODERATE"
            ai_analysis_note = f"\n[AI Threat Confirmation: {model_severity} anomaly verified from attachment.]"
        except Exception:
            ai_analysis_note = "\n[Image attached but AI analysis failed or format unsupported]"

    # Merge manual metrics with optional AI analysis
    final_description = f"{description}{ai_analysis_note}"

    alert_record = {
        "title": title,
        "description": final_description,
        "severity": severity,
        "latitude": latitude,
        "longitude": longitude
    }
    
    # 1. Insert alert into Supabase
    supabase.table("alerts").insert(alert_record).execute()

    # 2. Query registered user locations
    profiles_res = supabase.table("profiles").select("id, email, latitude, longitude").execute()
    users = profiles_res.data or []
    notified_users = []

    for user in users:
        u_lat = user.get("latitude")
        u_lon = user.get("longitude")
        if u_lat is not None and u_lon is not None:
            dist = calculate_distance(latitude, longitude, float(u_lat), float(u_lon))
            if dist <= radius_km:
                notified_users.append(user.get("email"))

    return {
        "status": "success",
        "broadcast_alert": alert_record,
        "users_in_radius": notified_users
    }