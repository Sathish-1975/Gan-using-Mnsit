import torch
import os
import io
import base64
import traceback
from flask import Flask, render_template, jsonify
from model import Generator, Discriminator, DigitClassifier
from PIL import Image
import torchvision.transforms as transforms
from waitress import serve

app = Flask(__name__)

# Parameters (should match training)
latent_dim = 100
img_shape = (1, 28, 28)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the generator and discriminator models
generator = Generator(latent_dim=latent_dim, img_shape=img_shape).to(device)
discriminator = Discriminator(img_shape=img_shape).to(device)
classifier = DigitClassifier().to(device)
model_path_g = "generator.pth"
model_path_d = "discriminator.pth"
model_path_c = "classifier.pth"
model_loaded = False

def load_models():
    global model_loaded
    if model_loaded:
        return True
    if os.path.exists(model_path_g) and os.path.exists(model_path_d) and os.path.exists(model_path_c):
        try:
            generator.load_state_dict(torch.load(model_path_g, map_location=device))
            discriminator.load_state_dict(torch.load(model_path_d, map_location=device))
            classifier.load_state_dict(torch.load(model_path_c, map_location=device))
            generator.eval()
            discriminator.eval()
            classifier.eval()
            model_loaded = True
            print("Models loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
    return False

# Initial load attempt
load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    print("Received generation request...")
    try:
        if not load_models():
            return jsonify({"error": "Models not found. Please train the model first."}), 404

        # Generate image
        with torch.no_grad():
            z = torch.randn(1, latent_dim, device=device)
            gen_img_raw = generator(z)
            gen_img = gen_img_raw.cpu().squeeze(0)  # (1, 28, 28) -> (28, 28)
            
            # Calculate discriminator score (realness)
            d_score = discriminator(gen_img_raw).item()
            confidence = d_score * 100

            # Predict digit
            # Classifier expects normalized [0, 1] or same as training (we used -1 to 1 normalize in train_classifier)
            # Generator outputs Tanh [-1, 1], which matches our training normalization
            class_output = classifier(gen_img_raw)
            prediction = torch.argmax(class_output, dim=1).item()
            class_conf = torch.exp(class_output)[0][prediction].item() * 100

        # Normalize map to [0, 1]
        gen_img = (gen_img + 1) / 2
        
        # Convert to PIL Image
        transform = transforms.ToPILImage()
        img = transform(gen_img)
        
        # Save to buffer
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return jsonify({
            "image": img_str,
            "metrics": {
                "realness": round(confidence, 2),
                "prediction": prediction,
                "classifier_confidence": round(class_conf, 2),
                "latent_dim": latent_dim,
                "device": str(device).upper()
            }
        })
    except Exception as e:
        print(f"ERROR during generation: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting GAN Flask App on http://localhost:5000")
    serve(app, host='0.0.0.0', port=5000)
