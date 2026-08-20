import os
import torch

from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv

from utils.utils import adaptive_instance_normalization
from utils.models import VGGEncoder, VGGDecoder


load_dotenv()

app = Flask(__name__)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key")
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

Bootstrap(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

class UploadForm(FlaskForm):

    content = FileField("Content image")
    style = FileField("Style image")

    content_path = HiddenField()
    style_path = HiddenField()

    alpha = FloatField(
        "Alpha",
        default=1.0
    )

    submit = SubmitField("Transfer Style")


# Render deployment is CPU based.
device = torch.device("cpu")

# Limit CPU threads to reduce memory/CPU pressure
torch.set_num_threads(2)

print("Loading VGG encoder...")

encoder = VGGEncoder(
    "models/vgg_normalized.pth"
).to(device)

print("Loading decoder...")

decoder = VGGDecoder().to(device)

decoder.load_state_dict(
    torch.load(
        "models/decoder_10.pth",
        map_location=device
    )
)

encoder.eval()
decoder.eval()

print("Models loaded successfully.")

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )

def style_transfer(
    content_image,
    style_image,
    encoder,
    decoder,
    alpha,
    device
):

    # Keep images small because Render CPU/RAM is limited
    content_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Convert PIL → Tensor
    content_tensor = content_transform(
        content_image
    ).unsqueeze(0).to(device)

    style_tensor = style_transform(
        style_image
    ).unsqueeze(0).to(device)

    # No gradients needed during inference
    with torch.inference_mode():

        # Extract content features
        content_features = encoder(
            content_tensor,
            is_test=True
        )

        # Extract style features
        style_features = encoder(
            style_tensor,
            is_test=True
        )

        # Adaptive Instance Normalization
        stylized_features = adaptive_instance_normalization(
            content_features,
            style_features
        )

        # Alpha blending
        stylized_features = (
            alpha * stylized_features
            + (1 - alpha) * content_features
        )

        # Decode features into image
        stylized_image = decoder(
            stylized_features
        )

    return stylized_image

def save_image(image, path):

    image = image.detach().cpu()

    image = image.squeeze(0)

    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(image)

    image.save(path)

@app.route("/", methods=["GET", "POST"])
def index():

    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if (
            form.content.data
            and form.content.data.filename
        ):

            if allowed_file(
                form.content.data.filename
            ):

                content_filename = secure_filename(
                    form.content.data.filename
                )

                content_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    content_filename
                )

                form.content.data.save(
                    content_path
                )

                form.content_path.data = (
                    content_filename
                )

            else:

                error = "Invalid content image format."


        if (
            form.style.data
            and form.style.data.filename
        ):

            if allowed_file(
                form.style.data.filename
            ):

                style_filename = secure_filename(
                    form.style.data.filename
                )

                style_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    style_filename
                )

                form.style.data.save(
                    style_path
                )

                form.style_path.data = (
                    style_filename
                )

            else:

                error = "Invalid style image format."


        if not content_filename:

            content_filename = (
                form.content_path.data
            )

        if not style_filename:

            style_filename = (
                form.style_path.data
            )


        if not content_filename:

            error = "Please upload a content image."

        elif not style_filename:

            error = "Please upload a style image."


        if (
            content_filename
            and style_filename
            and not error
        ):

            content_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                content_filename
            )

            style_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                style_filename
            )

            try:

                print("Opening images...")

                content_image = Image.open(
                    content_path
                ).convert("RGB")

                style_image = Image.open(
                    style_path
                ).convert("RGB")

                # Keep alpha between 0 and 1
                alpha = float(
                    form.alpha.data or 1.0
                )

                alpha = max(
                    0.0,
                    min(1.0, alpha)
                )

                print("Starting style transfer...")

                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    encoder,
                    decoder,
                    alpha,
                    device
                )

                print("Style transfer completed.")

                result_filename = (
                    "stylized_" + content_filename
                )

                result_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    result_filename
                )

                save_image(
                    stylized_image,
                    result_path
                )

                result_image = result_filename

                print(
                    "Result saved:",
                    result_path
                )

            except Exception as e:

                print(
                    "STYLE TRANSFER ERROR:",
                    repr(e)
                )

                error = str(e)

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )

@app.route("/uploads/<filename>")
def send_image(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


@app.route("/examples/<path:filename>")
def send_example(filename):

    return send_from_directory(
        "examples",
        filename
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )