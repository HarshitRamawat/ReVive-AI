
import hashlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps


# ==================================================
# 1. PROJECT PATHS AND SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "third_party" / "Real-ESRGAN"
MODEL_SCRIPT = MODEL_DIR / "inference_realesrgan.py"
MODEL_WEIGHTS = MODEL_DIR / "weights" / "RealESRGAN_x4plus.pth"

INPUT_DIR = BASE_DIR / "input" / "uploads"
OUTPUT_DIR = BASE_DIR / "output" / "enhanced"

SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]

MAX_INPUT_DIMENSION = 768
ENHANCEMENT_TIMEOUT = 300
MODEL_TILE_SIZE = 256


# ==================================================
# 2. STREAMLIT CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ReVive AI",
    page_icon="✨",
    layout="wide"
)


# ==================================================
# 3. IMAGE PREPARATION FUNCTION
# ==================================================

def prepare_image(image_bytes, filename, destination):

    with Image.open(io.BytesIO(image_bytes)) as image:

        # Correct image orientation using EXIF information.
        image = ImageOps.exif_transpose(image)

        original_size = image.size

        # Resize proportionally to reduce CPU workload.
        image.thumbnail(
            (MAX_INPUT_DIMENSION, MAX_INPUT_DIMENSION),
            Image.Resampling.LANCZOS
        )

        prepared_size = image.size

        extension = Path(filename).suffix.lower()

        if extension in [".jpg", ".jpeg"]:

            image = image.convert("RGB")

            image.save(
                destination,
                format="JPEG",
                quality=95,
                optimize=True
            )

        else:

            if image.mode not in ["RGB", "RGBA"]:
                image = image.convert("RGB")

            image.save(
                destination,
                format="PNG",
                optimize=True
            )

    return original_size, prepared_size


# ==================================================
# 4. FIND GENERATED IMAGE
# ==================================================

def find_generated_images(folder):

    return [
        file
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]


# ==================================================
# 5. APPLICATION HEADER
# ==================================================

st.title("✨ ReVive AI")

st.subheader("AI-Powered Image Enhancement")

st.write(
    "Upload your image and let our AI improve "
    "its resolution and details."
)

st.caption(
    "Large images are automatically resized "
    "to improve processing performance."
)

st.divider()


# ==================================================
# 6. IMAGE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=SUPPORTED_FORMATS
)


if uploaded_file is None:

    st.info(
        "Upload a JPG, JPEG, or PNG image to get started."
    )

else:

    original_name = Path(uploaded_file.name).name

    image_bytes = uploaded_file.getvalue()

    image_id = hashlib.sha256(image_bytes).hexdigest()

    st.success(
        f"Uploaded successfully: {original_name}"
    )

    # Clear previous results when a different image is uploaded.
    if st.session_state.get("image_id") != image_id:

        st.session_state["image_id"] = image_id

        st.session_state.pop("enhanced_image", None)
        st.session_state.pop("enhanced_name", None)

    # ==================================================
    # 7. ORIGINAL AND ENHANCED IMAGE DISPLAY
    # ==================================================

    original_col, enhanced_col = st.columns(
        2,
        gap="large"
    )

    with original_col:

        st.subheader("Original Image")

        st.image(
            image_bytes,
            caption="Uploaded Image",
            width=400
        )

    with enhanced_col:

        st.subheader("Enhanced Image")

        if "enhanced_image" in st.session_state:

            st.image(
                st.session_state["enhanced_image"],
                caption="AI Enhanced Image",
                width=400
            )

        else:

            st.info(
                "Your enhanced image will appear here "
                "after processing."
            )

    st.divider()

    # ==================================================
    # 8. ENHANCE IMAGE BUTTON
    # ==================================================

    if st.button(
        "✨ Enhance Image",
        type="primary",
        use_container_width=True
    ):

        # Check whether required model files exist.
        missing_files = []

        if not MODEL_SCRIPT.is_file():
            missing_files.append(str(MODEL_SCRIPT))

        if not MODEL_WEIGHTS.is_file():
            missing_files.append(str(MODEL_WEIGHTS))

        if missing_files:

            st.error("Required Real-ESRGAN files are missing.")

            for missing_file in missing_files:
                st.code(missing_file)

        else:

            # Remove an old result before starting another run.
            st.session_state.pop("enhanced_image", None)
            st.session_state.pop("enhanced_name", None)

            INPUT_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            try:

                with st.spinner(
                    "ReVive AI is enhancing your image. "
                    "Please wait..."
                ):

                    with tempfile.TemporaryDirectory(
                        dir=INPUT_DIR
                    ) as input_temp:

                        with tempfile.TemporaryDirectory(
                            dir=OUTPUT_DIR
                        ) as output_temp:

                            input_path = (
                                Path(input_temp) / original_name
                            )

                            output_path = Path(output_temp)

                            # Prepare and resize the uploaded image.
                            original_size, prepared_size = prepare_image(
                                image_bytes,
                                original_name,
                                input_path
                            )

                            if prepared_size != original_size:

                                st.info(
                                    f"Image resized from "
                                    f"{original_size[0]} x {original_size[1]} "
                                    f"to {prepared_size[0]} x "
                                    f"{prepared_size[1]}."
                                )

                            # Real-ESRGAN inference command.
                            command = [

                                sys.executable,

                                str(MODEL_SCRIPT),

                                "-n",
                                "RealESRGAN_x4plus",

                                "-i",
                                str(input_path),

                                "-o",
                                str(output_path),

                                "--fp32",

                                "-t",
                                str(MODEL_TILE_SIZE)
                            ]

                            # Execute model with a maximum time limit.
                            try:

                                result = subprocess.run(

                                    command,

                                    cwd=str(MODEL_DIR),

                                    capture_output=True,

                                    text=True,

                                    timeout=ENHANCEMENT_TIMEOUT,

                                    check=False
                                )

                            except subprocess.TimeoutExpired:

                                st.error(
                                    "Enhancement exceeded the "
                                    "5-minute time limit."
                                )

                                st.warning(
                                    "The cloud CPU may be throttled "
                                    "or the image may be too demanding. "
                                    "Try again later or use GPU hosting."
                                )

                                result = None

                            if result is not None:

                                # Check model execution status.
                                if result.returncode != 0:

                                    st.error(
                                        "Real-ESRGAN failed to enhance "
                                        "the image."
                                    )

                                    error_message = (
                                        result.stderr.strip()
                                        or result.stdout.strip()
                                        or "No error details were returned."
                                    )

                                    st.code(
                                        error_message[-12000:]
                                    )

                                else:

                                    generated_files = (
                                        find_generated_images(output_path)
                                    )

                                    if not generated_files:

                                        st.error(
                                            "Model execution finished, "
                                            "but no enhanced image was found."
                                        )

                                        if result.stdout:
                                            st.code(result.stdout[-12000:])

                                    else:

                                        enhanced_file = generated_files[0]

                                        enhanced_bytes = (
                                            enhanced_file.read_bytes()
                                        )

                                        enhanced_name = (
                                            Path(original_name).stem
                                            + "_enhanced"
                                            + enhanced_file.suffix.lower()
                                        )

                                        # Store enhanced result.
                                        st.session_state[
                                            "enhanced_image"
                                        ] = enhanced_bytes

                                        st.session_state[
                                            "enhanced_name"
                                        ] = enhanced_name

                                        st.success(
                                            "Your image has been enhanced "
                                            "successfully!"
                                        )

                                        st.rerun()

            except Exception as error:

                st.error(
                    f"An unexpected error occurred: {error}"
                )

    # ==================================================
    # 9. DOWNLOAD ENHANCED IMAGE
    # ==================================================

    if "enhanced_image" in st.session_state:

        st.download_button(

            label="⬇ Download Enhanced Image",

            data=st.session_state["enhanced_image"],

            file_name=st.session_state["enhanced_name"],

            mime=(
                "image/png"
                if st.session_state["enhanced_name"].lower().endswith(".png")
                else "image/jpeg"
            ),

            type="primary",

            use_container_width=True
        )
