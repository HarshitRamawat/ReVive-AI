
import importlib.util
import streamlit as st
import subprocess
import sys
import tempfile
import hashlib

from pathlib import Path


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "third_party" / "Real-ESRGAN"
MODEL_SCRIPT = MODEL_DIR / "inference_realesrgan.py"

INPUT_DIR = BASE_DIR / "input" / "uploads"
OUTPUT_DIR = BASE_DIR / "output" / "enhanced"

SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]



def fix_basicsr_compatibility():

    # Locate the installed BasicSR package
    spec = importlib.util.find_spec("basicsr")

    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("BasicSR package was not found.")

    basicsr_dir = Path(
        next(iter(spec.submodule_search_locations))
    )

    degradations_file = (
        basicsr_dir / "data" / "degradations.py"
    )

    content = degradations_file.read_text(encoding="utf-8")

    old_import = (
        "from torchvision.transforms.functional_tensor "
        "import rgb_to_grayscale"
    )

    new_import = (
        "from torchvision.transforms.functional "
        "import rgb_to_grayscale"
    )

    if old_import in content:

        content = content.replace(old_import, new_import)

        degradations_file.write_text(
            content,
            encoding="utf-8"
        )

    elif new_import not in content:

        raise RuntimeError(
            "Could not apply BasicSR compatibility fix."
        )


# --------------------------------------------------
# 2. STREAMLIT PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="ReVive AI",
    page_icon="✨",
    layout="wide"
)


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.title("✨ ReVive AI")
st.subheader("AI-Powered Image Enhancement")

st.write(
    "Upload your image and let our AI improve "
    "its resolution and details."
)

st.divider()


# --------------------------------------------------
# 4. IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=SUPPORTED_FORMATS
)


if uploaded_file is not None:

    original_name = Path(uploaded_file.name).name
    image_bytes = uploaded_file.getvalue()

    # Unique identifier for the uploaded image
    image_id = hashlib.md5(image_bytes).hexdigest()

    st.success(f"Uploaded successfully: {original_name}")

    # Reset previous result when a different image is uploaded
    if st.session_state.get("image_id") != image_id:

        st.session_state["image_id"] = image_id
        st.session_state.pop("enhanced_image", None)
        st.session_state.pop("enhanced_name", None)

    # --------------------------------------------------
    # 5. ORIGINAL AND ENHANCED IMAGE LAYOUT
    # --------------------------------------------------

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


    # --------------------------------------------------
    # 6. ENHANCE IMAGE BUTTON
    # --------------------------------------------------

    if st.button(
        "✨ Enhance Image",
        type="primary",
        use_container_width=True
    ):

        # Create required directories
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with st.spinner(
            "ReVive AI is enhancing your image. Please wait..."
        ):

            try:

                # Temporary input and output folders
                with tempfile.TemporaryDirectory(
                    dir=INPUT_DIR
                ) as input_temp:

                    with tempfile.TemporaryDirectory(
                        dir=OUTPUT_DIR
                    ) as output_temp:

                        input_path = Path(input_temp) / original_name
                        output_path = Path(output_temp)

                        # Save uploaded image temporarily
                        input_path.write_bytes(image_bytes)

                        # Real-ESRGAN command
                        command = [
                            sys.executable,
                            str(MODEL_SCRIPT),
                            "-n", "RealESRGAN_x4plus",
                            "-i", str(input_path),
                            "-o", str(output_path),
                            "--fp32",
                            "-t", "256"
                        ]

                        # Fix BasicSR and Torchvision compatibility
                        fix_basicsr_compatibility()

                        # Execute Real-ESRGAN
                        result = subprocess.run(
                            command,
                            cwd=str(MODEL_DIR),
                            capture_output=True,
                            text=True
                        )

                        # Check whether enhancement failed
                        if result.returncode != 0:

                            st.error(
                                "Image enhancement failed."
                            )

                            if result.stderr:
                                st.code(result.stderr)

                            if result.stdout:
                                st.code(result.stdout)

                        else:

                            # Find the generated image
                            generated_files = [
                                file
                                for file in output_path.iterdir()
                                if file.is_file()
                                and file.suffix.lower()
                                in [".jpg", ".jpeg", ".png"]
                            ]

                            if not generated_files:

                                st.error(
                                    "Enhanced image was not found."
                                )

                                if result.stdout:
                                    st.code(result.stdout)

                            else:

                                enhanced_file = generated_files[0]

                                # Read enhanced image before
                                # temporary folder is deleted
                                enhanced_bytes = enhanced_file.read_bytes()

                                # Save result in Streamlit session
                                st.session_state["enhanced_image"] = (
                                    enhanced_bytes
                                )

                                st.session_state["enhanced_name"] = (
                                    Path(original_name).stem
                                    + "_enhanced"
                                    + enhanced_file.suffix
                                )

                                st.success(
                                    "Your image has been enhanced successfully!"
                                )

                                # Refresh the app to show the result
                                st.rerun()

            except Exception as error:

                st.error(
                    f"An unexpected error occurred: {error}"
                )


    # --------------------------------------------------
    # 7. DOWNLOAD ENHANCED IMAGE
    # --------------------------------------------------

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

else:

    st.info(
        "Upload a JPG, JPEG, or PNG image to get started."
    )
