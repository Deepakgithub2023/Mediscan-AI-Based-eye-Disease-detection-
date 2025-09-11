import streamlit as st
import io
import pandas as pd
from PIL import Image, ImageEnhance
import base64
import requests  # Required for backend interaction

# API Endpoint
API_URL = "http://127.0.0.1:5000/predict"

# Function to Load Lottie Animation


def load_lottie(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.RequestException:
        pass
    return None

# Function to Encode Image as Base64


def get_base64_image(image_path, brightness=1.0):
    try:
        image = Image.open(image_path)
        enhancer = ImageEnhance.Brightness(image)
        bright_image = enhancer.enhance(brightness)
        buffer = io.BytesIO()
        bright_image.save(buffer, format="JPEG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return ""


# File paths for background images
navigation_bg_path = r"pexels-zandatsu-32213422.jpg"
home_bg_path = r"pexels-zandatsu-32213422.jpg"
upload_bg_path = r"pexels-cottonbro-8875642.jpg"
report_bg_path = r"pexels-pavel-danilyuk-5996604.jpg"



# Encode images as Base64
navigation_bg_base64 = get_base64_image(navigation_bg_path, brightness=0.7)
home_bg_base64 = get_base64_image(home_bg_path, brightness=0.5)
upload_bg_base64 = get_base64_image(upload_bg_path, brightness=0.5)
report_bg_base64 = get_base64_image(report_bg_path, brightness=0.5)

# CSS for Sidebar Background


def set_sidebar_bg(base64_image):
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background: url('data:image/jpeg;base64,{base64_image}') no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# CSS for Main Page Background and Styling


def set_page_bg(base64_image):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url('data:image/jpeg;base64,{base64_image}') no-repeat center center fixed;
            background-size: cover;
        }}

        /* White Buttons with Black Text */
        .stButton > button {{
            background-color: white !important;
            color: black !important;
            font-weight: bold;
            border: none !important;
            border-radius: 8px;
            transition: 0.3s ease-in-out;
            padding: 8px 20px;
        }}

        .stButton > button:hover {{
            background-color: #f0f0f0 !important;
            color: black !important;
            transform: scale(1.05);
        }}

        /* White Text for Titles, Labels, and Features */
        h1, h2, h3, label, .stMarkdown p, .features p {{
            color: white !important;
        }}

        /* White Text for List Items */
        ul li {{
            color: white !important;
            font-size: 18px;
            font-weight: bold;
        }}

        /* Image Grid Layout */
        .image-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }}
        .image-container img {{
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        }}

        /* Custom Info Box */
        .custom-info {{
            background-color: white;
            color: black;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
            font-weight: bold;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "detection_results" not in st.session_state:
    st.session_state.detection_results = []

if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

# Sidebar Navigation with background
set_sidebar_bg(navigation_bg_base64)
st.sidebar.title("🔹 Navigation")
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "Home"
if st.sidebar.button("📄 Details & Upload"):
    st.session_state.page = "Details & Upload"
if st.sidebar.button("📑 Report"):
    st.session_state.page = "Report"

# Page Content
if st.session_state.page == "Home":
    set_page_bg(home_bg_base64)

    st.markdown('<h1>MediScan: AI Medical Diagnosis.</h1>',
                unsafe_allow_html=True)
    st.markdown('<h2>About the Project</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <p>
        Medi Scan is a state-of-the-art medical imaging platform designed to streamline diagnostics.
        It allows healthcare professionals to manage, analyze, and generate detailed reports for retinal images.
        </p>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <h3>Key Features:</h3>
        <ul class="features">
            <li>✔ <b>Patient Management:</b> Add and organize patient details.</li>
            <li>✔ <b>Image Analysis:</b> Upload and process retinal scans for diagnostic purposes.</li>
            <li>✔ <b>Report Generation:</b> Create comprehensive and detailed medical reports.</li>
            <li>✔ <b>User-Friendly:</b> Developed for efficiency, precision, and ease of use.</li>
        </ul>
        """, unsafe_allow_html=True
    )

    if st.button("Start"):
        st.session_state.page = "Details & Upload"

elif st.session_state.page == "Details & Upload":
    set_page_bg(upload_bg_base64)

    st.markdown('<h1>📄 Retinal Detection</h1>',
                unsafe_allow_html=True)
    num_patients = st.number_input(
        "Number of patients:", min_value=1, max_value=10, value=1)

    patient_details = []
    uploaded_images = []

    for i in range(num_patients):
        st.markdown(f"<h3>Patient {i+1} Details</h3>", unsafe_allow_html=True)
        name = st.text_input(f"Name:", key=f"name_{i}")
        age = st.slider(f"Age:", min_value=1, max_value=120,
                        value=25, key=f"age_{i}")
        patient_id = st.text_input(f"Patient ID:", key=f"id_{i}")
        patient_details.append(
            {"Patient ID": patient_id, "Patient Name": name, "Age": age})

    st.markdown('<h2>Upload Retinal Images</h2>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload Images:", type=["jpg", "png"], accept_multiple_files=True)

    if st.button("🔍 Start Detection"):
        if not uploaded_files:
            st.error("⚠ Please upload images!")
        else:
            st.success("Images uploaded! Sending to backend...")
            results = []
            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                uploaded_images.append(image)

                # Prepare Data for Backend
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()

                # Encode for table and backend
                encoded_image = base64.b64encode(image_bytes).decode("utf-8")

                # Send image to API
                response = requests.post(
                    API_URL, files={"file": ("image.png", image_bytes, "image/png")})
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "Patient ID": patient_details[i]["Patient ID"],
                        "Patient Name": patient_details[i]["Patient Name"],
                        "Age": patient_details[i]["Age"],
                        "Predicted Disease": result["class"],
                        "Accuracy": result["accuracy"],
                        "encoded_image": encoded_image,  # Add encoded image for report
                    })

            st.session_state.detection_results = results
            st.success("✅ Detection Completed!")

            # Modified Info Box
            st.markdown(
                '<div class="custom-info">📑 Detection completed! Please visit the <strong>Report</strong> page in the navigation menu to view results.</div>',
                unsafe_allow_html=True
            )

elif st.session_state.page == "Report":
    set_page_bg(report_bg_base64)

    st.markdown('<h1>📑 Patient Report</h1>', unsafe_allow_html=True)

    if "detection_results" in st.session_state and st.session_state.detection_results:
        results_with_images = []
        for result in st.session_state.detection_results:
            result["Encoded Image"] = f"<img src='data:image/png;base64,{result['encoded_image']}' style='width: 100px; height: auto;'>"
            results_with_images.append(result)

        df = pd.DataFrame(results_with_images)

        # Custom CSS for the table
        st.markdown(
            """
            <style>
            table {
                background-color: white !important;
                color: black !important;
                border-collapse: collapse;
                width: 100%;
                border-radius: 8px;
                overflow: hidden;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Convert DataFrame to HTML and display
        html_table = df.to_html(escape=False, index=False, columns=[
                                "Patient ID", "Patient Name", "Age", "Predicted Disease", "Accuracy", "Encoded Image"])
        st.markdown(html_table, unsafe_allow_html=True)

    else:
        st.warning(
            "No detection results available. Please upload and analyze images first.")