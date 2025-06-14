import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# App configuration
st.set_page_config(
    page_title="Leather Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Leather Anomaly Detection System")
st.markdown("Upload an image of leather to detect if it's normal or defective")


# Load the model
@st.cache_resource
def load_model():
    """Load the trained model from Teachable Machine"""
    try:
        # Replace with your model path
        model = tf.keras.models.load_model('keras_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Load class labels
@st.cache_data
def load_labels():
    """Load class labels"""
    try:
        with open('labels.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    except Exception as e:
        st.error(f"Error loading labels: {str(e)}")
        return ["0 Normal", "1 Defective"]  # Default labels matching your format


def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize image to 224x224 (Teachable Machine default)
    image = image.resize((224, 224))

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy array and normalize
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_anomaly(model, image, labels):
    """Make prediction on the image"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)

        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        return labels[predicted_class_idx], confidence, predictions[0], predicted_class_idx
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None, None


def get_class_name(label):
    """Extract class name from label format '0 Normal' or '1 Defective'"""
    return label.split(' ', 1)[1] if ' ' in label else label


# Load model and labels
model = load_model()
labels = load_labels()

if model is not None:
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a leather image for anomaly detection"
    )

    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])

        with col1:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            # Make prediction
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, all_predictions, class_idx = predict_anomaly(
                    model, image, labels
                )

            if predicted_class is not None:
                # Display results
                st.subheader("🎯 Detection Results")

                # Extract class name for display
                class_name = get_class_name(predicted_class)

                # Main prediction with appropriate styling
                if class_idx == 0:  # Normal
                    st.success(f"✅ **{class_name}** leather detected")
                    st.success(f"Confidence: {confidence:.2%}")
                    st.balloons()  # Celebration for normal leather
                else:  # Defective
                    st.error(f"⚠️ **{class_name}** leather detected")
                    st.error(f"Confidence: {confidence:.2%}")

                # Confidence scores for all classes
                st.subheader("📊 Confidence Scores")
                for i, (label, score) in enumerate(zip(labels, all_predictions)):
                    class_display = get_class_name(label)

                    # Color code the progress bars
                    if i == 0:  # Normal
                        st.write(f"**✅ {class_display}**: {score:.2%}")
                    else:  # Defective
                        st.write(f"**⚠️ {class_display}**: {score:.2%}")

                    st.progress(float(score))

                # Quality assessment
                st.subheader("📋 Quality Assessment")
                if class_idx == 0:
                    if confidence > 0.8:
                        st.success(
                            "🟢 **High Confidence**: This leather appears to be of good quality with no visible defects.")
                    elif confidence > 0.6:
                        st.warning(
                            "🟡 **Medium Confidence**: The leather appears normal, but consider additional inspection.")
                    else:
                        st.info("🔵 **Low Confidence**: Results are uncertain. Manual inspection recommended.")
                else:
                    if confidence > 0.8:
                        st.error(
                            "🔴 **High Confidence**: Defects detected. This leather should be rejected or marked for repair.")
                    elif confidence > 0.6:
                        st.warning("🟡 **Medium Confidence**: Possible defects detected. Further inspection needed.")
                    else:
                        st.info("🔵 **Low Confidence**: Uncertain results. Manual quality check recommended.")

        # Additional information
        st.markdown("---")
        st.subheader("ℹ️ About This Model")
        st.info("""
        This leather quality inspection model was trained using Google's Teachable Machine on the MVTec AD leather dataset.

        **Classification Categories:**
        - **Normal (Class 0)**: High-quality leather without visible defects
        - **Defective (Class 1)**: Leather with defects such as scratches, cuts, holes, or other quality issues

        The model analyzes visual patterns and textures to automatically classify leather quality for industrial inspection purposes.
        """)

        # Technical details in an expander
        with st.expander("🔧 Technical Details"):
            st.markdown(f"""
            **Model Architecture:** MobileNet-based CNN (Teachable Machine)
            **Input Size:** 224 × 224 pixels
            **Classes:** {len(labels)}
            **Prediction Classes:**
            """)
            for i, label in enumerate(labels):
                st.write(f"- **{label}**")

    else:
        # Instructions when no image is uploaded
        st.info("👆 Please upload a leather image to start the quality inspection")

        # Sample usage guide
        st.markdown("---")
        st.subheader("📖 How to Use This System")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Step 1: Upload Image**
            - Click the file uploader above
            - Select a leather image (PNG, JPG, JPEG)
            - Wait for upload to complete

            **Step 2: View Results**
            - See classification: Normal or Defective
            - Check confidence percentage
            - Review quality assessment
            """)

        with col2:
            st.markdown("""
            **Best Practices:**
            - Use well-lit, clear images
            - Ensure leather surface is visible
            - Avoid blurry or dark images
            - Images should show texture details

            **Applications:**
            - Quality control in manufacturing
            - Leather goods inspection
            - Automated sorting systems
            """)

else:
    st.error("❌ Model could not be loaded. Please check if the model files are in the correct location.")
    st.markdown("""
    ### Required files:
    - **keras_model.h5** - The trained model from Teachable Machine
    - **labels.txt** - Class labels file with format:
    ```
    0 Normal
    1 Defective
    ```

    Make sure these files are in the same directory as your Streamlit app.
    """)

# Sidebar with additional information
with st.sidebar:
    st.header("🛠️ System Information")
    st.markdown("""
    **Technology Stack:**
    - 🧠 TensorFlow/Keras
    - 🤖 Google Teachable Machine
    - 🚀 Streamlit
    - 🖼️ PIL/Pillow

    **Model Specifications:**
    - Input Size: 224×224 pixels
    - Classes: Normal, Defective
    - Architecture: MobileNet-based CNN
    """)

    st.markdown("---")
    st.header("📊 Dataset Info")
    st.markdown("""
    **MVTec AD Leather Dataset:**
    - Normal leather samples
    - Various defect types:
      - Cuts and scratches
      - Holes and tears  
      - Color variations
      - Surface irregularities
    """)

    st.markdown("---")
    st.header("🎯 Performance Tips")
    st.markdown("""
    **For Best Results:**
    - Good lighting conditions
    - Sharp, focused images
    - Clear view of leather surface
    - Consistent image quality
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Leather Quality Inspection System | Powered by Teachable Machine & Streamlit"
    "</div>",
    unsafe_allow_html=True
)