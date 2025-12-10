"""
FactCheck-MM Demo Application
A Streamlit-based UI showcasing multimodal fact-checking pipeline

Run with: streamlit run app.py
"""

import streamlit as st
import sys
import os
import json
from pathlib import Path
from PIL import Image
import io

# Add parent directory to path for real module imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import mock pipeline (always available)
from mock_pipeline import (
    detect_sarcasm,
    paraphrase_text,
    verify_claim,
    process_multimodal_input
)

# Attempt to import real modules (graceful fallback)
USE_REAL_MODELS = False
try:
    # These imports will fail if models aren't trained/available
    # Replace with actual import paths from your repo
    from sarcasm_detection.predict import detect_sarcasm as real_detect_sarcasm
    from paraphrasing.generate import paraphrase as real_paraphrase
    from fact_verification.verify import verify_claim as real_verify_claim
    USE_REAL_MODELS = True
    st.sidebar.success("✅ Real models loaded successfully!")
except (ImportError, ModuleNotFoundError) as e:
    st.sidebar.info("ℹ️ Using mock models (real models not available)")

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="FactCheck-MM Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .step-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .verdict-supports {
        background-color: #d4edda;
        padding: 15px;
        border-left: 5px solid #28a745;
        border-radius: 5px;
    }
    .verdict-refutes {
        background-color: #f8d7da;
        padding: 15px;
        border-left: 5px solid #dc3545;
        border-radius: 5px;
    }
    .verdict-nei {
        background-color: #fff3cd;
        padding: 15px;
        border-left: 5px solid #ffc107;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown('<p class="main-header">🔍 FactCheck-MM</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multimodal Fact-Checking with Sarcasm Detection</p>', unsafe_allow_html=True)

# ========== SIDEBAR ==========
st.sidebar.title("⚙️ Configuration")

# Model selection
use_real = st.sidebar.checkbox(
    "Use repo models (if available)",
    value=False,
    help="Attempts to use trained models from main repo. Falls back to mocks if unavailable."
)

st.sidebar.markdown("---")

# Input type selection
input_type = st.sidebar.radio(
    "📥 Input Type",
    options=["Text", "Text + Image", "Image Only", "Audio", "Video", "Load Demo Example"],
    index=0
)

st.sidebar.markdown("---")

# Load demo examples
demo_inputs_path = Path(__file__).parent / "demo_inputs.json"
demo_inputs = []
if demo_inputs_path.exists():
    with open(demo_inputs_path, 'r') as f:
        demo_inputs = json.load(f)
    
    if input_type == "Load Demo Example":
        example_options = [f"Example {d['id']}: {d['type']}" for d in demo_inputs]
        selected_example = st.sidebar.selectbox("Select Example", example_options)
        example_idx = int(selected_example.split()[1].replace(":", "")) - 1
        selected_demo = demo_inputs[example_idx]
        st.sidebar.json(selected_demo)

# ========== MAIN CONTENT ==========

# Create three columns
col1, col2, col3 = st.columns([2, 3, 2])

# ========== LEFT COLUMN: INPUT ==========
with col1:
    st.header("📝 Input")
    
    text_input = None
    image_file = None
    audio_file = None
    video_file = None
    
    if input_type == "Text":
        text_input = st.text_area(
            "Enter claim to verify:",
            height=150,
            placeholder="Type your claim here... e.g., 'The government announced free electricity for all citizens last week.'"
        )
        
    elif input_type == "Text + Image":
        text_input = st.text_area(
            "Caption or claim:",
            height=100,
            placeholder="Describe the image or provide context..."
        )
        image_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
        if image_file:
            st.image(image_file, caption="Uploaded Image", use_column_width=True)
            
    elif input_type == "Image Only":
        image_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
        if image_file:
            st.image(image_file, caption="Uploaded Image", use_column_width=True)
        st.info("💡 Image-only mode will extract text via OCR (mocked)")
        
    elif input_type == "Audio":
        audio_file = st.file_uploader("Upload audio", type=['wav', 'mp3', 'ogg'])
        if audio_file:
            st.audio(audio_file)
        st.info("💡 Audio will be transcribed to text (mocked)")
        
    elif input_type == "Video":
        video_file = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'])
        if video_file:
            st.video(video_file)
        text_input = st.text_area("Video description (optional):", height=80)
        st.info("💡 Video frames + audio will be analyzed (mocked)")
        
    elif input_type == "Load Demo Example":
        if 'selected_demo' in locals():
            text_input = selected_demo.get('text', '')
            st.text_area("Claim:", value=text_input, height=100, disabled=True)
            
            if selected_demo.get('file'):
                asset_path = Path(__file__).parent / "assets" / selected_demo['file']
                if asset_path.exists():
                    if selected_demo['type'] in ['text_image', 'image']:
                        st.image(str(asset_path), caption="Demo Image")
                    elif selected_demo['type'] == 'audio':
                        st.audio(str(asset_path))
                    elif selected_demo['type'] == 'video':
                        st.video(str(asset_path))
    
    st.markdown("---")
    run_demo = st.button("🚀 Run FactCheck Pipeline", type="primary", use_container_width=True)

# ========== CENTER COLUMN: PIPELINE STEPS ==========
with col2:
    st.header("🔄 Pipeline Execution")
    
    if run_demo:
        if not text_input and not image_file and not audio_file and not video_file:
            st.error("❌ Please provide at least one input (text, image, audio, or video)")
        else:
            # Create progress steps
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # ===== STEP 1: INPUT PROCESSING =====
            status_text.text("Step 1/4: Processing input...")
            progress_bar.progress(25)
            
            with st.expander("📊 **Step 1: Input Processing**", expanded=True):
                if image_file:
                    st.write(f"🖼️ **Image detected**: {image_file.name if hasattr(image_file, 'name') else 'uploaded_image.jpg'}")
                if audio_file:
                    st.write(f"🎵 **Audio detected**: {audio_file.name if hasattr(audio_file, 'name') else 'uploaded_audio.wav'}")
                if video_file:
                    st.write(f"🎬 **Video detected**: {video_file.name if hasattr(video_file, 'name') else 'uploaded_video.mp4'}")
                if text_input:
                    st.write(f"📝 **Text input**: {text_input[:100]}{'...' if len(text_input) > 100 else ''}")
                    
                st.success("✅ Input processed successfully")
            
            # ===== STEP 2: SARCASM DETECTION =====
            status_text.text("Step 2/4: Detecting sarcasm...")
            progress_bar.progress(50)
            
            # Call appropriate function
            if use_real and USE_REAL_MODELS:
                try:
                    sarcasm_result = real_detect_sarcasm(
                        text=text_input,
                        image=image_file,
                        audio=audio_file,
                        video=video_file
                    )
                except Exception as e:
                    st.warning(f"Real model failed: {e}. Using mock.")
                    sarcasm_result = detect_sarcasm(text_input, image_file, audio_file, video_file)
            else:
                sarcasm_result = detect_sarcasm(text_input, image_file, audio_file, video_file)
            
            with st.expander("😏 **Step 2: Sarcasm Detection**", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Label", sarcasm_result['label'].upper())
                with col_b:
                    st.metric("Confidence", f"{sarcasm_result['score']:.2%}")
                
                if sarcasm_result['label'] == 'sarcastic':
                    st.warning("⚠️ Sarcastic content detected - paraphrasing required")
                else:
                    st.success("✅ No sarcasm detected")
            
            # ===== STEP 3: PARAPHRASING =====
            status_text.text("Step 3/4: Generating literal paraphrase...")
            progress_bar.progress(75)
            
            if sarcasm_result['label'] == 'sarcastic' and text_input:
                if use_real and USE_REAL_MODELS:
                    try:
                        paraphrase_result = real_paraphrase(text_input)
                    except Exception as e:
                        st.warning(f"Real model failed: {e}. Using mock.")
                        paraphrase_result = paraphrase_text(text_input)
                else:
                    paraphrase_result = paraphrase_text(text_input)
            else:
                paraphrase_result = text_input if text_input else "[Generated claim from image/audio/video]"
            
            with st.expander("✏️ **Step 3: Paraphrasing**", expanded=True):
                st.write("**Original:**")
                st.info(text_input if text_input else "[Extracted from media]")
                st.write("**Literal Paraphrase:**")
                st.success(paraphrase_result)
            
            # ===== STEP 4: FACT VERIFICATION =====
            status_text.text("Step 4/4: Verifying claim...")
            progress_bar.progress(100)
            
            claim_to_verify = paraphrase_result if paraphrase_result else text_input
            
            if use_real and USE_REAL_MODELS:
                try:
                    verdict_result = real_verify_claim(claim_to_verify)
                except Exception as e:
                    st.warning(f"Real model failed: {e}. Using mock.")
                    verdict_result = verify_claim(claim_to_verify)
            else:
                verdict_result = verify_claim(claim_to_verify)
            
            with st.expander("⚖️ **Step 4: Fact Verification**", expanded=True):
                # Evidence retrieval
                st.write("**📚 Retrieved Evidence:**")
                for i, ev in enumerate(verdict_result['evidence'], 1):
                    with st.container():
                        st.markdown(f"**Source {i}:** {ev['source']}")
                        st.markdown(f"> {ev['snippet']}")
                        st.markdown(f"*Relevance: {ev.get('relevance', 0.85):.2%}*")
                        st.markdown("---")
                
                # Final verdict
                verdict_class = f"verdict-{verdict_result['verdict'].lower().replace('_', '-')}"
                
                st.markdown(f'<div class="{verdict_class}">', unsafe_allow_html=True)
                st.markdown(f"### 🎯 Verdict: **{verdict_result['verdict']}**")
                st.markdown(f"**Confidence:** {verdict_result['confidence']:.2%}")
                st.markdown(f"**Explanation:** {verdict_result.get('explanation', 'N/A')}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            status_text.text("✅ Pipeline completed!")
            st.balloons()

# ========== RIGHT COLUMN: RAW OUTPUT ==========
with col3:
    st.header("📋 Raw Output")
    
    if run_demo and (text_input or image_file or audio_file or video_file):
        # Compile full result
        full_result = {
            "input": {
                "type": input_type,
                "text": text_input,
                "has_image": image_file is not None,
                "has_audio": audio_file is not None,
                "has_video": video_file is not None
            },
            "sarcasm_detection": sarcasm_result,
            "paraphrase": paraphrase_result,
            "fact_verification": verdict_result,
            "model_type": "real" if (use_real and USE_REAL_MODELS) else "mock"
        }
        
        st.json(full_result)
        
        # Download button
        result_json = json.dumps(full_result, indent=2)
        st.download_button(
            label="💾 Download Results (JSON)",
            data=result_json,
            file_name="factcheck_results.json",
            mime="application/json"
        )
    else:
        st.info("Results will appear here after running the pipeline")

# ========== FOOTER ==========
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown("**📖 [Documentation](../docs/)**")
with col_f2:
    st.markdown("**🐙 [GitHub Repo](https://github.com/RiyaGupta2230/FactCheck-MM)**")
with col_f3:
    st.markdown("**💡 [Integration Guide](static/notes_for_integrators.md)**")
