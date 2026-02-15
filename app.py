import streamlit as st
import joblib
import streamlit.components.v1 as components
import numpy as np

# Set page config first
st.set_page_config(page_title="AI Sentiment 3D", layout="wide", initial_sidebar_state="collapsed")

# Load model + vectorizer (cached to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    return joblib.load("lgbm_model.pkl"), joblib.load("tfidf_vectorizer.pkl")

try:
    model, vectorizer = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.image("https://http.cat/500", width=400) # Fun fallback
    st.stop()


# Custom CSS for Glassmorphism & UI Overhaul
st.markdown("""
<style>
    /* Reset & Base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
        background: transparent;
    }

    /* Input & Interactive Elements - Glassmorphism */
    .stTextArea > div > div {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div:focus-within {
        border-color: rgba(64, 224, 208, 0.5);
        box-shadow: 0 0 15px rgba(64, 224, 208, 0.2);
        background: rgba(255, 255, 255, 0.1);
    }

    .stTextArea label {
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 14px;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        padding: 12px 30px;
        font-weight: 600;
        border-radius: 30px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(118, 75, 162, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    /* Result Containers */
    .success-box {
        padding: 20px;
        border-radius: 12px;
        background: rgba(46, 204, 113, 0.15);
        border: 1px solid rgba(46, 204, 113, 0.3);
        backdrop-filter: blur(5px);
        color: #2ecc71;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 20px;
        animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .warning-box {
         padding: 15px;
         border-radius: 10px;
         background: rgba(241, 196, 15, 0.15);
         border: 1px solid rgba(241, 196, 15, 0.3);
         color: #f1c40f;
         text-align: center;
         margin-top: 20px;
    }

    @keyframes popIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }

</style>
""", unsafe_allow_html=True)

# 3D Background & GSAP Animation Component
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>3D Background</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
    <style>
        body { margin: 0; overflow: hidden; background-color: #0f2027; }
        #canvas-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1; 
        }
        
        .hero-text {
            position: absolute;
            top: 15%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            font-family: 'Inter', sans-serif;
            color: white;
            z-index: 1;
            pointer-events: none;
            width: 100%;
        }
        
        h1 {
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            letter-spacing: -2px;
            filter: drop-shadow(0 0 20px rgba(79, 172, 254, 0.5));
        }
        
        p {
            font-size: 1.2rem;
            color: rgba(255,255,255,0.7);
            margin-top: 10px;
            font-weight: 300;
        }

    </style>
</head>
<body>
    <div id="canvas-container"></div>
    <div class="hero-text" id="hero">
        <h1>Sentiment AI 3.0</h1>
        <p>Experience the next dimension of analysis</p>
    </div>

    <script>
        // Three.js Setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('canvas-container').appendChild(renderer.domElement);

        // Particles
        const geometry = new THREE.BufferGeometry();
        const particlesCount = 3000;
        const posArray = new Float32Array(particlesCount * 3);

        for(let i = 0; i < particlesCount * 3; i++) {
            posArray[i] = (Math.random() - 0.5) * 15; // Spread
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));

        const material = new THREE.PointsMaterial({
            size: 0.02,
            color: 0x40E0D0,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending
        });

        const particlesMesh = new THREE.Points(geometry, material);
        scene.add(particlesMesh);
        
        // Geometric Floating Shape (Icosahedron)
        const geometry2 = new THREE.IcosahedronGeometry(1, 1);
        const material2 = new THREE.MeshBasicMaterial({ color: 0x764ba2, wireframe: true, transparent: true, opacity: 0.2 });
        const sphere = new THREE.Mesh(geometry2, material2);
        sphere.position.set(3, 1, 0);
        scene.add(sphere);

        // Second Shape (Torus)
        const geometry3 = new THREE.TorusGeometry(0.8, 0.2, 16, 100);
        const material3 = new THREE.MeshBasicMaterial({ color: 0x4facfe, wireframe: true, transparent: true, opacity: 0.2 });
        const torus = new THREE.Mesh(geometry3, material3);
        torus.position.set(-3, -1, 0);
        scene.add(torus);

        camera.position.z = 5;

        // Mouse Interaction
        let mouseX = 0;
        let mouseY = 0;

        document.addEventListener('mousemove', (event) => {
            mouseX = event.clientX;
            mouseY = event.clientY;
            
            // GSAP Parallax for smooth camera/object movement
            gsap.to(particlesMesh.rotation, {
                x: mouseY * 0.00005,
                y: mouseX * 0.00005,
                duration: 2
            });
             gsap.to(sphere.rotation, {
                x: mouseY * 0.0002,
                y: mouseX * 0.0002,
                duration: 2
            });
             gsap.to(torus.rotation, {
                x: mouseX * 0.0002,
                y: mouseY * 0.0002,
                duration: 2
            });
        });

        // Animation Loop
        const clock = new THREE.Clock();

        function animate() {
            const elapsedTime = clock.getElapsedTime();

            particlesMesh.rotation.y = elapsedTime * 0.05;
            
            sphere.rotation.x = elapsedTime * 0.2;
            sphere.rotation.y = elapsedTime * 0.1;
            sphere.position.y = 1 + Math.sin(elapsedTime * 0.5) * 0.2; // Float
            
            torus.rotation.x = elapsedTime * 0.1;
            torus.rotation.y = elapsedTime * 0.2;
            torus.position.y = -1 + Math.sin(elapsedTime * 0.6) * 0.2; // Float

            renderer.render(scene, camera);
            requestAnimationFrame(animate);
        }

        animate();
        
        // Handle Resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        // Intro Animation
        gsap.from("h1", { duration: 1.5, y: 50, opacity: 0, ease: "power4.out", delay: 0.2 });
        gsap.from("p", { duration: 1.5, y: 30, opacity: 0, ease: "power4.out", delay: 0.5 });

    </script>
</body>
</html>
"""

# Render the 3D Background
components.html(html_code, height=450, scrolling=False)

# Main UI Layout
st.write("") 
st.write("") 

col1, col2, col3 = st.columns([1, 6, 1]) # Centered column for content

with col2:
    # Text Area with default label (styled via CSS)
    user_input = st.text_area("Analyze Text", height=120, placeholder="Type something here... e.g., 'The interface is absolutely mind-blowing!'", label_visibility="collapsed")

    if st.button("Analyze Sentiment"):
        if user_input.strip():  # Check input is not empty
            try:
                # Transform input text
                text_vector = vectorizer.transform([user_input])
                prediction = model.predict(text_vector)[0]

                # Normalize for case-insensitive matching
                prediction = str(prediction).lower()

                # Map to emoji & color
                if prediction == "negative":
                    label = "Negative 😡"
                    color = "#e74c3c"
                elif prediction == "neutral":
                    label = "Neutral 😐"
                    color = "#f1c40f"
                else:
                    label = "Positive 😄"
                    color = "#2ecc71"

                # Display result
                st.markdown(f"""
                <div class="success-box" style="border-color: {color}; color: {color}; background: {color}1a; box-shadow: 0 0 20px {color}40;">
                    Prediction: <strong>{label}</strong>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

            finally:
                # This always runs
                print("Prediction attempt finished.")

        else:
            st.markdown('<div class="warning-box">Please enter some text to analyze.</div>', unsafe_allow_html=True)
