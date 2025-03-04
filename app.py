import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import tempfile
import os

# Configuração crucial para o Streamlit Cloud
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/tessdata/"

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )
    return processed

def extract_text_from_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=300)
        extracted_text = []
        
        for img in images:
            open_cv_image = np.array(img)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            processed_image = preprocess_image(open_cv_image)
            
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(
                processed_image,
                config=custom_config,
                lang='por'
            )
            extracted_text.append(text.strip())
        
        full_text = "\n\n".join(extracted_text)
        return full_text if full_text else "Nenhum texto detectado."
    
    except Exception as e:
        return f"Erro durante o processamento: {str(e)}"

def main():
    st.title("📄 Extrator de Texto de PDF (OCR)")
    uploaded_file = st.file_uploader("Selecione um PDF", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            temp_path = temp_pdf.name
        
        with st.spinner("Processando..."):
            result = extract_text_from_pdf(temp_path)
        
        st.subheader("Resultado")
        st.text_area("Texto Extraído", result, height=400)
        os.remove(temp_path)

if __name__ == "__main__":
    main()
