import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import tempfile
import os

# Configuração crucial para o Tesseract no Streamlit Cloud
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/tessdata/"

def preprocess_image(image):
    """Melhora a qualidade da imagem para OCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )

def extract_text_from_pdf(pdf_path):
    """Processa o PDF e extrai texto com OCR"""
    try:
        images = convert_from_path(pdf_path, dpi=300)
        full_text = []
        
        for img in images:
            # Converte para formato OpenCV
            cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            processed = preprocess_image(cv_image)
            
            # Configuração otimizada para documentos
            text = pytesseract.image_to_string(
                processed,
                lang='por',
                config='--oem 3 --psm 6'
            )
            full_text.append(text.strip())
        
        return "\n\n".join(full_text) if any(full_text) else "Nenhum texto detectado."
    
    except Exception as e:
        return f"Erro: {str(e)}"

def main():
    st.title("📑 Conversor PDF para Texto")
    st.markdown("### Utilizando Tesseract OCR com suporte a português")
    
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo PDF", 
        type="pdf",
        help="Tamanho máximo: 200MB"
    )
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name
        
        with st.spinner("Processando documento..."):
            result = extract_text_from_pdf(pdf_path)
        
        st.subheader("Resultado da Extração")
        st.text_area("Texto Extraído", value=result, height=400)
        
        os.remove(pdf_path)

if __name__ == "__main__":
    main()
