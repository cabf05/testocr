import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import tempfile
import os

# Configurações do Tesseract (automáticas no Streamlit Cloud)
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/"

def preprocess_image(image):
    """Pré-processamento da imagem para melhorar o OCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )
    return processed

def extract_text_from_pdf(pdf_path):
    """Extrai texto de um PDF usando OCR com tratamento de erros"""
    try:
        images = convert_from_path(pdf_path, dpi=300)
        extracted_text = []
        
        for img in images:
            # Conversão para formato OpenCV
            open_cv_image = np.array(img)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            
            # Pré-processamento
            processed_image = preprocess_image(open_cv_image)
            
            # Configuração do Tesseract
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

# Interface Streamlit
def main():
    st.title("📄 Extrator de Texto de PDF (OCR)")
    st.markdown("**Envie um PDF para extrair o texto usando reconhecimento óptico de caracteres**")
    
    uploaded_file = st.file_uploader("Selecione um arquivo PDF", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            temp_path = temp_pdf.name
        
        with st.spinner("Processando PDF..."):
            result = extract_text_from_pdf(temp_path)
        
        st.subheader("Texto Extraído")
        st.text_area("Resultado", result, height=400)
        
        os.remove(temp_path)

if __name__ == "__main__":
    main()
