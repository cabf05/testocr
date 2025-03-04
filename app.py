import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import tempfile
import os

# Define a função para pré-processar a imagem
def preprocess_image(image):
    # Converte para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplica binarização adaptativa
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 31, 2)
    # Retorna a imagem processada
    return processed

# Define a função para extrair texto do PDF usando OCR
def extract_text_from_pdf(pdf_path):
    try:
        # Converte o PDF em imagens
        images = convert_from_path(pdf_path, dpi=300)
        extracted_text = ""
        
        for img in images:
            # Converte imagem do PIL para OpenCV
            open_cv_image = np.array(img)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            
            # Pré-processa a imagem
            processed_image = preprocess_image(open_cv_image)
            
            # Aplica OCR com otimização de segmentação
            custom_config = r'--oem 3 --psm 6'  # Melhor configuração para texto estruturado
            text = pytesseract.image_to_string(processed_image, config=custom_config, lang='por')
            
            extracted_text += text + "\n"
        
        if extracted_text.strip():
            return extracted_text
        else:
            return "OCR não conseguiu extrair texto."
    
    except Exception as e:
        return f"Erro no OCR: {str(e)}"

# Interface do Streamlit
st.title("Extração de Texto de PDFs (OCR Aprimorado)")

uploaded_file = st.file_uploader("Envie um arquivo PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    st.text("Processando o arquivo...")
    extracted_text = extract_text_from_pdf(temp_pdf_path)
    
    # Exibe o texto extraído
    st.text_area("Texto Extraído", extracted_text, height=300)

    # Remove o arquivo temporário
    os.remove(temp_pdf_path)
