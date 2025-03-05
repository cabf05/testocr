import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import tempfile
import os
import re
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caminho para os dados do Tesseract (ajuste conforme necessário)
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"

# Configuração do Tesseract para melhor precisão
TESSERACT_CONFIG = r'--oem 3 --psm 6 -c preserve_interword_spaces=1 -l por+eng'

def preprocess_image(image):
    """Melhora a qualidade da imagem para OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    return thresh

def extract_text_from_pdf(pdf_path):
    """Extrai texto de todas as páginas de um PDF."""
    images = convert_from_path(pdf_path, dpi=300, poppler_path="/usr/bin")
    text_parts = []
    for image in images:
        processed_image = preprocess_image(np.array(image))
        text = pytesseract.image_to_string(processed_image, config=TESSERACT_CONFIG)
        text_parts.append(text)
    return "\n".join(text_parts)

def correct_text_format(text):
    """Corrige formatos comuns de texto em NFS-e."""
    corrections = {
        r'(\d{2})[\.]?(\d{3})[\.]?(\d{3})[/]?0001[-]?(\d{2})': r'\1.\2.\3/0001-\4',  # CNPJ
        r'(\d{2})[\/.-](\d{2})[\/.-](\d{4})': r'\1/\2/\3',  # Datas
        r'R\$ (\d+)[,.](\d{2})': r'R$\1,\2'  # Valores
    }
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)
    return text

def validate_extracted_text(text):
    """Valida se o texto extraído contém informações chave."""
    required_patterns = [
        r'NOTA FISCAL DE SERVIÇOS ELETRÔNICA',
        r'CNPJ',
        r'Valor Total',
        r'Data e Hora de Emissão'
    ]
    for pattern in required_patterns:
        if not re.search(pattern, text, re.IGNORECASE):
            return False
    return True

def main():
    st.title("Extração de Texto de NFS-e")
    uploaded_file = st.file_uploader("Carregue seu arquivo PDF", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            pdf_path = temp_file.name
        extracted_text = extract_text_from_pdf(pdf_path)
        corrected_text = correct_text_format(extracted_text)
        if validate_extracted_text(corrected_text):
            st.success("Texto extraído e validado com sucesso!")
            st.text_area("Texto extraído", corrected_text, height=300)
        else:
            st.error("Não foi possível validar o texto extraído. Verifique o arquivo.")
        os.unlink(pdf_path)

if __name__ == "__main__":
    main()
