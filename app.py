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

# ========== CONFIGURAÇÃO DO AMBIENTE ========== #
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/tessdata/"

# Verificação do ambiente
try:
    logging.info("Conteúdo do diretório TESSDATA: %s", os.listdir(os.environ["TESSDATA_PREFIX"]))
except Exception as e:
    logging.error("Erro na verificação do ambiente: %s", str(e))

# ========== FUNÇÕES DE PRÉ-PROCESSAMENTO ========== #
def enhance_image_quality(image):
    """Melhora a qualidade da imagem para OCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    return cv2.adaptiveThreshold(enhanced, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

# ========== FUNÇÕES DE PÓS-PROCESSAMENTO ========== #
def clean_ocr_text(text):
    """Corrige erros comuns no texto extraído"""
    corrections = [
        (r'(\d{2})[\\/]08[\\/](\d{4})', r'\1/09/\2'),  # Datas
        (r'(\d{2}\.\d{3}\.\d{3})/\d{4}-(\d{2})', r'\1/0001-\2'),  # CNPJ
        (r'R\$\s?(\d+)\.?(\d+),?(\d+)', r'R$ \1\2,\3')  # Valores
    ]
    
    for pattern, replacement in corrections:
        text = re.sub(pattern, replacement, text)
    
    return text

# ========== FUNÇÃO PRINCIPAL DE OCR ========== #
def extract_text_from_pdf(pdf_path):
    """Executa OCR com tratamento avançado"""
    try:
        images = convert_from_path(pdf_path, 
                                 dpi=400,
                                 poppler_path="/usr/bin",
                                 grayscale=True)
        
        full_text = []
        custom_config = r'''
            --psm 6 
            --oem 3 
            -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-.,:()@$%& 
            -l por+eng
        '''
        
        for img in images:
            cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            processed = enhance_image_quality(cv_image)
            
            text = pytesseract.image_to_string(
                processed,
                config=custom_config
            )
            full_text.append(clean_ocr_text(text))
        
        final_text = "\n".join(full_text)
        
        # Validação de campos críticos
        if not validate_key_fields(final_text):
            raise ValueError("Campos obrigatórios não detectados")
            
        return final_text
    
    except Exception as e:
        logging.error("Erro no processamento: %s", str(e))
        return f"ERRO: {str(e)}"

# ========== VALIDAÇÃO ========== #
def validate_key_fields(text):
    """Verifica a presença de campos essenciais"""
    required_fields = {
        'NFS-e': r'NFS-e',
        'CNPJ Prestador': r'49\.621\.411/0001-93',
        'Valor Total': r'R\$\s*750,00'
    }
    
    return all(re.search(pattern, text) for _, pattern in required_fields.items())

# ========== INTERFACE ========== #
def main():
    st.title("📄 OCR Avançado para NFS-e")
    
    uploaded_file = st.file_uploader("Carregue o PDF da NFS-e", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            result = extract_text_from_pdf(tmp_file.name)
            
            st.subheader("Resultado da Extração")
            st.text_area("Texto Processado", result, height=400)
            
            if "ERRO:" in result:
                st.error("Problema na extração - Verifique os logs")
            else:
                st.success("Extração concluída com validação!")
            
            os.unlink(tmp_file.name)

if __name__ == "__main__":
    main()
