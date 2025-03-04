import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import tempfile
import os

# ========== CONFIGURAÇÃO CRÍTICA ========== #
# Caminho oficial do Tesseract no Streamlit Cloud
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"

# Verificação explícita do caminho (para debug)
try:
    print("Conteúdo do diretório TESSDATA:", os.listdir(os.environ["TESSDATA_PREFIX"]))
except Exception as e:
    print(f"Erro na verificação do caminho: {str(e)}")

# ========== FUNÇÕES DE PROCESSAMENTO ========== #
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 2)

def extract_text_from_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=300, poppler_path="/usr/bin")
        full_text = []
        
        for img in images:
            cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            processed = preprocess_image(cv_image)
            
            # Configuração reforçada com caminho absoluto
            text = pytesseract.image_to_string(
                processed,
                lang='por',
                config=f'--tessdata-dir {os.environ["TESSDATA_PREFIX"]} --oem 3 --psm 6'
            )
            full_text.append(text.strip())
        
        return "\n\n".join(full_text) if any(full_text) else "Nenhum texto detectado."
    
    except Exception as e:
        return f"Erro crítico: {str(e)}"

# ========== INTERFACE ========== #
def main():
    st.title("📄 Conversor PDF-Texto (Solução Definitiva)")
    
    uploaded_file = st.file_uploader("Carregue seu PDF", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            result = extract_text_from_pdf(tmp_file.name)
            st.text_area("Texto Extraído", result, height=400)
            os.unlink(tmp_file.name)

if __name__ == "__main__":
    main()
