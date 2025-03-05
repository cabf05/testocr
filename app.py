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
if "TESSDATA_PREFIX" not in os.environ:
    os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"
    logger.info(f"TESSDATA_PREFIX não configurado via variável de ambiente, usando padrão: {os.environ['TESSDATA_PREFIX']}")


def preprocess_image(image, binarization_threshold=31, denoise_strength=10):
    """Melhora a qualidade da imagem para OCR."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=denoise_strength, templateWindowSize=7, searchWindowSize=21)
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, binarization_threshold, 2)
        return thresh
    except Exception as e:
        logger.error(f"Erro no pré-processamento da imagem: {e}")
        st.error(f"Erro no pré-processamento da imagem: {e}. Verifique os parâmetros ou a imagem.")
        return None

def extract_text_from_pdf(pdf_path, dpi=300, psm=6, oem=3, binarization_threshold=31, denoise_strength=10, poppler_path="/usr/bin"):
    """Extrai texto de todas as páginas de um PDF."""
    try:
        images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
        text_parts = []
        for i, image in enumerate(images):
            logger.info(f"Processando página {i+1}/{len(images)}")
            processed_image = preprocess_image(np.array(image), binarization_threshold=binarization_threshold, denoise_strength=denoise_strength)
            if processed_image is not None:
                custom_config = f'--oem {oem} --psm {psm} -c preserve_interword_spaces=1 -l por+eng'
                text = pytesseract.image_to_string(processed_image, config=custom_config)
                text_parts.append(text)
            else:
                logger.warning(f"Pré-processamento da página {i+1} falhou, pulando para a próxima página.")
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Erro na extração de texto do PDF: {e}")
        st.error(f"Erro na extração de texto do PDF: {e}. Verifique se o arquivo PDF é válido.")
        return ""

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
    st.markdown("Carregue o arquivo PDF da sua NFS-e para extrair o texto.")

    uploaded_file = st.file_uploader("Carregue seu arquivo PDF", type="pdf")

    # Sidebar para configurações avançadas
    with st.sidebar.expander("Configurações de Imagem", expanded=False):
        dpi = st.slider("DPI da imagem", 200, 400, 300, 50, help="Resolução da imagem para OCR. Aumente para PDFs de baixa qualidade.")
        binarization_threshold = st.slider("Limiar de binarização", 10, 50, 31, 1, help="Ajuste para melhorar o contraste do texto.")
        denoise_strength = st.slider("Intensidade de remoção de ruído", 5, 20, 10, 1, help="Reduz ruídos na imagem, útil para PDFs escaneados.")

    with st.sidebar.expander("Configurações de OCR", expanded=False):
        psm = st.slider("PSM (Modo de Segmentação de Página)", 3, 13, 6, 1, help="Define como o Tesseract segmenta a página. Modo 6 é bom para blocos de texto.")
        oem = st.slider("OEM (Modo de Motor OCR)", 1, 3, 3, 1, help="Define o motor do Tesseract. Modo 3 é o motor neural mais preciso.")

    poppler_path_config = st.sidebar.text_input("Caminho Poppler (opcional)", "/usr/bin", help="Informe o caminho para o executável do Poppler se não estiver no PATH do sistema.")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            pdf_path = temp_file.name

        with st.spinner("Extraindo texto..."):
            extracted_text = extract_text_from_pdf(
                pdf_path,
                dpi=dpi,
                psm=psm,
                oem=oem,
                binarization_threshold=binarization_threshold,
                denoise_strength=denoise_strength,
                poppler_path=poppler_path_config
            )
            corrected_text = correct_text_format(extracted_text)

        if extracted_text:
            if validate_extracted_text(corrected_text):
                st.success("Texto extraído e validado com sucesso!")
            else:
                st.warning("O texto foi extraído, mas a validação falhou. Verifique o conteúdo. Pode não ser uma NFS-e ou a extração pode ter falhado parcialmente.")
            st.text_area("Texto extraído", corrected_text, height=300)
        else:
            st.error("Falha na extração do texto. Verifique o arquivo PDF e as configurações.")

        os.unlink(pdf_path)

if __name__ == "__main__":
    main()
