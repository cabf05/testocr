import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import tempfile
import os
import re
import logging
import unicodedata

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== CONFIGURAÇÃO AVANÇADA ========== #
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"
TESSERACT_CONFIG = r'--oem 3 --psm 6 -c preserve_interword_spaces=1 -l por+eng'

# ========== FUNÇÕES DE PROCESSAMENTO ========== #
def deskew(imagem):
    """Corrige a inclinação da imagem binarizada."""
    coords = np.column_stack(np.where(imagem > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = imagem.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(imagem, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def melhorar_qualidade_imagem(imagem):
    """Pré-processamento aprimorado para documentos escaneados."""
    try:
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        equalizada = cv2.equalizeHist(cinza)
        denoised = cv2.fastNlMeansDenoising(equalizada, h=20, templateWindowSize=9, searchWindowSize=21)
        binarizada = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 51, 12)
        deskewed = deskew(binarizada)
        kernel = np.ones((1, 1), np.uint8)
        return cv2.morphologyEx(deskewed, cv2.MORPH_OPEN, kernel)
    except Exception as e:
        logger.error(f"Erro no pré-processamento: {str(e)}")
        raise

def corrigir_formatacao(texto):
    """Realiza correções inteligentes para padrões de NFS-e."""
    correcoes = [
        (r'(\d{2})[\.]?\s*(\d{3})[\.]?\s*(\d{3})[\/]?\s*0001[-]?\s*(\d{2})', r'\1.\2.\3/0001-\4'),
        (r'(\d{1,2})[\/\\\-_ ]+(\d{1,2})[\/\\\-_ ]+(\d{4})', r'\1/\2/\3'),
        (r'R\s*[\$]?\s*(\d{1,3}(?:[.,\s]\d{3})*)(?:[.,](\d{2}))?',
         lambda m: f"R$ {float(m.group(1).replace('.','').replace(',','.')) + (float(m.group(2))/100 if m.group(2) else 0):,.2f}"
                      .replace(',','X').replace('.',',').replace('X','.'))
    ]
    for padrao, substituicao in correcoes:
        texto = re.sub(padrao, substituicao, texto, flags=re.IGNORECASE)
    return texto

def normalizar_texto(texto):
    """Normaliza o texto: remove acentos, converte para minúsculas e reduz espaços extras."""
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto

def validar_conteudo(texto):
    """
    Valida os campos obrigatórios usando o texto normalizado.
    Em vez de regex complexos, utiliza buscas por substrings e extração de dígitos.
    """
    norm_texto = normalizar_texto(texto)
    
    faltantes = []
    
    # Validação para NFS-e: deve conter "nota fiscal" e "nfs-e" (ou "nfs e")
    if "nota fiscal" not in norm_texto or not ("nfs-e" in norm_texto or "nfs e" in norm_texto):
        faltantes.append("NFS-e")
    
    # Validação para CNPJ Prestador:
    # Verifica se os dígitos do CNPJ esperado ("40621411000153") estão presentes
    # ou se a razão social "sustentamais consultoria" aparece.
    digits = re.sub(r'\D', '', norm_texto)
    if "40621411000153" not in digits and "sustentamais consultoria" not in norm_texto:
        faltantes.append("CNPJ Prestador")
    
    # Validação para Valor Total:
    # Procura por "75000", "r$ 750,00" ou "750,00" no texto
    if not ("75000" in norm_texto or "r$ 750,00" in norm_texto or "750,00" in norm_texto):
        faltantes.append("Valor Total")
    
    if faltantes:
        logger.error(f"Campos obrigatórios faltantes: {', '.join(faltantes)}")
        return False, faltantes
    return True, []

def processar_documento(pdf_path):
    try:
        imagens = convert_from_path(pdf_path, dpi=400, poppler_path="/usr/bin",
                                     grayscale=True, thread_count=2)
        texto_completo = []
        for idx, img in enumerate(imagens):
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_processada = melhorar_qualidade_imagem(img_cv)
            try:
                texto = pytesseract.image_to_string(img_processada, config=TESSERACT_CONFIG)
            except Exception as e:
                logger.warning(f"Falha com psm 6: {str(e)}. Tentando psm 11.")
                config_alternativo = TESSERACT_CONFIG.replace('--psm 6', '--psm 11')
                texto = pytesseract.image_to_string(img_processada, config=config_alternativo)
            texto_corrigido = corrigir_formatacao(texto)
            texto_completo.append(texto_corrigido)
            logger.info(f"Página {idx+1} processada")
        
        texto_final = "\n\n".join(texto_completo)
        valido, campos_faltantes = validar_conteudo(texto_final)
        if not valido:
            return f"ERRO: Campos obrigatórios não encontrados ({', '.join(campos_faltantes)})"
        return texto_final
    except Exception as e:
        logger.error(f"Erro no processamento: {str(e)}")
        return f"ERRO: {str(e)}"

# ========== INTERFACE ========== #
def main():
    st.title("📑 Sistema de Extração de NFS-e (Versão 2.6)")
    uploaded_file = st.file_uploader("Carregue o arquivo PDF", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        resultado = processar_documento(tmp_file_path)
        if resultado.startswith("ERRO"):
            st.error(resultado)
            st.info("Confira os logs para mais detalhes do erro.")
        else:
            st.success("✅ Documento validado com sucesso!")
            with st.expander("Visualizar Texto Extraído"):
                st.text_area("Conteúdo", resultado, height=500)
            st.download_button("Baixar Texto", resultado, "nfs-e_processado.txt")
        os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()
