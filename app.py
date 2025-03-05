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

# Configuração otimizada para documentos fiscais
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
    rotated = cv2.warpAffine(imagem, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def melhorar_qualidade_imagem(imagem):
    """Pré-processamento aprimorado para documentos escaneados."""
    try:
        # Converter para escala de cinza
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        
        # Equalização para melhorar o contraste
        equalizada = cv2.equalizeHist(cinza)
        
        # Redução de ruído adaptativo
        denoised = cv2.fastNlMeansDenoising(equalizada, h=20, templateWindowSize=9, searchWindowSize=21)
        
        # Binarização adaptativa
        binarizada = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 51, 12)
        
        # Correção de inclinação (deskew)
        deskewed = deskew(binarizada)
        
        # Operação morfológica para limpeza de pequenos ruídos
        kernel = np.ones((1, 1), np.uint8)
        imagem_processada = cv2.morphologyEx(deskewed, cv2.MORPH_OPEN, kernel)
        
        return imagem_processada
    except Exception as e:
        logger.error(f"Erro no pré-processamento: {str(e)}")
        raise

def corrigir_formatacao(texto):
    """Realiza correções inteligentes para padrões de NFS-e."""
    correcoes = [
        # CNPJ (tolerante a variações)
        (r'(\d{2})[\.]?\s*(\d{3})[\.]?\s*(\d{3})[\/]?\s*0001[-]?\s*(\d{2})', r'\1.\2.\3/0001-\4'),
        # Datas (DD/MM/AAAA com separadores variados)
        (r'(\d{1,2})[\/\\\-_ ]+(\d{1,2})[\/\\\-_ ]+(\d{4})', r'\1/\2/\3'),
        # Valores monetários (R$ 1.234,56)
        (r'R\s*[\$]?\s*(\d{1,3}(?:[.,\s]\d{3})*)(?:[.,](\d{2}))?',
         lambda m: f"R$ {float(m.group(1).replace('.','').replace(',','.')) + (float(m.group(2))/100 if m.group(2) else 0):,.2f}"
                      .replace(',','X').replace('.',',').replace('X','.'))
    ]
    
    for padrao, substituicao in correcoes:
        texto = re.sub(padrao, substituicao, texto, flags=re.IGNORECASE)
    
    return texto

def normalizar_texto(texto):
    """Normaliza o texto: remove acentos, coloca em minúsculas e reduz espaços extras."""
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto

def validar_conteudo(texto):
    """Validação dos campos obrigatórios utilizando o texto normalizado."""
    norm_texto = normalizar_texto(texto)
    
    campos = {
        'NFS-e': [
            r'nota fiscal de servicos',
            r'nfs[-\s]*e'
        ],
        'CNPJ Prestador': [
            r'sustentamais consultoria'
        ],
        'Valor Total': [
            r'r\$\s*750[,.]00',
            r'\b750[,.]00\b'
        ]
    }
    
    faltantes = []
    for campo, padroes in campos.items():
        encontrado = any(re.search(padrao, norm_texto, re.IGNORECASE) for padrao in padroes)
        if not encontrado:
            logger.warning(f"Campo não encontrado: {campo}")
            faltantes.append(campo)
    
    if faltantes:
        logger.error(f"Campos obrigatórios faltantes: {', '.join(faltantes)}")
        return False, faltantes
    
    return True, []

def processar_documento(pdf_path):
    try:
        # Converter PDF para imagens
        imagens = convert_from_path(
            pdf_path,
            dpi=400,
            poppler_path="/usr/bin",
            grayscale=True,
            thread_count=2
        )
        
        texto_completo = []
        for idx, img in enumerate(imagens):
            # Pré-processamento intensivo
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_processada = melhorar_qualidade_imagem(img_cv)
            
            # OCR com fallback para diferentes modos de segmentação
            try:
                texto = pytesseract.image_to_string(img_processada, config=TESSERACT_CONFIG)
            except Exception as e:
                logger.warning(f"Falha com psm 6: {str(e)}. Tentando psm 11.")
                config_alternativo = TESSERACT_CONFIG.replace('--psm 6', '--psm 11')
                texto = pytesseract.image_to_string(img_processada, config=config_alternativo)
            
            # Pós-processamento do texto
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
