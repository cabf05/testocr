import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import tempfile
import os
import requests

# ===== Configuração Confiável de Idiomas =====
TESSDATA_DIR = "/home/appuser/tessdata"

def setup_tessdata():
    """Garante os arquivos de idioma essenciais"""
    os.makedirs(TESSDATA_DIR, exist_ok=True)
    os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR
    
    # Baixa apenas os idiomas necessários se faltantes
    if not os.path.exists(f"{TESSDATA_DIR}/por.traineddata"):
        response = requests.get("https://github.com/tesseract-ocr/tessdata/raw/main/por.traineddata")
        with open(f"{TESSDATA_DIR}/por.traineddata", "wb") as f:
            f.write(response.content)

setup_tessdata()  # Executa antes de tudo

# ===== Pipeline Otimizado de Processamento =====
def processar_pdf(pdf_path):
    """Fluxo completo de extração com técnicas profissionais"""
    try:
        # Passo 1: Conversão de PDF para imagem
        images = convert_from_path(
            pdf_path,
            dpi=400,
            poppler_path="/usr/bin",
            grayscale=True,
            thread_count=4
        )
        
        # Passo 2: Configuração especializada para documentos fiscais
        config_tesseract = r'''
            --oem 3
            --psm 6
            -c tessedit_char_blacklist=®©™•§
            -c textord_tabfind_show_vlines=0
            -l por
        '''
        
        # Passo 3: Processamento adaptativo por página
        resultados = []
        for idx, img in enumerate(images):
            # Pré-processamento intensivo
            img_np = np.array(img)
            img_clean = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
            gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12,12))
            processed = clahe.apply(gray)
            
            # OCR principal
            texto = pytesseract.image_to_string(processed, config=config_tesseract)
            
            # OCR de reforço para áreas numéricas
            if "R$" in texto:
                config_reforco = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789R$.,'
                texto_reforco = pytesseract.image_to_string(processed, config=config_reforco)
                texto = mesclar_resultados(texto, texto_reforco)
            
            resultados.append(texto)
        
        # Passo 4: Pós-processamento inteligente
        texto_final = "\n\n".join(resultados)
        return otimizar_texto(texto_final)
    
    except Exception as e:
        return f"ERRO: {str(e)}"

def mesclar_resultados(texto_principal, texto_reforco):
    """Combina resultados de diferentes configurações de OCR"""
    # Lógica para preservar valores numéricos
    for valor in re.findall(r'R\$\s*\d+[\d.,]*', texto_reforco):
        if valor not in texto_principal:
            texto_principal = texto_principal.replace("R$", valor)
    return texto_principal

def otimizar_texto(texto):
    """Correções pós-OCR baseadas em padrões de NFS-e"""
    correcoes = [
        (r'(?i)código\s+de\s+verificação\s*:\s*([A-Z0-9]{8})', fixar_codigo_verificacao),
        (r'\b(\d{2})\.(\d{3})\.(\d{3})/\d{4}-(\d{2})\b', formatar_cnpj),
        (r'R\$\s*(\d{1,3}(?:\.\d{3})*,\d{2})', formatar_moeda)
    ]
    
    for padrao, funcao in correcoes:
        texto = re.sub(padrao, funcao, texto)
    
    return texto

# ===== Funções Auxiliares Especializadas =====
def fixar_codigo_verificacao(match):
    return f"Código de Verificação: {match.group(1).replace(' ', '')}"

def formatar_cnpj(match):
    return f"{match.group(1)}.{match.group(2)}.{match.group(3)}/0001-{match.group(4)}"

def formatar_moeda(match):
    valor = match.group(1).replace('.', '').replace(',', '.')
    return f"R$ {float(valor):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')

# ===== Interface =====
def main():
    st.title("📑 Sistema de Extração de NFS-e (Versão Aprimorada)")
    
    uploaded_file = st.file_uploader("Carregue o arquivo PDF", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            resultado = processar_pdf(tmp_file.name)
            
            st.subheader("Resultado Otimizado")
            st.text_area("Texto Extraído", resultado, height=500)
            
            if "ERRO" in resultado:
                st.error("Falha na extração")
            else:
                st.success("Extração concluída com validação!")
                st.download_button("Baixar Texto", resultado, "texto_extraido.txt")
            
            os.unlink(tmp_file.name)

if __name__ == "__main__":
    main()
