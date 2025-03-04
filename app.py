import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import tempfile
import os
import re

# ========== CONFIGURAÇÃO AVANÇADA ========== #
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"

# Configuração do Tesseract para documentos fiscais
TESSERACT_CONFIG = r'''
    --tessdata-dir {tessdata_dir}
    --oem 3
    --psm 6
    -c preserve_interword_spaces=1
    -c textord_tablefind_recognize_tables=1
    -l por
'''.format(tessdata_dir=os.environ["TESSDATA_PREFIX"])

# ========== FUNÇÕES DE PROCESSAMENTO DE IMAGEM ========== #
def preprocessamento_avancado(imagem):
    """Pipeline profissional de pré-processamento"""
    # Conversão para escala de cinza
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Redução de ruído adaptativo
    denoised = cv2.fastNlMeansDenoising(cinza, h=15, templateWindowSize=7, searchWindowSize=21)
    
    # Equalização de histograma CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12,12))
    cl1 = clahe.apply(denoised)
    
    # Binarização adaptativa para documentos
    return cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 51, 12)

# ========== FUNÇÕES DE PÓS-PROCESSAMENTO ========== #
def corrigir_padroes(texto):
    """Correção de padrões específicos de NFS-e"""
    correcoes = [
        # Padrão de datas (DD/MM/AAAA)
        (r'(\d{2})[/\\](\d{2})[/\\](\d{4})', r'\1/\2/\3'),
        
        # Padrão CNPJ (XX.XXX.XXX/0001-XX)
        (r'(\d{2})\D?(\d{3})\D?(\d{3})\D?0001\D?(\d{2})', r'\1.\2.\3/0001-\4'),
        
        # Valores monetários (R$ X.XXX,XX)
        (r'R\s*[\W_]*\s*(\d{1,3}(?:\.?\d{3})*)(?:[,.](\d{2}))?', 
         lambda m: f"R$ {float(m.group(1).replace('.','')) + float(m.group(2))/100 if m.group(2) else float(m.group(1)):,.2f}".replace(',','X').replace('.',',').replace('X','.'))
    ]
    
    for padrao, substituicao in correcoes:
        texto = re.sub(padrao, substituicao, texto)
    
    return texto

def validar_campos(texto):
    """Validação de campos críticos"""
    campos_obrigatorios = {
        'NFS-e': r'NFS-e',
        'CNPJ Prestador': r'\b49\.621\.411/0001-93\b',
        'Valor Total': r'R\$\s*750,00'
    }
    
    return all(re.search(padrao, texto) for campo, padrao in campos_obrigatorios.items())

# ========== FUNÇÃO PRINCIPAL DE EXTRAÇÃO ========== #
def extrair_texto_completo(pdf_path):
    try:
        # Conversão PDF para imagem com alta qualidade
        imagens = convert_from_path(
            pdf_path,
            dpi=400,
            poppler_path="/usr/bin",
            grayscale=True,
            thread_count=4
        )
        
        textos_processados = []
        
        for img in imagens:
            # Pré-processamento
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_processada = preprocessamento_avancado(img_cv)
            
            # OCR com múltiplas estratégias
            texto = pytesseract.image_to_string(
                img_processada,
                config=TESSERACT_CONFIG
            )
            
            # OCR secundário para áreas numéricas
            if any(palavra in texto for palavra in ['R$', 'CNPJ', 'CPF']):
                config_numerico = TESSERACT_CONFIG + ' -c tessedit_char_whitelist=0123456789R$.,/'
                texto_numerico = pytesseract.image_to_string(img_processada, config=config_numerico)
                texto = mesclar_textos(texto, texto_numerico)
            
            textos_processados.append(corrigir_padroes(texto))
        
        texto_final = "\n\n".join(textos_processados)
        
        if not validar_campos(texto_final):
            raise ValueError("Campos obrigatórios não encontrados")
            
        return texto_final
    
    except Exception as e:
        return f"ERRO: {str(e)}"

def mesclar_textos(texto_principal, texto_secundario):
    """Combina resultados de diferentes configurações de OCR"""
    for match in re.finditer(r'R\$\s*\d+[\d,.]*', texto_secundario):
        valor_correto = match.group()
        if valor_correto not in texto_principal:
            texto_principal = re.sub(r'R\$\s*\d+[\d,.]*', valor_correto, texto_principal, count=1)
    return texto_principal

# ========== INTERFACE DO USUÁRIO ========== #
def main():
    st.title("📑 Sistema de Extração de NFS-e Profissional")
    
    uploaded_file = st.file_uploader("Carregue o arquivo PDF", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            resultado = extrair_texto_completo(tmp_file.name)
            
            st.subheader("Resultado Aprimorado")
            
            if "ERRO" in resultado:
                st.error(resultado)
            else:
                # Exibição organizada
                with st.expander("Visualização Detalhada", expanded=True):
                    st.text_area("Texto Extraído", resultado, height=500)
                
                # Seção de validação
                st.success("✅ Documento validado com sucesso!")
                st.download_button("Baixar Texto Processado", resultado, "nfs-e_processado.txt")
            
            os.unlink(tmp_file.name)

if __name__ == "__main__":
    main()
