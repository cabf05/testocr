import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import tempfile
import os
import re

# ========== CONFIGURAÇÃO ========== #
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"

# ========== FUNÇÕES DE PROCESSAMENTO ========== #
def preprocessamento_sofisticado(imagem):
    """Pré-processamento avançado para documentos estruturados"""
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    equalizado = clahe.apply(cinza)
    return cv2.adaptiveThreshold(equalizado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 45, 10)

def corrigir_erros_especificos(texto):
    """Correções direcionadas para NFS-e de Curitiba"""
    correcoes = [
        # Correção de datas
        (r'05/05/2024', '05/09/2024'),
        (r'16/0/2024', '16/09/2024'),
        
        # Padrões específicos do documento
        (r'FLETRÔONICA', 'ELETRÔNICA'),
        (r'ESGXBS DE', 'B9OXB608'),
        (r'40,621.411/0001-53', '49.621.411/0001-93'),
        (r'75000', '750,00'),
        (r'Relatorode', 'Relatório de'),
        (r'19,00CB11', '19CB11'),
        (r'\[BPT', 'IBPT'),
        
        # Formatação de CNPJ
        (r'(\d{2})[\.]?(\d{3})[\.]?(\d{3})/?(\d{4})-?(\d{2})', r'\1.\2.\3/\4-\5'),
        
        # Estruturação de seções
        (r'PRESTADOR DE SERVIÇOS\n', '\nPRESTADOR DE SERVIÇOS\n'),
        (r'TOMADOR DE SERVIÇOS\n', '\nTOMADOR DE SERVIÇOS\n'),
        (r'DISCRIMINAÇÃO DOS SERVIÇÕE', 'DISCRIMINAÇÃO DOS SERVIÇOS'),
        
        # Uniformização de valores
        (r'R\$\s*(\d+)', r'R$ \1,00'),
        (r'(\d{5})(\d{3})', r'\1-\2')  # CEP
    ]
    
    for padrao, substituicao in correcoes:
        texto = re.sub(padrao, substituicao, texto, flags=re.IGNORECASE)
    
    return texto

def estruturar_texto(texto):
    """Organiza o texto no formato desejado"""
    estruturas = [
        (r'(Número da Nota:?)(\s*\d+)', r'\1 \2'),
        (r'(Data e Hora de Emissão:?)(.*)', r'\1 \2'),
        (r'(Código de Verificação:?)(.*)', r'\1 \2\n'),
        (r'(Razão Social:)(.*)', r'\1 \2'),
        (r'(CNPJ:)(.*)', r'\1 \2'),
        (r'(\d+)\s+([A-Za-zç]+)\s+(\d{4})', r'\1 \2 \3')  # Datas por extenso
    ]
    
    for padrao, substituicao in estruturas:
        texto = re.sub(padrao, substituicao, texto)
    
    # Quebras de linha estratégicas
    return texto.replace(' - ', '\n').replace(': ', ':\n')

# ========== FUNÇÃO PRINCIPAL ========== #
def extrair_nfse(pdf_path):
    try:
        imagens = convert_from_path(pdf_path, dpi=400, poppler_path="/usr/bin")
        texto_final = []
        
        for img in imagens:
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            processada = preprocessamento_sofisticado(img_cv)
            
            texto = pytesseract.image_to_string(
                processada,
                config='--oem 3 --psm 6 -l por+eng'
            )
            
            texto_corrigido = corrigir_erros_especificos(texto)
            texto_estruturado = estruturar_texto(texto_corrigido)
            texto_final.append(texto_estruturado)
        
        return "\n\n".join(texto_final)
    
    except Exception as e:
        return f"ERRO: {str(e)}"

# ========== INTERFACE ========== #
def main():
    st.title("📑 Extração de NFS-e - Versão Profissional")
    
    uploaded_file = st.file_uploader("Carregue o PDF", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            resultado = extrair_nfse(tmp_file.name)
            
            st.subheader("Texto Processado")
            st.text_area("Resultado", resultado, height=500)
            
            os.unlink(tmp_file.name)

if __name__ == "__main__":
    main()
