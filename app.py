import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import tempfile
import os
import re
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== CONFIGURAÇÃO AVANÇADA ========== #
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"

# Configuração otimizada para documentos fiscais
TESSERACT_CONFIG = r'''
    --oem 3
    --psm 6
    -c preserve_interword_spaces=1
    -l por+eng
'''

# ========== FUNÇÕES DE PROCESSAMENTO ========== #
def melhorar_qualidade_imagem(imagem):
    """Pré-processamento profissional para documentos escaneados"""
    try:
        # Converter para escala de cinza
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        
        # Redução de ruído adaptativo
        denoised = cv2.fastNlMeansDenoising(cinza, h=20, templateWindowSize=9, searchWindowSize=21)
        
        # Realce de bordas
        bordas = cv2.Canny(denoised, 50, 150)
        
        # Combinação dos resultados
        combinado = cv2.addWeighted(denoised, 0.7, bordas, 0.3, 0)
        
        # Binarização adaptativa
        return cv2.adaptiveThreshold(combinado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 51, 12)
    except Exception as e:
        logger.error(f"Erro no pré-processamento: {str(e)}")
        raise

def corrigir_formatacao(texto):
    """Correções inteligentes para padrões de NFS-e"""
    correcoes = [
        # CNPJ (tolerante a variações)
        (r'(\d{2})[\.]?(\d{3})[\.]?(\d{3})[/]?0001[-]?(\d{2})', r'\1.\2.\3/0001-\4'),
        
        # Datas (DD/MM/AAAA com separadores variados)
        (r'(\d{1,2})[\/\\\-_ ]+(\d{1,2})[\/\\\-_ ]+(\d{4})', r'\1/\2/\3'),
        
        # Valores monetários (R$ 1.234,56)
        (r'R\s*[\$\*]?\s*(\d{1,3}(?:[.,\s]\d{3})*)(?:[.,](\d{2}))?', 
         lambda m: f"R$ {float(m.group(1).replace('.','').replace(',','.')) + (float(m.group(2))/100 if m.group(2) else 0):,.2f}".replace(',','X').replace('.',',').replace('X','.'))
    ]
    
    for padrao, substituicao in correcoes:
        texto = re.sub(padrao, substituicao, texto, flags=re.IGNORECASE)
    
    return texto

def validar_conteudo(texto):
    """Validação tolerante com logging detalhado"""
    campos = {
        'NFS-e': [
            r'NFS[\s\-_]?e',  # Tolerância para diferentes formatações
            r'NOTA FISCAL DE SERVIÇOS ELETRÔNICA'
        ],
        'CNPJ Prestador': [
            r'49[\D]?621[\D]?411[/]0001[\D]?93',  # CNPJ com separadores variados
            r'SUSTENTAMAIS CONSULTORIA'
        ],
        'Valor Total': [
            r'R\$\s*750[\D]?00',  # Tolerância para formatação numérica
            r'VALOR TOTAL DA NOTA.*750'
        ]
    }
    
    faltantes = []
    for campo, padroes in campos.items():
        encontrado = any(re.search(padrao, texto, re.IGNORECASE) for padrao in padroes)
        if not encontrado:
            logger.warning(f"Campo não encontrado: {campo}")
            faltantes.append(campo)
    
    if faltantes:
        logger.error(f"Campos obrigatórios faltantes: {', '.join(faltantes)}")
        return False, faltantes
    
    return True, []

# ========== FUNÇÃO PRINCIPAL DE EXTRAÇÃO ========== #
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
            
            # OCR com fallback
            try:
                texto = pytesseract.image_to_string(img_processada, config=TESSERACT_CONFIG)
            except:
                texto = pytesseract.image_to_string(img_processada, config=TESSERACT_CONFIG.replace('psm 6', 'psm 11'))
            
            # Pós-processamento
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
    st.title("📑 Sistema de Extração de NFS-e (Versão 2.1)")
    
    uploaded_file = st.file_uploader("Carregue o arquivo PDF", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            resultado = processar_documento(tmp_file.name)
            
            if resultado.startswith("ERRO"):
                st.error(resultado)
                with st.expander("Detalhes do Erro"):
                    st.code(texto_final)  # Mostra o texto extraído para análise
            else:
                st.success("✅ Documento validado com sucesso!")
                with st.expander("Visualizar Texto Extraído"):
                    st.text_area("Conteúdo", resultado, height=500)
                
                st.download_button("Baixar Texto", resultado, "nfs-e_processado.txt")
            
            os.unlink(tmp_file.name)

if __name__ == "__main__":
    main()
