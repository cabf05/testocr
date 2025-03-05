import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import tempfile
import os
import re
import logging
from typing import List, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configurações de ambiente
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"

# Configurações OCR avançadas
TESSERACT_CONFIGS = [
    {
        "name": "Modo Padrão",
        "config": r'''
            --oem 3
            --psm 6
            -c preserve_interword_spaces=1
            -l por+eng
            --dpi 300
            tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,./-:() 
            tessedit_do_invert=1
        '''
    },
    {
        "name": "Modo Texto Livre", 
        "config": r'''
            --oem 3
            --psm 11
            -c preserve_interword_spaces=1
            -l por+eng
            --dpi 300
        '''
    },
    {
        "name": "Modo Automático",
        "config": r'''
            --oem 3
            --psm 3
            -c preserve_interword_spaces=1
            -l por+eng
            --dpi 300
        '''
    }
]

def melhorar_qualidade_imagem(imagem: np.ndarray) -> np.ndarray:
    """
    Processamento avançado de imagem para melhorar qualidade OCR
    
    Args:
        imagem (np.ndarray): Imagem de entrada
    
    Returns:
        np.ndarray: Imagem processada
    """
    try:
        # Conversão para escala de cinza
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        
        # Remoção de ruído adaptativa
        denoised = cv2.fastNlMeansDenoising(
            cinza, 
            h=10, 
            templateWindowSize=7, 
            searchWindowSize=21
        )
        
        # Equalização de histograma para melhorar contraste
        equalized = cv2.equalizeHist(denoised)
        
        # Binarização Otsu
        _, binarized = cv2.threshold(
            equalized, 
            0, 
            255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return binarized
    
    except Exception as e:
        logger.error(f"Erro no pré-processamento: {str(e)}")
        return imagem  # Fallback para imagem original

def corrigir_formatacao(texto: str) -> str:
    """
    Correções inteligentes para padrões de documentos fiscais
    
    Args:
        texto (str): Texto extraído
    
    Returns:
        str: Texto corrigido
    """
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

def validar_conteudo(texto: str) -> Tuple[bool, List[str]]:
    """
    Validação tolerante de conteúdo do documento
    
    Args:
        texto (str): Texto completo do documento
    
    Returns:
        Tuple[bool, List[str]]: Validade e campos faltantes
    """
    campos = {
        'NFS-e': [
            r'NOTA FISCAL DE SERVIÇOS (ELETRÔNICA|ELETRONICA)',
            r'NFS[\s\-_]?e'
        ],
        'CNPJ Prestador': [
            r'40[\D]?621[\D]?411[/]0001[\D]?53',
            r'SUSTENTAMAIS CONSULTORIA'
        ],
        'Valor Total': [
            r'VALOR\s*TOTAL.*R\$?\s*750[,.]?00',
            r'TOTAL\s*DA\s*NOTA.*750'
        ]
    }
    
    faltantes = []
    for campo, padroes in campos.items():
        encontrado = any(re.search(padrao, texto, re.IGNORECASE) for padrao in padroes)
        if not encontrado:
            logger.warning(f"Campo não encontrado: {campo}")
            faltantes.append(campo)
    
    return len(faltantes) == 0, faltantes

def processar_documento(pdf_path: str) -> str:
    """
    Processamento completo do documento PDF com múltiplas estratégias OCR
    
    Args:
        pdf_path (str): Caminho do arquivo PDF
    
    Returns:
        str: Texto extraído ou mensagem de erro
    """
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
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_processada = melhorar_qualidade_imagem(img_cv)
            
            texto_pagina = ""
            for estrategia in TESSERACT_CONFIGS:
                try:
                    texto = pytesseract.image_to_string(
                        img_processada, 
                        config=estrategia['config']
                    )
                    
                    if texto.strip():
                        logger.info(f"Sucesso com estratégia: {estrategia['name']}")
                        texto_pagina = texto
                        break
                except Exception as e:
                    logger.warning(f"Falha na estratégia {estrategia['name']}: {e}")
            
            # Pós-processamento
            texto_corrigido = corrigir_formatacao(texto_pagina)
            texto_completo.append(texto_corrigido)
            
            logger.info(f"Página {idx+1} processada")
        
        texto_final = "\n\n".join(texto_completo)
        valido, campos_faltantes = validar_conteudo(texto_final)
        
        if not valido:
            return f"ERRO: Campos obrigatórios não encontrados ({', '.join(campos_faltantes)})"
        
        return texto_final
    
    except Exception as e:
        logger.error(f"Erro crítico no processamento: {str(e)}")
        return f"ERRO CRÍTICO: {str(e)}"

def main():
    """Interface principal do Streamlit"""
    st.title("📑 Sistema de Extração de NFS-e (Versão 3.0)")
    
    uploaded_file = st.file_uploader("Carregue o arquivo PDF", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            try:
                tmp_file.write(uploaded_file.read())
                tmp_file.close()
                
                resultado = processar_documento(tmp_file.name)
                
                if resultado.startswith("ERRO"):
                    st.error(resultado)
                    with st.expander("Detalhes"):
                        st.code(resultado)
                else:
                    st.success("✅ Documento processado com sucesso!")
                    with st.expander("Visualizar Texto Extraído"):
                        st.text_area("Conteúdo", resultado, height=500)
                    
                    st.download_button(
                        "Baixar Texto Extraído", 
                        resultado, 
                        "nfs-e_processado.txt"
                    )
            
            except Exception as e:
                st.error(f"Erro inesperado: {e}")
            
            finally:
                # Garantir que o arquivo temporário seja removido
                os.unlink(tmp_file.name)

if __name__ == "__main__":
    main()
