import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import tempfile
import os
import re
import logging
from typing import List, Tuple, Dict

# Configurar logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configurações de ambiente
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"

def preprocessar_imagem(imagem: np.ndarray) -> np.ndarray:
    """
    Pré-processamento avançado de imagem para melhorar OCR
    
    Args:
        imagem (np.ndarray): Imagem de entrada
    
    Returns:
        np.ndarray: Imagem processada
    """
    try:
        # Conversão para escala de cinza
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        
        # Remoção de ruído
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Equalização de histograma
        equalized = cv2.equalizeHist(denoised)
        
        # Binarização adaptativa
        binary = cv2.adaptiveThreshold(
            equalized, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # Dilatação para melhorar caracteres
        kernel = np.ones((1,1), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        return dilated
    except Exception as e:
        logger.error(f"Erro no pré-processamento: {e}")
        return imagem

def processar_imagem(imagem: np.ndarray) -> str:
    """
    Extração de texto com múltiplas técnicas
    
    Args:
        imagem (np.ndarray): Imagem de entrada
    
    Returns:
        str: Texto extraído
    """
    tecnicas = [
        ("Original", imagem),
        ("Preprocessada", preprocessar_imagem(imagem)),
        ("Inverso", cv2.bitwise_not(preprocessar_imagem(imagem)))
    ]
    
    configuracoes_tesseract = [
        {
            "config": r'''
                --oem 3
                --psm 6
                -c preserve_interword_spaces=1
                -l por+eng
                --dpi 300
                tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,./-:() 
                tessedit_do_invert=1
            ''',
            "nome": "Modo Padrão"
        },
        {
            "config": r'''
                --oem 3
                --psm 11
                -c preserve_interword_spaces=1
                -l por+eng
            ''',
            "nome": "Modo Texto Livre"
        }
    ]
    
    for nome_tecnica, img_processada in tecnicas:
        for config in configuracoes_tesseract:
            try:
                texto = pytesseract.image_to_string(
                    img_processada, 
                    config=config['config']
                )
                
                if texto.strip():
                    logger.info(f"Sucesso com técnica: {nome_tecnica}, Configuração: {config['nome']}")
                    return texto
            except Exception as e:
                logger.warning(f"Falha: {nome_tecnica} + {config['nome']} - {e}")
    
    return ""

def corrigir_formatacao(texto: str) -> str:
    """
    Correções inteligentes para documentos fiscais
    
    Args:
        texto (str): Texto extraído
    
    Returns:
        str: Texto corrigido
    """
    correcoes = [
        # CNPJ
        (r'(\d{2})[\.]?(\d{3})[\.]?(\d{3})[/]?0001[-]?(\d{2})', r'\1.\2.\3/0001-\4'),
        
        # Datas
        (r'(\d{1,2})[\/\\\-_ ]+(\d{1,2})[\/\\\-_ ]+(\d{4})', r'\1/\2/\3'),
        
        # Valores monetários
        (r'R\s*[\$\*]?\s*(\d{1,3}(?:[.,\s]\d{3})*)(?:[.,](\d{2}))?', 
         lambda m: f"R$ {float(m.group(1).replace('.','').replace(',','.')) + (float(m.group(2))/100 if m.group(2) else 0):,.2f}".replace(',','X').replace('.',',').replace('X','.'))
    ]
    
    for padrao, substituicao in correcoes:
        texto = re.sub(padrao, substituicao, texto, flags=re.IGNORECASE)
    
    return texto

def validar_conteudo(texto: str) -> Tuple[bool, List[str]]:
    """
    Validação flexível de conteúdo do documento
    
    Args:
        texto (str): Texto completo
    
    Returns:
        Tuple[bool, List[str]]: Validade e campos faltantes
    """
    campos_validacao: Dict[str, List[str]] = {
        'NFS-e': [
            r'NOTA\s*FISCAL\s*DE\s*SERVIÇOS?\s*(ELETRÔNICA|ELETRONICA)',
            r'NFS[\s\-_]?e'
        ],
        'CNPJ Prestador': [
            r'40[\D.]?621[\D.]?411[/]0001[\D\-]?93',
            r'SUSTENTAMAIS\s*CONSULTORIA'
        ],
        'Valor Total': [
            r'VALOR\s*TOTAL.*R\$?\s*750[,.]?00',
            r'TOTAL\s*DA\s*NOTA.*750'
        ]
    }
    
    # Processar texto removendo quebras de linha
    texto_processado = texto.replace('\n', ' ').replace('\r', '')
    
    campos_faltantes = []
    for campo, padroes in campos_validacao.items():
        if not any(re.search(padrao, texto_processado, re.IGNORECASE) for padrao in padroes):
            campos_faltantes.append(campo)
            logger.warning(f"Campo não encontrado: {campo}")
    
    return len(campos_faltantes) == 0, campos_faltantes

def processar_documento(pdf_path: str) -> str:
    """
    Processamento completo do documento PDF
    
    Args:
        pdf_path (str): Caminho do arquivo PDF
    
    Returns:
        str: Texto extraído ou mensagem de erro
    """
    try:
        # Converter PDF com resolução aumentada
        imagens = convert_from_path(
            pdf_path,
            dpi=500,  # Resolução aumentada
            poppler_path="/usr/bin",
            grayscale=False
        )
        
        texto_completo = []
        for idx, img in enumerate(imagens):
            # Converter imagem para array numpy
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Processar imagem
            texto_pagina = processar_imagem(img_array)
            
            # Pós-processamento
            texto_corrigido = corrigir_formatacao(texto_pagina)
            texto_completo.append(texto_corrigido)
        
        # Consolidar texto
        texto_final = "\n\n".join(texto_completo)
        
        # Validar conteúdo
        valido, campos_faltantes = validar_conteudo(texto_final)
        
        if not valido:
            return f"ERRO: Campos obrigatórios não encontrados ({', '.join(campos_faltantes)})"
        
        return texto_final
    
    except Exception as e:
        logger.error(f"Erro crítico no processamento: {str(e)}")
        return f"ERRO CRÍTICO: {str(e)}"

def main():
    """Interface principal do Streamlit"""
    st.title("📑 Sistema de Extração de NFS-e (Versão OCR Avançada)")
    
    uploaded_file = st.file_uploader("Carregue o arquivo PDF", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            try:
                tmp_file.write(uploaded_file.read())
                tmp_file.close()
                
                resultado = processar_documento(tmp_file.name)
                
                if resultado.startswith("ERRO"):
                    st.error(resultado)
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
                # Remover arquivo temporário
                os.unlink(tmp_file.name)

if __name__ == "__main__":
    main()
