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

# Diretório para arquivos de debug
DEBUG_DIR = "/tmp/nfs_ocr_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

def processar_imagem(imagem: np.ndarray, tecnica: str = "padrao") -> str:
    """
    Processar imagem com múltiplas estratégias OCR
    
    Args:
        imagem (np.ndarray): Imagem de entrada
        tecnica (str): Técnica de processamento
    
    Returns:
        str: Texto extraído
    """
    tecnicas_processamento = {
        "padrao": imagem,
        "cinza": cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY),
        "adaptativo": cv2.adaptiveThreshold(
            cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY), 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
    }
    
    imagem_processada = tecnicas_processamento.get(tecnica, imagem)
    
    configs_ocr = [
        {
            "config": r'--oem 3 --psm 6 -l por+eng',
            "nome": "Padrão"
        },
        {
            "config": r'--oem 3 --psm 11 -l por+eng',
            "nome": "Texto Livre"
        }
    ]
    
    for config in configs_ocr:
        try:
            texto = pytesseract.image_to_string(
                imagem_processada, 
                config=config['config']
            )
            
            if texto.strip():
                logger.info(f"Sucesso com técnica {tecnica} e configuração {config['nome']}")
                return texto
        except Exception as e:
            logger.warning(f"Falha com técnica {tecnica} e configuração {config['nome']}: {e}")
    
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
            r'NOTA\s*FISCAL\s*DE\s*SERVIÇOS\s*(ELETRÔNICA|ELETRONICA)',
            r'NFS[\s\-_]?e',
            r'NOTA\s*FISCAL'
        ],
        'CNPJ Prestador': [
            r'40[\D]?621[\D]?411[/]0001[\D]?53',
            r'SUSTENTAMAIS\s*CONSULTORIA',
            r'CNPJ:?\s*40\.621\.411/0001-53'
        ],
        'Valor Total': [
            r'VALOR\s*TOTAL.*R\$?\s*750[,.]?00',
            r'TOTAL\s*DA\s*NOTA.*750',
            r'R\$\s*750[,.]?00'
        ]
    }
    
    campos_faltantes = []
    for campo, padroes in campos_validacao.items():
        if not any(re.search(padrao, texto, re.IGNORECASE | re.MULTILINE) for padrao in padroes):
            campos_faltantes.append(campo)
            logger.warning(f"Campo não encontrado: {campo}")
    
    # Salvar texto completo para debug
    with open(os.path.join(DEBUG_DIR, "texto_completo.txt"), "w", encoding="utf-8") as f:
        f.write(texto)
    
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
            # Salvar imagem original
            img.save(os.path.join(DEBUG_DIR, f"pagina_{idx+1}_original.png"))
            
            # Array numpy para processamento
            img_array = np.array(img)
            
            # Tentar diferentes técnicas de processamento
            tecnicas = ["padrao", "cinza", "adaptativo"]
            texto_pagina = ""
            
            for tecnica in tecnicas:
                texto = processar_imagem(img_array, tecnica)
                if texto.strip():
                    texto_pagina = texto
                    break
            
            # Salvar texto extraído
            with open(os.path.join(DEBUG_DIR, f"pagina_{idx+1}_texto.txt"), "w", encoding="utf-8") as f:
                f.write(texto_pagina)
            
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
    st.title("📑 Sistema de Extração de NFS-e (Versão Debug)")
    
    # Limpar diretório de debug
    for arquivo in os.listdir(DEBUG_DIR):
        os.remove(os.path.join(DEBUG_DIR, arquivo))
    
    uploaded_file = st.file_uploader("Carregue o arquivo PDF", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            try:
                tmp_file.write(uploaded_file.read())
                tmp_file.close()
                
                resultado = processar_documento(tmp_file.name)
                
                if resultado.startswith("ERRO"):
                    st.error(resultado)
                    
                    # Mostrar conteúdo extraído para análise
                    try:
                        with open(os.path.join(DEBUG_DIR, "texto_completo.txt"), "r", encoding="utf-8") as f:
                            texto_debug = f.read()
                        
                        with st.expander("Detalhes do Texto Extraído"):
                            st.text_area("Conteúdo Extraído", texto_debug, height=300)
                    except Exception as e:
                        st.warning(f"Não foi possível ler arquivo de debug: {e}")
                else:
                    st.success("✅ Documento processado com sucesso!")
                    
                    with st.expander("Visualizar Texto Extraído"):
                        st.text_area("Conteúdo", resultado, height=500)
                    
                    st.download_button(
                        "Baixar Texto Extraído", 
                        resultado, 
                        "nfs-e_processado.txt"
                    )
                
                # Mostrar arquivos de debug
                st.divider()
                st.subheader("🔍 Arquivos de Diagnóstico")
                debug_files = os.listdir(DEBUG_DIR)
                for file in debug_files:
                    with open(os.path.join(DEBUG_DIR, file), "rb") as f:
                        st.download_button(
                            f"Baixar {file}", 
                            f.read(), 
                            file,
                            key=file
                        )
            
            except Exception as e:
                st.error(f"Erro inesperado: {e}")
            
            finally:
                # Remover arquivo temporário
                os.unlink(tmp_file.name)

if __name__ == "__main__":
    main()
