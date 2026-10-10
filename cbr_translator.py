"""
Módulo de Tradução e Normalização Terminológica Radiológica (Padrão CBR - Colégio Brasileiro de Radiologia)
Conecta o modelo de tradução neural local (NLLB-200) com um dicionário especializado em radiologia torácica.
"""

import re
import torch
from typing import Optional, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Dicionário Terminológico Radiológico: Português -> Inglês (para perguntas do usuário / prompts)
PT_TO_EN_LEXICON = [
    (r'\bderrame pleural\b|\bderrames pleurais\b', 'pleural effusion'),
    (r'\bpneumot[oó]rax\b', 'pneumothorax'),
    (r'\bhidropneumot[oó]rax\b', 'hydropneumothorax'),
    (r'\bcardiomegalia\b|\baumento da [aá]rea card[ií]aca\b|\baumento do cora[cç][aã]o\b|\bcora[cç][aã]o aumentado\b', 'cardiomegaly'),
    (r'\bsilhueta card[ií]aca\b', 'cardiac silhouette'),
    (r'\bcontornos mediastinais\b|\bmediastino\b', 'mediastinal contours'),
    (r'\bvias a[eé]reas\b|\bvias a[eé]rea\b', 'airway'),
    (r'\btraqueia\b|\btraqu[eé]ia\b', 'trachea'),
    (r'\bcarina\b', 'carina'),
    (r'\bbr[oô]nquio\b|\bbr[oô]nquios\b', 'bronchi'),
    (r'\bcampos pulmonares\b|\bcampos pleuropulmonares\b|\bpulm[oõ]es\b|\bpar[eê]nquima pulmonar\b', 'lungs and breathing'),
    (r'\bconsolida[cç][aã]o\b|\bconsolida[cç][oõ]es\b', 'consolidation'),
    (r'\binfiltrado\b|\binfiltrados\b|\bopacidade\b|\bopacidades\b', 'opacities and infiltrates'),
    (r'\batelectasia\b|\batelectasias\b', 'atelectasis'),
    (r'\bedema pulmonar\b|\bedema agudo\b', 'pulmonary edema'),
    (r'\bseios costofr[eê]nicos\b|\bseio costofr[eê]nico\b', 'costophrenic angles'),
    (r'\bc[uú]pulas diafragm[aá]ticas\b|\bdiafragma\b', 'diaphragm'),
    (r'\barcabou[cç]o [oó]sseo\b|\bcostelas\b|\bcostela\b|\barcos costais\b', 'ribs and thoracic bones'),
    (r'\bfratura de costela\b|\bfraturas costais\b|\bfratura\b', 'rib fracture'),
    (r'\bpartes moles\b', 'soft tissues'),
    (r'\benfisema subcut[aâ]neo\b', 'subcutaneous emphysema'),
    (r'\bdreno de t[oó]rax\b|\bdreno tor[aá]cico\b|\btubo tor[aá]cico\b', 'chest tube'),
    (r'\btubo endotraqueal\b|\btubo orotraqueal\b|\btot\b', 'endotracheal tube'),
    (r'\bcateter venoso central\b|\bcvc\b|\bcateter\b', 'central venous catheter'),
    (r'\bmarca-passo\b|\bmarcapasso\b', 'pacemaker'),
    (r'\bn[oó]dulo pulmonar\b|\bn[oó]dulo\b|\bn[oó]dulos\b', 'pulmonary nodule'),
    (r'\bmassa pulmonar\b|\bmassa\b', 'pulmonary mass'),
    (r'\bcavita[cç][aã]o\b', 'cavitation'),
    (r'\bcongest[aã]o pulmonar\b', 'pulmonary congestion'),
]

# Dicionário de Refinamento Terminológico: Inglês Traduzido -> Português Radiológico (Padrão CBR)
EN_TO_PT_REFINEMENTS = [
    # Derrame pleural / pleura
    (r'\befus[aã]o pleural\b', 'derrame pleural'),
    (r'\befus[oõ]es pleurais\b', 'derrames pleurais'),
    (r'\bderrame pleurales\b', 'derrames pleurais'),
    (r'\bderrame pleural bilaterais\b', 'derrames pleurais bilaterais'),
    (r'\bderrame pleural bilateral\b', 'derrame pleural bilateral'),
    (r'\bsem efus[aã]o pleural\b', 'ausência de derrame pleural'),
    (r'\bsem derrames pleurais\b', 'ausência de derrame pleural'),

    # Pneumotórax
    (r'\bun pneumothorax\b|\bpneumothorax\b', 'pneumotórax'),
    (r'\bapical sinistro\b', 'apical esquerdo'),
    (r'\bapical dextro\b', 'apical direito'),
    (r'\bn[aã]o h[aá] pneumot[oó]rax grande\b', 'ausência de pneumotórax volumoso'),

    # Coração e mediastino
    (r'\bsilhueta cardiomediastinal\b|\bcontorno cardiomediastinal\b', 'silhueta cardiomediastínica'),
    (r'\b[eé] normal\b|\bnormal\b', 'dentro dos limites da normalidade'),
    (r'\bsem cardiomegalia detectada\b', 'área cardíaca dentro dos limites normais'),
    (r'\bcardiomegalia est[aá]vel\b', 'aumento da área cardíaca de dimensões estáveis'),

    # Pulmões e campos pleuropulmonares
    (r'\bos pulm[oõ]es est[aã]o limpos\b|\bos pulm[oõ]es est[aã]o claros\b|\bpulm[oõ]es claros\b', 'campos pleuropulmonares livres de consolidações focais evidentes'),
    (r'\bsem processo cardiopulmonar agudo\b|\bn[aã]o h[aá] processo cardiopulmonar agudo\b', 'ausência de processo pleuropulmonar agudo evidente'),
    (r'\bopacidades bibasilares\b', 'opacidades bibasais'),
    (r'\bopacidades perihilares\b', 'opacidades perihilares'),
    (r'\bconsolida[cç][aã]o do lobo inferior direito\b', 'consolidação em lobo inferior direito (LID)'),
    (r'\bconsolida[cç][aã]o do lobo inferior esquerdo\b', 'consolidação em lobo inferior esquerdo (LIE)'),
    (r'\bconsolida[cç][aã]o do lobo superior direito\b', 'consolidação em lobo superior direito (LSD)'),
    (r'\bconsolida[cç][aã]o do lobo superior esquerdo\b', 'consolidação em lobo superior esquerdo (LSE)'),
    (r'\bedema pulmonar aumentado\b', 'sinais de congestão/edema pulmonar acentuado'),

    # Seios e diafragma
    (r'\b[aâ]ngulos costofr[eê]nicos\b|\bsulcos costofr[eê]nicos\b', 'seios costofrênicos'),
    (r'\bseios costofr[eê]nicos est[aã]o claros\b|\bseios costofr[eê]nicos livres\b', 'seios costofrênicos livres de derrames'),
    (r'\bseios costofr[eê]nicos exclu[ií]dos do estudo\b', 'seios costofrênicos não completamente incluídos na incidência'),

    # Dispositivos e suporte
    (r'\bponta do tubo endotraqueal fica a (\d+) cm acima da carina\b', r'tubo orotraqueal com extremidade distal localizada a \1 cm acima da carina'),
    (r'\bponta do cateter venoso central\b', 'extremidade do cateter venoso central'),
]


class CBRTranslator:
    """
    Tradutor e adaptador terminológico radiológico Português <-> Inglês.
    Utiliza o modelo neural NLLB-200 (600M) da Meta AI executado localmente.
    """
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load(self):
        """Carrega os pesos do modelo NLLB na memória GPU/CPU."""
        if not self._loaded:
            print(f"[CBRTranslator] Carregando {self.model_name} no dispositivo {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device).eval()
            self._loaded = True
            print("[CBRTranslator] Modelo de tradução carregado com sucesso!")

    def pt_to_en(self, text_pt: str) -> str:
        """
        Traduz texto em português (pergunta do usuário) para inglês radiológico.
        Aplica pré-substituição de jargão radiológico e tradução neural.
        """
        if not text_pt or not text_pt.strip():
            return ""

        text_clean = text_pt.strip()
        
        # 1. Pré-processamento com dicionário de radiologia torácica
        preprocessed = text_clean
        for pattern, english_term in PT_TO_EN_LEXICON:
            preprocessed = re.sub(pattern, english_term, preprocessed, flags=re.IGNORECASE)

        # Se já estiver traduzido ou for simples, podemos usar NLLB para o restante da oração
        if not self._loaded:
            self.load()

        inputs = self.tokenizer(preprocessed, return_tensors="pt").to(self.device)
        forced_id = self.tokenizer.convert_tokens_to_ids("eng_Latn")
        
        with torch.no_grad():
            tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_id,
                max_new_tokens=256
            )
        translated_en = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()

        # Garantir termos essenciais em inglês se estavam presentes no original
        for pattern, english_term in PT_TO_EN_LEXICON:
            if re.search(pattern, text_clean, re.IGNORECASE) and english_term.lower() not in translated_en.lower():
                # reforça o termo se o NLLB porventura traduziu de forma literal
                translated_en = f"{translated_en} (Focus: {english_term})"

        return translated_en

    def en_to_pt(self, text_en: str) -> str:
        """
        Traduz texto em inglês gerado pelo CheXagent para português formal radiológico (Padrão CBR).
        """
        if not text_en or not text_en.strip():
            return ""

        text_clean = text_en.strip()

        # Respostas curtas comuns do CheXagent
        lower_en = text_clean.lower()
        if lower_en in ["yes", "yes."]:
            return "Sim, alteração identificada."
        elif lower_en in ["no", "no."]:
            return "Não, sem evidência desta alteração."
        elif "(a) pa" in lower_en:
            return "Incidência Posteroanterior (PA)"
        elif "(b) ap" in lower_en:
            return "Incidência Anteroposterior (AP)"
        elif "(c) lateral" in lower_en:
            return "Incidência em Perfil (Lateral)"

        if not self._loaded:
            self.load()

        inputs = self.tokenizer(text_clean, return_tensors="pt").to(self.device)
        forced_id = self.tokenizer.convert_tokens_to_ids("por_Latn")

        with torch.no_grad():
            tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_id,
                max_new_tokens=512
            )
        translated_pt = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()

        # 2. Pós-processamento e refinamento terminológico para padrão radiológico brasileiro
        refined_pt = translated_pt
        for pattern, replacement in EN_TO_PT_REFINEMENTS:
            refined_pt = re.sub(pattern, replacement, refined_pt, flags=re.IGNORECASE)

        # Ajuste ortográfico de pontuação e espaçamento
        refined_pt = re.sub(r'\s+', ' ', refined_pt)
        refined_pt = re.sub(r'\s+([,.:;?!])', r'\1', refined_pt)

        return refined_pt
