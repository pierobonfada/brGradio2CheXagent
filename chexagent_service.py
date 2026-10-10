"""
Serviço de Inferência CheXagent-2-3b e Pipeline Radiológico Estruturado

Integração local do modelo Stanford AIMI CheXagent-2-3b com:
- Rastreio simultâneo de 21 patologias e condições torácicas.
- Elaboração de laudo radiológico estruturado segundo a terminologia do CBR.
- Localização visual de lesões (phrase grounding) correlacionada aos achados descritos.
- Resposta a questionamentos clínicos em linguagem natural.
"""

import os
import re
import tempfile
import torch
from typing import List, Dict, Tuple, Optional, Generator
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoTokenizer

from cbr_translator import CBRTranslator

# Catálogo de Patologias Torácicas para Rastreio
CONDITIONS_LIST = [
    ("Pneumotórax", "Pneumothorax"),
    ("Derrame Pleural / Borramento de Seio Costofrênico", "Pleural Effusion or blunted costophrenic angle"),
    ("Consolidação / Pneumonia", "Pneumonia or consolidation"),
    ("Atelectasia", "Atelectasis"),
    ("Edema Pulmonar", "Pulmonary Edema"),
    ("Nódulo Pulmonar", "Pulmonary Nodule"),
    ("Massa Pulmonar", "Lung Mass"),
    ("Infiltrado Intersticial", "Interstitial Infiltrate or reticular opacities"),
    ("Cavitação / Caverna", "Cavitation or cavitary lesion"),
    ("Enfisema / Hiperinsuflação", "Emphysema or hyperinflation"),
    ("Cardiomegalia", "Cardiomegaly"),
    ("Alargamento de Mediastino", "Mediastinal Widening or mediastinal enlargement"),
    ("Ectasia / Aneurisma de Aorta", "Aortic Ectasia or aortic aneurysm"),
    ("Congestão Hilar", "Hilar Congestion or prominent pulmonary vessels"),
    ("Fratura de Costela", "Rib Fracture"),
    ("Fratura de Clavícula", "Clavicle Fracture"),
    ("Enfisema Subcutâneo", "Subcutaneous Emphysema"),
    ("Hérnia Diafragmática", "Diaphragmatic Hernia"),
    ("Tubo Orotraqueal (TOT)", "Endotracheal Tube"),
    ("Dreno Torácico", "Chest Tube"),
    ("Cateter Venoso Central (CVC)", "Central Venous Catheter"),
    ("Marca-passo Cardíaco", "Pacemaker")
]

ABNORMAL_KEYWORDS_PT = [
    r'\bborramento\b', r'\bvelamento\b', r'\bderrame\b', r'\befus[aã]o\b',
    r'\bpneumot[oó]rax\b', r'\bconsolida[cç][aã]o\b', r'\bopacidade\b',
    r'\binfiltrado\b', r'\bcardiomegalia\b', r'\baumentad[oa]\b',
    r'\bfratura\b', r'\batelectasia\b', r'\bedema\b', r'\bn[oó]dulo\b',
    r'\bmassa\b', r'\bcavita[cç][aã]o\b', r'\benfisema\b',
    r'\bh[eé]rnia\b', r'\bdreno\b', r'\btubo\b', r'\bcateter\b', r'\bectasia\b'
]

NORMAL_KEYWORDS_PT = [
    r'dentro dos limites da normalidade', r'livres de derrames', r'livre de derrames',
    r'livres de consolida[cç][oõ]es', r'livre de consolida[cç][aã]o',
    r'sem altera[cç][oõ]es pleuropulmonares', r'sem evid[eê]ncia de altera[cç][aã]o',
    r'aus[eê]ncia de processo', r'normal'
]


def check_is_abnormal(text_en: str, text_pt: str) -> bool:
    """Identifica se uma seção do laudo descreve alteração patológica real."""
    en_lower = text_en.lower()
    pt_lower = text_pt.lower()

    if pt_lower.startswith("ausência de") or pt_lower.startswith("sem "):
        if not any(k in pt_lower for k in ["mas ", "porém ", "contudo ", "exceto "]):
            return False

    is_declared_normal = any(re.search(norm, pt_lower) for norm in NORMAL_KEYWORDS_PT)
    if is_declared_normal and not any(k in pt_lower for k in ["mas ", "porém ", "contudo ", "exceto ", "borramento", "derrame", "pneumotórax"]):
        return False

    has_abnormal_pt = any(re.search(pat, pt_lower) for pat in ABNORMAL_KEYWORDS_PT)
    has_abnormal_en = any(w in en_lower for w in [
        "blunt", "effusion", "pneumothorax", "consolidation", "opacity", "opacities",
        "infiltrat", "cardiomegaly", "fracture", "atelectasis", "edema", "nodule", "mass",
        "tube", "drain", "catheter", "congestion", "cavitation", "emphysema", "hernia"
    ])

    if not (has_abnormal_pt or has_abnormal_en):
        return False

    negations_en = [
        "no pneumothorax", "no pleural effusion", "no cardiomegaly", "no focal consolidation",
        "no active disease", "no acute cardiopulmonary", "no fractures", "no blunting", "clear"
    ]
    if any(neg in en_lower for neg in negations_en) and not any(w in en_lower for w in ["is present", "are present", "there is", "noted", "detected"]):
        return False

    return True


class CheXagentEngine:
    def __init__(self, model_id: str = "StanfordAIMI/CheXagent-2-3b", device: str = "cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.tokenizer = None
        self.model = None
        self.translator = CBRTranslator(device=self.device)
        self._is_loaded = False

    def load_model(self):
        if self._is_loaded:
            return

        print(f"[CheXagent] Carregando {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True
        ).eval()

        if hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
            if hasattr(self.model.model.visual, "pos_embed"):
                self.model.model.visual.pos_embed.data = self.model.model.visual.pos_embed.data.to(self.dtype)

        self.translator.load()
        self._is_loaded = True
        print("[CheXagent] Modelos carregados com sucesso.")

    def _save_temp_images(self, images: List[Image.Image]) -> List[str]:
        paths = []
        for img in images:
            if img is None:
                continue
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                temp_path = tf.name
            img.convert("RGB").save(temp_path)
            paths.append(temp_path)
        return paths

    def _cleanup_temp_images(self, paths: List[str]):
        for p in paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    def generate_single_prompt(self, image_paths: List[str], prompt: str, max_new_tokens: int = 256) -> str:
        if not self._is_loaded:
            self.load_model()

        query_items = [{'image': p} for p in image_paths]
        query_items.append({'text': prompt})
        query = self.tokenizer.from_list_format(query_items)
        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query}
        ]
        input_ids = self.tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                max_new_tokens=max_new_tokens
            )[0]
        
        return self.tokenizer.decode(output[input_ids.size(1):-1]).strip().replace("</s>", "")

    def draw_bounding_boxes(self, image: Image.Image, response_en: str, label_pt: str) -> Tuple[Image.Image, bool]:
        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)
        w, h = annotated.size

        parsed_list = self.tokenizer.to_list_format(response_en)
        boxes = [_d["box"] for _d in parsed_list if "box" in _d]

        found = False
        for box_str in boxes:
            try:
                coords = [int(c.strip()) / 100.0 for c in box_str.replace("(", "").replace(")", "").split(",")]
                if len(coords) == 4:
                    x1 = coords[0] * w
                    y1 = coords[1] * h
                    x2 = coords[2] * w
                    y2 = coords[3] * h
                    draw.rectangle([(x1, y1), (x2, y2)], outline="#ef4444", width=4)
                    draw.rectangle([(x1, max(0, y1 - 22)), (x1 + 220, y1)], fill="#ef4444")
                    draw.text((x1 + 6, max(0, y1 - 18)), label_pt[:28], fill="white")
                    found = True
            except Exception:
                pass
        return annotated, found

    def build_screening_html(self, screening_data: List[Tuple[str, str, str]]) -> str:
        """
        Gera tabela HTML limpa e alinhada.
        Linhas com achado PRESENTE têm texto em vermelho vivo (#ef4444) sem fundo rosa.
        """
        html = """
        <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; font-family: inherit; font-size: 13px; line-height: 1.4;">
          <thead>
            <tr style="border-bottom: 2px solid var(--border-color-primary, #475569); text-align: left;">
              <th style="padding: 8px 10px; font-weight: 700;">Patologia Torácica</th>
              <th style="padding: 8px 10px; text-align: center; font-weight: 700; width: 110px;">Status</th>
              <th style="padding: 8px 10px; font-weight: 700;">Avaliação do Modelo</th>
            </tr>
          </thead>
          <tbody>
        """
        for patologia, status, avaliacao in screening_data:
            if status == "PRESENTE":
                html += f"""
                <tr style="border-bottom: 1px solid rgba(239, 68, 68, 0.35);">
                  <td style="padding: 7px 10px; color: #ef4444 !important; font-weight: 700;">{patologia}</td>
                  <td style="padding: 7px 10px; text-align: center; color: #ef4444 !important; font-weight: 800;">PRESENTE</td>
                  <td style="padding: 7px 10px; color: #ef4444 !important; font-weight: 700;">{avaliacao}</td>
                </tr>
                """
            else:
                html += f"""
                <tr style="border-bottom: 1px solid var(--border-color-primary, #334155);">
                  <td style="padding: 7px 10px;">{patologia}</td>
                  <td style="padding: 7px 10px; text-align: center; color: #22c55e !important; font-weight: 600;">AUSENTE</td>
                  <td style="padding: 7px 10px; opacity: 0.82;">{avaliacao}</td>
                </tr>
                """
        html += "</tbody></table></div>"
        return html

    def analyze_everything(
        self,
        img1: Optional[Image.Image],
        img2: Optional[Image.Image],
        user_query: str = ""
    ) -> Generator[Dict, None, None]:
        """
        Executa pipeline radiológico integrado:
        1. Rastreio sistemático de patologias e condições torácicas.
        2. Laudo radiológico estruturado segundo diretrizes CBR.
        3. Localização espacial (phrase grounding) dos achados descritos.
        """
        if not self._is_loaded:
            self.load_model()

        images = [img for img in [img1, img2] if img is not None]
        if not images:
            yield {
                "screening_html": "<p>Nenhuma radiografia enviada.</p>",
                "annotated_image": None,
                "grounding_text": "",
                "structured_report": "Por favor, envie uma radiografia de tórax.",
                "clinical_answer": ""
            }
            return

        temp_paths = self._save_temp_images(images)
        main_img = images[0]

        try:
            # Estado inicial com indicador de carregamento
            yield {
                "screening_html": "<div style='padding: 12px; color: var(--body-text-color-subdued, #94a3b8);'>Processando rastreio sistemático de 21 patologias torácicas...</div>",
                "annotated_image": main_img,
                "grounding_text": "*Processando localização visual...*",
                "structured_report": "*Gerando laudo estruturado passo a passo...*",
                "clinical_answer": f"**Pergunta:** *{user_query}*\n\n*Processando resposta...*" if user_query else ""
            }

            # 1. GERAÇÃO DAS SEÇÕES DO LAUDO ESTRUTURADO (PADRÃO CBR)
            view_raw = self.generate_single_prompt(temp_paths, "What is the view of this chest X-ray? Options: (a) PA, (b) AP, (c) LATERAL", max_new_tokens=32)
            view_pt = self.translator.en_to_pt(view_raw)

            sections_cfg = [
                ("Vias Aéreas", 'Please provide a detailed description of "Airway" in the chest X-ray'),
                ("Campos Pleuropulmonares", 'Please provide a detailed description of "Breathing" in the chest X-ray'),
                ("Silhueta Cardíaca e Mediastino", 'Please provide a detailed description of "Cardiac" in the chest X-ray'),
                ("Cúpulas Diafragmáticas e Seios Costofrênicos", 'Please provide a detailed description of "Diaphragm" in the chest X-ray'),
                ("Arcabouço Ósseo e Partes Moles", 'Please provide a detailed description of "Everything else (e.g., mediastinal contours, bones, soft tissues, tubes, valves, and pacemakers)" in the chest X-ray'),
            ]

            findings_pt_formatted = []
            findings_en_raw = {}
            findings_pt_raw = {}

            for title_pt, prompt_en in sections_cfg:
                resp_en = self.generate_single_prompt(temp_paths, prompt_en, max_new_tokens=180)
                resp_pt = self.translator.en_to_pt(resp_en)
                
                findings_en_raw[title_pt] = resp_en
                findings_pt_raw[title_pt] = resp_pt

                is_altered = check_is_abnormal(resp_en, resp_pt)
                if is_altered:
                    findings_pt_formatted.append(f"• **{title_pt}:** <span style=\"color: #ef4444; font-weight: 600;\">{resp_pt}</span>")
                else:
                    findings_pt_formatted.append(f"• **{title_pt}:** {resp_pt}")

            full_findings_en = " ".join(findings_en_raw.values())
            impression_prompt = f"Write the Impression section for the following Findings: {full_findings_en}"
            query_items = [{'text': impression_prompt}]
            conv = [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": self.tokenizer.from_list_format(query_items)}]
            input_ids = self.tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(input_ids, do_sample=False, num_beams=1, max_new_tokens=200)[0]
            impression_en = self.tokenizer.decode(out[input_ids.size(1):-1]).strip().replace("</s>", "")
            impression_pt = self.translator.en_to_pt(impression_en)

            is_impression_altered = check_is_abnormal(impression_en, impression_pt)
            if is_impression_altered:
                impression_display = f"<span style=\"color: #ef4444; font-weight: 700;\">{impression_pt}</span>"
            else:
                impression_display = f"**{impression_pt}**"

            indication_display = user_query.strip() if user_query else "Não informada / Avaliação geral"

            structured_report = f"""**Indicação Clínica:** {indication_display}  
**Incidência:** {view_pt}

#### Achados por Sistemas:
{chr(10).join(findings_pt_formatted)}

#### Conclusão Diagnóstica:
{impression_display}
"""

            # 2. RASTREIO DAS PATOLOGIAS TORÁCICAS (ALINHADO COM OS ACHADOS)
            screening_data = []
            for pt_name, en_name in CONDITIONS_LIST:
                resp_en = self.generate_single_prompt(temp_paths, f"Does this chest X-ray contain a {en_name}?", max_new_tokens=16)
                resp_pt = self.translator.en_to_pt(resp_en)
                
                is_pos = "sim" in resp_pt.lower() or "yes" in resp_en.lower()
                
                # Neutraliza falso positivo espúrio de ectasia aórtica se o coração/mediastino for normal
                if "Ectasia" in pt_name and not check_is_abnormal(findings_en_raw.get("Silhueta Cardíaca e Mediastino", ""), findings_pt_raw.get("Silhueta Cardíaca e Mediastino", "")):
                    is_pos = False
                    resp_pt = "Sem evidência de ectasia ou aneurisma no mediastino."

                # Se o laudo acusou borramento no diafragma, confirma derrame pleural
                if "Derrame Pleural" in pt_name and check_is_abnormal(findings_en_raw.get("Cúpulas Diafragmáticas e Seios Costofrênicos", ""), findings_pt_raw.get("Cúpulas Diafragmáticas e Seios Costofrênicos", "")):
                    is_pos = True
                    resp_pt = "Identificado borramento de seio costofrênico compatível com derrame pleural."

                status_text = "PRESENTE" if is_pos else "AUSENTE"
                screening_data.append((pt_name, status_text, resp_pt))

            screening_html = self.build_screening_html(screening_data)

            # 3. LOCALIZAÇÃO VISUAL (PHRASE GROUNDING) RIGOROSA E ALINHADA AO LAUDO
            targets_to_locate = []

            # Se o usuário perguntou sobre algo específico, prioriza
            if user_query:
                q_low = user_query.lower()
                if any(w in q_low for w in ["seio", "costofr", "borramento", "velamento", "derrame", "efus"]):
                    targets_to_locate.append(("blunted costophrenic angle or pleural effusion", "Borramento do seio costofrênico"))
                elif "pneumo" in q_low:
                    targets_to_locate.append(("pneumothorax", "Pneumotórax"))
                elif "fratura" in q_low or "costela" in q_low:
                    targets_to_locate.append(("rib fracture", "Fratura costal"))
                elif "dreno" in q_low or "tubo" in q_low:
                    targets_to_locate.append(("chest drain or chest tube", "Dreno de tórax"))

            # Checa os achados reais do laudo
            diaph_en = findings_en_raw.get("Cúpulas Diafragmáticas e Seios Costofrênicos", "")
            diaph_pt = findings_pt_raw.get("Cúpulas Diafragmáticas e Seios Costofrênicos", "")
            if check_is_abnormal(diaph_en, diaph_pt):
                if any(w in diaph_en.lower() for w in ["blunt", "effusion", "fluid", "angle", "sulcus"]) or any(w in diaph_pt.lower() for w in ["borramento", "velamento", "derrame"]):
                    targets_to_locate.append(("blunted costophrenic angle", "Borramento do seio costofrênico"))

            breath_en = findings_en_raw.get("Campos Pleuropulmonares", "")
            breath_pt = findings_pt_raw.get("Campos Pleuropulmonares", "")
            if check_is_abnormal(breath_en, breath_pt):
                if "pneumothorax" in breath_en.lower() or "pneumotórax" in breath_pt.lower():
                    targets_to_locate.append(("pneumothorax", "Pneumotórax"))
                if any(w in breath_en.lower() for w in ["consolidation", "pneumonia", "opacity", "opacities"]):
                    targets_to_locate.append(("consolidation or opacity", "Consolidação / Opacidade"))
                if any(w in breath_en.lower() for w in ["nodule", "mass"]):
                    targets_to_locate.append(("pulmonary nodule or mass", "Nódulo / Massa"))

            cardiac_en = findings_en_raw.get("Silhueta Cardíaca e Mediastino", "")
            cardiac_pt = findings_pt_raw.get("Silhueta Cardíaca e Mediastino", "")
            if check_is_abnormal(cardiac_en, cardiac_pt):
                if "cardiomegaly" in cardiac_en.lower() or "enlarged" in cardiac_en.lower():
                    targets_to_locate.append(("cardiomegaly", "Cardiomegalia"))

            other_en = findings_en_raw.get("Arcabouço Ósseo e Partes Moles", "")
            other_pt = findings_pt_raw.get("Arcabouço Ósseo e Partes Moles", "")
            if check_is_abnormal(other_en, other_pt):
                if "fracture" in other_en.lower():
                    targets_to_locate.append(("rib fracture", "Fratura costal"))
                if any(w in other_en.lower() for w in ["tube", "drain"]):
                    targets_to_locate.append(("chest drain or chest tube", "Dreno de tórax"))

            unique_targets = []
            seen = set()
            for t_en, t_pt in targets_to_locate:
                if t_pt not in seen:
                    seen.add(t_pt)
                    unique_targets.append((t_en, t_pt))

            annotated_img = main_img.copy().convert("RGB")
            located_labels = []

            for target_en, target_pt in unique_targets:
                grounding_prompt = f"Please locate the following phrase: {target_en}"
                grounding_raw = self.generate_single_prompt(temp_paths, grounding_prompt, max_new_tokens=100)
                annotated_img, found = self.draw_bounding_boxes(annotated_img, grounding_raw, target_pt)
                if found:
                    located_labels.append(target_pt)

            if located_labels:
                grounding_text = f"**Achados demarcados em vermelho:** {', '.join(located_labels)}"
            elif unique_targets:
                grounding_text = f"*Achado suspeito no laudo ({unique_targets[0][1]}), sem demarcação focal delimitada.*"
            else:
                grounding_text = "*Sem lesões focais agudas para demarcação.*"

            # 4. RESPOSTA CLÍNICA À PERGUNTA DO USUÁRIO
            clinical_answer = ""
            if user_query and user_query.strip():
                q_en = self.translator.pt_to_en(user_query)
                ans_en = self.generate_single_prompt(temp_paths, q_en, max_new_tokens=150)
                ans_pt = self.translator.en_to_pt(ans_en)
                clinical_answer = f"**Pergunta:** *{user_query}*\n\n**Resposta:** {ans_pt}"

            yield {
                "screening_html": screening_html,
                "annotated_image": annotated_img,
                "grounding_text": grounding_text,
                "structured_report": structured_report,
                "clinical_answer": clinical_answer
            }

        finally:
            self._cleanup_temp_images(temp_paths)

    def answer_quick_question(self, image: Optional[Image.Image], question_pt: str) -> str:
        if not self._is_loaded:
            self.load_model()
        if image is None or not question_pt.strip():
            return "Envie uma radiografia e digite uma pergunta."

        temp_paths = self._save_temp_images([image])
        try:
            q_en = self.translator.pt_to_en(question_pt)
            resp_en = self.generate_single_prompt(temp_paths, q_en, max_new_tokens=180)
            return self.translator.en_to_pt(resp_en)
        finally:
            self._cleanup_temp_images(temp_paths)
