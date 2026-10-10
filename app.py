"""
brGradio2CheXagent

Interface web em Gradio integrada ao modelo CheXagent-2-3b (Stanford AIMI)
com tradução radiológica para Português (Padrão CBR), rastreio sistemático de
patologias torácicas, laudo estruturado e localização espacial (phrase grounding).
"""

import os
import gradio as gr
from PIL import Image
from chexagent_service import CheXagentEngine

engine = CheXagentEngine()

CUSTOM_CSS = """
.container {
    max-width: 1440px;
    margin: 0 auto;
    padding: 10px;
}

/* Alinhamento estrito das colunas de resultados */
.results-row {
    display: flex !important;
    align-items: flex-start !important;
    gap: 20px !important;
    margin-top: 15px !important;
}

.col-screening, .col-report {
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
}

.card {
    border: 1px solid var(--border-color-primary, #e2e8f0) !important;
    border-radius: 10px !important;
    background: var(--background-fill-primary, #ffffff) !important;
    padding: 16px 20px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}

:root.dark .card, .dark .card {
    border-color: #334155 !important;
    background: #1e293b !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3) !important;
}

.card-title {
    margin: 0 0 12px 0 !important;
    padding: 0 0 10px 0 !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    color: var(--body-text-color, #0f172a) !important;
    border-bottom: 1px solid var(--border-color-primary, #e2e8f0) !important;
}

:root.dark .card-title, .dark .card-title {
    border-bottom-color: #334155 !important;
    color: #f1f5f9 !important;
}

#grounding-img img {
    cursor: zoom-in !important;
    transition: transform 0.2s ease;
}

#grounding-img img:hover {
    filter: brightness(1.03);
}
"""

FULLSCREEN_MODAL_HTML = """
<div id="image-modal" style="display:none; position:fixed; z-index:999999; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.92); justify-content:center; align-items:center; cursor:zoom-out;">
  <img id="modal-img" style="max-width:95vw; max-height:95vh; object-fit:contain; border-radius:4px; box-shadow:0 0 30px rgba(0,0,0,0.9);" />
  <div style="position:absolute; top:20px; right:30px; color:#fff; font-size:36px; font-weight:bold; cursor:pointer;" onclick="document.getElementById('image-modal').style.display='none'">&times;</div>
  <div style="position:absolute; bottom:20px; color:#cbd5e1; font-size:14px; background:rgba(0,0,0,0.65); padding:6px 16px; border-radius:20px; pointer-events:none;">Clique em qualquer lugar ou pressione Esc para fechar a tela cheia</div>
</div>
"""

MODAL_HEAD_JS = """
<script>
document.addEventListener('click', function(e) {
    var modal = document.getElementById('image-modal');
    if (modal && modal.style.display === 'flex') {
        modal.style.display = 'none';
        return;
    }
    var imgContainer = e.target.closest('#grounding-img');
    if (imgContainer) {
        var img = imgContainer.querySelector('img');
        if (img && img.src) {
            var modalImg = document.getElementById('modal-img');
            modalImg.src = img.src;
            modal.style.display = 'flex';
        }
    }
});

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        var modal = document.getElementById('image-modal');
        if (modal) modal.style.display = 'none';
    }
});
</script>
"""


def full_analysis_pipeline(img1, img2, query):
    if img1 is None and img2 is None:
        gr.Warning("Por favor, selecione ao menos uma radiografia antes de clicar em analisar.")
        yield (
            gr.update(visible=False),
            "<p>Selecione uma radiografia.</p>",
            None,
            "",
            "",
            gr.update(visible=False, value="")
        )
        return

    has_query = bool(query and query.strip())

    yield (
        gr.update(visible=True),
        "<div style='padding: 10px; opacity: 0.8;'>Iniciando rastreio sistemático e análise anatômica...</div>",
        img1 if img1 is not None else img2,
        "*Processando demarcação espacial...*",
        "*Gerando laudo estruturado passo a passo...*",
        gr.update(visible=has_query, value=f"**Pergunta:** *{query}*\n\n*Analisando...*" if has_query else "")
    )

    generator = engine.analyze_everything(img1, img2, query)
    for step in generator:
        yield (
            gr.update(visible=True),
            step["screening_html"],
            step["annotated_image"],
            step["grounding_text"],
            step["structured_report"],
            gr.update(visible=has_query, value=step["clinical_answer"])
        )


def reset_all():
    return (
        None,                       # img_primary
        None,                       # img_lateral
        "",                         # user_indication
        gr.update(visible=False),   # results_container
        "",                         # out_screening
        None,                       # out_annotated_image
        "",                         # out_grounding_info
        "",                         # out_report
        gr.update(visible=False, value=""), # out_clinical_answer
        [],                         # chatbot_history
        ""                          # chat_msg
    )


def chat_followup(question, history, img_primary, img_lateral):
    if not question or not question.strip():
        return history, ""
    
    current_img = img_primary if img_primary is not None else img_lateral
    if current_img is None:
        answer = "Nenhuma radiografia carregada para responder a pergunta."
    else:
        answer = engine.answer_quick_question(current_img, question)

    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return history, ""


with gr.Blocks(title="brGradio2CheXagent") as demo:
    gr.HTML(FULLSCREEN_MODAL_HTML)

    with gr.Column(elem_classes=["container"]):
        gr.Markdown("# brGradio2CheXagent")

        # Entrada de imagens e parâmetros clínicos
        with gr.Column():
            with gr.Row():
                img_primary = gr.Image(
                    label="Incidência Posteroanterior ou Anteroposterior (PA / AP)",
                    type="pil",
                    image_mode="RGB"
                )
                img_lateral = gr.Image(
                    label="Incidência em Perfil / Lateral",
                    type="pil",
                    image_mode="RGB"
                )

            user_indication = gr.Textbox(
                label="Indicação Clínica ou Pergunta Específica (Opcional)",
                placeholder="Ex: Paciente com dispneia e dor torácica súbita à esquerda. Há pneumotórax ou derrame pleural?",
                lines=2
            )

            with gr.Row():
                btn_run_all = gr.Button("Analisar Radiografia", variant="primary", scale=4, size="lg")
                btn_clear = gr.Button("Limpar", variant="secondary", scale=1, size="lg")

        # Painel de resultados
        with gr.Column(visible=False) as results_container:
            gr.Markdown("---")
            gr.Markdown("## Resultados da Avaliação Completa")

            with gr.Row(equal_height=False, elem_classes=["results-row"]):
                # Coluna Esquerda: Painel de Rastreio (scale=5)
                with gr.Column(scale=5, elem_classes=["col-screening", "card"]):
                    gr.HTML('<div class="card-title">Painel de Rastreio de Patologias</div>')
                    out_screening = gr.HTML(value="")

                # Coluna Direita: Laudo Estruturado e Localização Visual (scale=6)
                with gr.Column(scale=6, elem_classes=["col-report"]):
                    # Card 1: Laudo Estruturado
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML('<div class="card-title">Laudo Radiológico Estruturado</div>')
                        out_clinical_answer = gr.Markdown(value="", visible=False)
                        out_report = gr.Markdown(value="")

                    # Card 2: Localização Visual de Lesões (posicionada abaixo do laudo)
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML('<div class="card-title">Localização Visual de Lesões <span style="font-size:12px; font-weight:normal; opacity:0.75; margin-left:8px;">(Clique na imagem para expandir em tela cheia)</span></div>')
                        out_annotated_image = gr.Image(
                            type="pil",
                            height=380,
                            show_label=False,
                            elem_id="grounding-img"
                        )
                        out_grounding_info = gr.Markdown(value="")

            # Chat para esclarecimento de dúvidas adicionais
            with gr.Column(elem_classes=["card"]):
                gr.HTML('<div class="card-title">Perguntas e Respostas Adicionais sobre este Exame</div>')
                chatbot_history = gr.Chatbot(label="Diálogo Clínico", height=240, show_label=False)
                with gr.Row():
                    chat_msg = gr.Textbox(
                        placeholder="Digite uma dúvida adicional em português sobre a radiografia analisada...",
                        lines=1,
                        scale=9,
                        show_label=False
                    )
                    chat_send = gr.Button("Perguntar", variant="primary", scale=2)

    # Eventos
    btn_run_all.click(
        fn=full_analysis_pipeline,
        inputs=[img_primary, img_lateral, user_indication],
        outputs=[
            results_container,
            out_screening,
            out_annotated_image,
            out_grounding_info,
            out_report,
            out_clinical_answer
        ]
    )

    btn_clear.click(
        fn=reset_all,
        inputs=[],
        outputs=[
            img_primary,
            img_lateral,
            user_indication,
            results_container,
            out_screening,
            out_annotated_image,
            out_grounding_info,
            out_report,
            out_clinical_answer,
            chatbot_history,
            chat_msg
        ]
    )

    chat_send.click(
        fn=chat_followup,
        inputs=[chat_msg, chatbot_history, img_primary, img_lateral],
        outputs=[chatbot_history, chat_msg]
    )
    chat_msg.submit(
        fn=chat_followup,
        inputs=[chat_msg, chatbot_history, img_primary, img_lateral],
        outputs=[chatbot_history, chat_msg]
    )


if __name__ == "__main__":
    engine.load_model()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CUSTOM_CSS,
        head=MODAL_HEAD_JS,
        share=False
    )
