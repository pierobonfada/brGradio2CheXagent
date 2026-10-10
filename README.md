# brGradio2CheXagent

Sistema local para avaliação de radiografias de tórax baseado no modelo de visão-linguagem **CheXagent-2-3b** (Stanford AIMI), com interface web em Gradio e ponte neural de tradução/adaptação terminológica para a língua portuguesa segundo as diretrizes do **Colégio Brasileiro de Radiologia (CBR)**.

---

## Principais Funcionalidades

- **Rastreio Sistemático de 21 Patologias:** Rastreamento simultâneo de condições pleuropulmonares, cardiovasculares, diafragmáticas, ósseas e dispositivos de suporte (pneumotórax, derrame pleural, pneumonia, cardiomegalia, fraturas, drenos, tubos, etc.). As alterações detectadas são destacadas diretamente em vermelho vivo (`#ef4444`).
- **Laudo Estruturado (Padrão CBR):** Descrição anatômica compartimentada em Vias Aéreas, Campos Pleuropulmonares, Silhueta Cardíaca e Mediastino, Cúpulas Diafragmáticas / Seios Costofrênicos e Arcabouço Ósseo / Partes Moles.
- **Localização Visual de Lesões (Phrase Grounding):** Demarcação espacial precisa por retângulos delimitadores (bounding boxes) estritamente correlacionados com os achados descritos no laudo.
- **Visualização em Tela Cheia (Lightbox):** Expansão modal com um clique sobre a imagem demarcada para inspeção visual detalhada de alta resolução.
- **Diálogo Clínico Integrado:** Chat interativo dedicado para perguntas em linguagem natural sobre a radiografia sob análise.
- **Execução 100% Local:** Processamento confidencial sem envio de dados a servidores externos, otimizado para placas de vídeo com aceleração CUDA.

---

## Estrutura do Projeto

```
brGradio2CheXagent/
├── app.py                 # Interface web Gradio com pipeline de análise e eventos
├── chexagent_service.py   # Motor de inferência CheXagent, rastreio e demarcação espacial
├── cbr_translator.py      # Tradução neural (Meta NLLB-200) e léxico radiológico CBR
├── run.sh                 # Script de inicialização do ambiente
├── requirements.txt       # Dependências do projeto
├── README.md              # Documentação técnica
└── .gitignore             # Arquivos ignorados pelo controle de versão
```

---

## Requisitos do Sistema

- **Sistema Operacional:** Linux (Ubuntu 20.04+ recomendado)
- **Python:** 3.10 ou superior
- **GPU:** NVIDIA com suporte a CUDA e no mínimo 12 GB de VRAM (ex: RTX 3060 12GB ou superior)
- **CUDA Toolkit:** 12.x

---

## Instalação e Execução

### 1. Clonar o Repositório

```bash
git clone git@github.com:pierobonfada/brGradio2CheXagent.git
cd brGradio2CheXagent
```

### 2. Configurar o Ambiente Virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Nota:** Certifique-se de que o PyTorch está instalado com suporte a CUDA para o seu driver (`torch>=2.0.0+cu124`).

### 3. Iniciar a Aplicação

```bash
./run.sh
```

Ou diretamente via Python:

```bash
source .venv/bin/activate
python app.py
```

A interface estará disponível em:
👉 **`http://localhost:7860`**

---

## Fluxo de Operação

1. **Seleção das Incidências:** Carregue a projeção Posteroanterior / Anteroposterior (PA / AP) e, opcionalmente, a incidência em Perfil / Lateral.
2. **Indicação Clínica (Opcional):** Insira informações de história clínica ou perguntas direcionadas (ex: *"Paciente com dor pleurítica súbita à esquerda. Há pneumotórax?"*).
3. **Analisar Radiografia:** Clique no botão principal para iniciar o rastreio sistemático, a geração do laudo e a demarcação das lesões.
4. **Inspeção de Resultados:**
   - Visualize a tabela de rastreio de patologias à esquerda.
   - Analise o laudo estruturado e a imagem demarcada à direita (clique na radiografia para tela cheia).
   - Tire dúvidas adicionais no diálogo clínico ao final da página.
5. **Limpar:** Reseta todas as entradas e resultados para iniciar uma nova avaliação a partir do zero.

---

## Licença e Créditos

- Modelo base: **CheXagent-2-3b** desenvolvido por **Stanford AIMI** ([Stanford Center for Artificial Intelligence in Medicine & Imaging](https://github.com/Stanford-AIMI/CheXagent)).
- Tradução neural e suporte multilíngue: **Meta NLLB-200** ([No Language Left Behind](https://github.com/facebookresearch/fairseq/tree/nllb)).
- Adaptação terminológica: Diretrizes de laudos estruturados do **Colégio Brasileiro de Radiologia e Diagnóstico por Imagem (CBR)**.
