# Notas de Melhorias (PT-BR)

Este documento resume as melhorias introduzidas pelos commits recentes de autoria local e explica como utilizá-las na prática.

Autor local identificado nos commits: Luís De Marchi

## 1) GUI de Streaming em Tempo Real e Loader
- Commits:
  - 08fce93 | feat: add realtime streaming GUI and loader
- Arquivos principais:
  - `app_vc_realtime.py`
  - `inference_realtime.py`
- O que foi melhorado:
  - Adiciona uma GUI (interface gráfica) para conversão de voz em tempo real, além de utilitários de carregamento (loader) necessários para streaming.
- Como usar:
  - GUI em tempo real:
    ```bash
    python real-time-gui.py --checkpoint-path <caminho-para-checkpoint> --config-path <caminho-para-config>
    ```
    - Ajuste os parâmetros conforme o desempenho do seu dispositivo (vide `README-PT.md`, seção de GUI em tempo real).
  - Streaming por script (sem GUI):
    ```bash
    python inference_realtime.py \
      --source <arquivo.wav> \
      --target <referencia.wav> \
      --output <dir_saida>
    ```

## 2) Script de Inferência ASTRAL e Requisitos de Gravação Opcionais
- Commits:
  - 6d0dc21 | feat: add ASTRAL inference script and optional recording requirements
- Arquivos principais:
  - `inference_astral.py`
  - `requirements-recording.txt`
- O que foi melhorado:
  - Adiciona um caminho de inferência baseado em ASTRAL (quantização de conteúdo linguístico) e um arquivo de requisitos opcional para recursos de gravação.
- Como usar:
  - (Opcional) Instalar dependências de gravação:
    ```bash
    pip install -r requirements-recording.txt
    ```
  - Rodar inferência com ASTRAL:
    ```bash
    python inference_astral.py \
      --source <arquivo.wav> \
      --target <referencia.wav> \
      --output <dir_saida>
    ```

## 3) Base Docker e Ferramentas para Serverless/Handler
- Commits:
  - 49dd43b | chore: add docker base image and serverless handler tooling
- Arquivos principais:
  - `Dockerfile.base`, `Dockerfile.handler`
  - `api-inference.sh`, `api-train.sh`, `entrypoint.sh`
  - `environment.yml`
  - `handler.py`
- O que foi melhorado:
  - Facilita a conteinerização (Docker) do projeto e provê scripts para rodar inferência/treino via API em ambientes serverless.
- Como usar (exemplos):
  - Construir imagem base:
    ```bash
    docker build -f Dockerfile.base -t seed-vc-base:latest .
    ```
  - Construir imagem com handler:
    ```bash
    docker build -f Dockerfile.handler -t seed-vc-handler:latest .
    ```
  - Rodar container (exemplo genérico):
    ```bash
    docker run --rm -p 7860:7860 seed-vc-handler:latest
    ```
  - Scripts de API:
    - `api-inference.sh`: endpoints de inferência;
    - `api-train.sh`: endpoints de treino.

## 4) Pyproject e CLI simplificada para Inferência
- Commits:
  - 568b03f | feat: add pyproject and run_inference CLI from seed-vc2
- Arquivos principais:
  - `pyproject.toml`
  - `run_inference.py`
- O que foi melhorado:
  - Organização do pacote/projeto via `pyproject.toml` e um CLI simplificado de inferência (`run_inference.py`).
- Como usar:
  - Instalar dependências padrão:
    ```bash
    pip install -r requirements.txt
    ```
  - Executar CLI de inferência simplificada:
    ```bash
    python run_inference.py \
      --source <arquivo.wav> \
      --target <referencia.wav> \
      --output <dir_saida>
    ```

## 5) Integração da API de Streaming V1 e Utilitários
- Commits:
  - a8bdfe5 | feat: integrate V1 streaming API and utilities
- Arquivos principais:
  - `Models/audio.py`
  - `api.py`
  - `__init__.py`, `_entry.py`, `_paths.py`
- O que foi melhorado:
  - Disponibiliza utilitários e uma API para streaming V1, facilitando integrações de conversão de voz em pipelines e serviços.
- Como usar:
  - Consulte `api.py` para endpoints e exemplos de uso programático;
  - Combine com os apps web (`app_vc.py`, `app_svc.py`, `app_vc_v2.py`) quando desejar expor via interface.

## Referências rápidas (do README-PT)
- Web UI (VC comum):
  ```bash
  python app_vc.py --checkpoint <ckpt> --config <cfg> --fp16 True
  ```
- Web UI (SVC/canto):
  ```bash
  python app_svc.py --checkpoint <ckpt> --config <cfg> --fp16 True
  ```
- Web UI (V2):
  ```bash
  python app_vc_v2.py --cfm-checkpoint-path <ckpt_cfm> --ar-checkpoint-path <ckpt_ar>
  ```
- CLI V1:
  ```bash
  python inference.py --source <src> --target <ref> --output <out> --diffusion-steps 25
  ```
- CLI V2:
  ```bash
  python inference_v2.py --source <src> --target <ref> --output <out> --convert-style true
  ```

## Observações finais
- Para tempo real, recomenda-se GPU. Ajuste parâmetros na GUI para manter "Tempo de Inferência por Bloco" menor que "Tempo de Bloco".
- Caso não consiga acessar o Hugging Face, use `HF_ENDPOINT=https://hf-mirror.com`.
- Veja `README-PT.md` para instruções detalhadas de instalação, uso e treinamento.
