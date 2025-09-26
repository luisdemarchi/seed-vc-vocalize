# Seed-VC
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)  [![arXiv](https://img.shields.io/badge/arXiv-2411.09943-%3CCOLOR%3E.svg)](https://arxiv.org/abs/2411.09943)

*[Português (Brasil)] | [English](README.md) | [简体中文](README-ZH.md) | [日本語](README-JA.md)*

[real-time-demo.webm](https://github.com/user-attachments/assets/86325c5e-f7f6-4a04-8695-97275a5d046c)

O modelo atualmente lançado suporta *conversão de voz zero-shot* 🔊, *conversão de voz em tempo real zero-shot* 🗣️ e *conversão de voz de canto zero-shot* 🎶. Sem qualquer treinamento, ele é capaz de clonar uma voz a partir de uma amostra de áudio de referência de 1 a 30 segundos.

Oferecemos suporte para fine-tuning (ajuste fino) com dados personalizados para melhorar o desempenho em um ou mais locutores específicos, com requisitos de dados extremamente baixos **(mínimo de 1 amostra por locutor)** e velocidade de treinamento extremamente rápida **(mínimo de 100 passos, 2 min em uma GPU T4)**!

A **conversão de voz em tempo real** é suportada, com um atraso de algoritmo de ~300ms e um atraso do lado do dispositivo de ~100ms, adequado para reuniões online, jogos e transmissões ao vivo.

Para encontrar uma lista de demonstrações e comparações com modelos de conversão de voz anteriores, visite nossa [página de demonstração](https://plachtaa.github.io/seed-vc/)🌐 e nossa [página de Avaliação](EVAL.md)📊.

Estamos continuamente melhorando a qualidade do modelo e adicionando mais recursos.

## Avaliação📊
Consulte [EVAL.md](EVAL.md) para resultados de avaliação objetiva e comparações com outras baselines.

## Instalação📥
Suporte mínimo: Python 3.12. Foco em GPU (CUDA).

- Se quiser rodar em ambiente local, use:
  ```bash
  pip install -r requirements.txt
  ```
  Observação: o `requirements.txt` NÃO instala `torch/torchvision/torchaudio`. Para GPU, recomendamos usar Docker abaixo (instala as wheels CUDA corretas). Para ambiente local fora de Docker, instale PyTorch CUDA manualmente conforme sua GPU/driver.

- Requisitos de sistema para I/O de áudio:
  - `ffmpeg` para reamostragem/conversão (obrigatório se você usar as flags de pré-processamento abaixo).
  - `libsndfile` (instalado automaticamente com `soundfile`).

Requisitos de sistema para I/O de áudio:
- `ffmpeg` para reamostragem/conversão (obrigatório se você usar as flags de pré-processamento abaixo).
- `libsndfile` (instalado automaticamente com `soundfile`).

Para usuários de Windows, você pode considerar instalar `triton-windows` para habilitar o uso de `--compile`, o que acelera os modelos V2:
```bash
pip install triton-windows==3.2.0.post13
```

## Uso🛠️
Lançamos 4 modelos para diferentes finalidades:

| Versão | Nome                                                                                                                                                                                                                       | Finalidade                     | Taxa de Amostragem | Codificador de Conteúdo                                                | Vocoder | Dim Oculta | N Camadas | Parâmetros         | Observações                                            |
|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|--------------------|------------------------------------------------------------------------|---------|------------|-----------|--------------------|--------------------------------------------------------|
| v1.0   | seed-uvit-tat-xlsr-tiny ([🤗](https://huggingface.co/Plachta/Seed-VC/blob/main/DiT_uvit_tat_xlsr_ema.pth)[📄](configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml))                                                     | Conversão de Voz (VC)          | 22050              | XLSR-large                                                             | HIFT    | 384        | 9         | 25M                | adequado para conversão de voz em tempo real           |
| v1.0   | seed-uvit-whisper-small-wavenet ([🤗](https://huggingface.co/Plachta/Seed-VC/blob/main/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth)[📄](configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml)) | Conversão de Voz (VC)          | 22050              | Whisper-small                                                          | BigVGAN | 512        | 13        | 98M                | adequado para conversão de voz offline                 |
| v1.0   | seed-uvit-whisper-base ([🤗](https://huggingface.co/Plachta/Seed-VC/blob/main/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth)[📄](configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml))       | Conversão de Voz de Canto (SVC) | 44100              | Whisper-small                                                          | BigVGAN | 768        | 17        | 200M               | forte desempenho zero-shot, conversão de voz de canto  |
| v2.0   | hubert-bsqvae-small ([🤗](https://huggingface.co/Plachta/Seed-VC/blob/main/v2)[📄](configs/v2/vc_wrapper.yaml))                                                                                                            | Conversão de Voz e Sotaque (VC) | 22050              | [ASTRAL-Quantization](https://github.com/Plachtaa/ASTRAL-quantization) | BigVGAN | 512        | 13        | 67M(CFM) + 90M(AR) | Melhor em suprimir características do locutor original |

Os checkpoints do lançamento mais recente do modelo serão baixados automaticamente na primeira execução da inferência.
Se você não conseguir acessar o Hugging Face por motivos de rede, tente usar um espelho adicionando `HF_ENDPOINT=https://hf-mirror.com` antes de cada comando.

Inferência por linha de comando:
```bash
python inference.py --source <audio-de-origem> \
--target <audio-de-referencia> \
--output <diretorio-de-saida> \
--diffusion-steps 25 # recomendado 30~50 para conversão de voz de canto
--length-adjust 1.0 \
--inference-cfg-rate 0.7 \
--f0-condition False # defina como True para conversão de voz de canto
--auto-f0-adjust False # defina como True para ajustar automaticamente o tom da origem para o nível do tom do alvo, normalmente não usado em conversão de voz de canto
--semi-tone-shift 0 # deslocamento de tom em semitons para conversão de voz de canto
--checkpoint <caminho-para-checkpoint> \
--config <caminho-para-config> \
 --fp16 True
```
onde:
- `source`: caminho para o arquivo de áudio a ser convertido para a voz de referência
- `target`: caminho para o arquivo de áudio como referência de voz
- `output`: caminho para o diretório de saída
- `diffusion-steps`: número de passos de difusão a serem usados, padrão é 25, use 30-50 para melhor qualidade, use 4-10 para inferência mais rápida
- `length-adjust`: fator de ajuste de duração, padrão é 1.0, defina <1.0 para acelerar a fala, >1.0 para desacelerar
- `inference-cfg-rate`: tem uma diferença sutil na saída, padrão é 0.7
- `f0-condition`: flag para condicionar o tom da saída ao tom do áudio de origem, padrão é False, defina como True para conversão de voz de canto
- `auto-f0-adjust`: flag para ajustar automaticamente o tom da origem para o nível do tom do alvo, padrão é False, normalmente não usado em conversão de voz de canto
- `semi-tone-shift`: deslocamento de tom em semitons para conversão de voz de canto, padrão é 0
- `checkpoint`: caminho para o checkpoint do modelo se você treinou ou ajustou seu próprio modelo, deixe em branco para baixar automaticamente o modelo padrão do Hugging Face (`seed-uvit-whisper-small-wavenet` se `f0-condition` for `False`, senão `seed-uvit-whisper-base`)
- `config`: caminho para a configuração do modelo se você treinou ou ajustou seu próprio modelo, deixe em branco para baixar automaticamente a configuração padrão do Hugging Face
- `fp16`: flag para usar inferência em float16, padrão é True
- `preprocess-source-ffmpeg`: se True, reamostra o source para 22.05 kHz mono via ffmpeg antes da inferência (padrão False)
- `preprocess-target-ffmpeg`: se True, reamostra o target para 22.05 kHz mono via ffmpeg antes da inferência (padrão False)

Nota:
- A V1 depende do Descript Audio Codec (DAC). Já incluímos `descript-audio-codec==1.0.0` em `requirements-py313.txt`.
- Se habilitar as flags de pré-processamento, garanta `ffmpeg` instalado no sistema ou use a imagem Docker fornecida (já inclui ffmpeg).

## Docker (GPU, Python 3.12)
Recomendado para produção/servidor com GPU. A imagem `Dockerfile.gpu` instala Python 3.12, PyTorch CUDA 12.1 (torch/vision/audio) e as dependências de projeto do `requirements.txt`.

### Construir a imagem (GPU)
```bash
docker build -f Dockerfile.gpu -t seed-vc:gpu .
```

### Rodar inferência V1 (exemplo)
```bash
docker run --rm --gpus all -v "$PWD:/app" seed-vc:gpu \
  python inference.py \
    --source examples/source/source_s1.wav \
    --target examples/reference/s1p1.wav \
    --output output \
    --diffusion-steps 30 \
    --inference-cfg-rate 0.7 \
    --length-adjust 1.0 \
    --f0-condition False \
    --auto-f0-adjust False
```

### docker compose (GPU)
```bash
docker compose build
docker compose run --rm seed-vc \
  python inference.py \
    --source examples/source/source_s1.wav \
    --target examples/reference/s1p1.wav \
    --output output \
    --diffusion-steps 30 \
    --inference-cfg-rate 0.7 \
    --length-adjust 1.0 \
    --f0-condition False \
    --auto-f0-adjust False
```

Notas:
- O projeto usa um ÚNICO `requirements.txt` compartilhado. As bibliotecas `torch/torchvision/torchaudio` são instaladas no `Dockerfile.gpu` com CUDA 12.1, e por isso não aparecem no `requirements.txt`.
- Entrada/saída de áudio via `soundfile`/`libsndfile`; `ffmpeg` já vem instalado na imagem para reamostragem/conversão.
- Ao montar o repositório com `-v "$PWD:/app"`, os arquivos gerados aparecem no seu diretório local `output/`.

Da mesma forma, para usar o modelo V2, você pode executar:
```bash
python inference_v2.py --source <audio-de-origem> \
--target <audio-de-referencia> \
--output <diretorio-de-saida> \
--diffusion-steps 25 # recomendado 30~50 para conversão de voz de canto
--length-adjust 1.0 # igual ao V1
--intelligibility-cfg-rate 0.7 # controla quão claro é o conteúdo linguístico da saída, recomendado 0.0~1.0
--similarity-cfg-rate 0.7 # controla quão semelhante a voz de saída é à voz de referência, recomendado 0.0~1.0
--convert-style true # se deve usar o modelo AR para conversão de sotaque e emoção, definir como false fará apenas a conversão de timbre, semelhante ao V1
--anonymization-only false # definir como true ignorará o áudio de referência, mas apenas anonimizará a fala de origem para uma "voz média"
--top-p 0.9 # controla a diversidade da saída do modelo AR, recomendado 0.5~1.0
--temperature 1.0 # controla a aleatoriedade da saída do modelo AR, recomendado 0.7~1.2
--repetition-penalty 1.0 # penaliza a repetição da saída do modelo AR, recomendado 1.0~1.5
--cfm-checkpoint-path <caminho-para-checkpoint-cfm> # caminho para o checkpoint do modelo CFM, deixe em branco para baixar automaticamente o modelo padrão do Hugging Face
--ar-checkpoint-path <caminho-para-checkpoint-ar> # caminho para o checkpoint do modelo AR, deixe em branco para baixar automaticamente o modelo padrão do Hugging Face
```

## Interface Web de Conversão de Voz (V1)
```bash
python app_vc.py --checkpoint <caminho-para-checkpoint> --config <caminho-para-config> --fp16 True
```
- `checkpoint`: caminho para o checkpoint do modelo se você treinou ou ajustou seu próprio modelo, deixe em branco para baixar automaticamente o modelo padrão do Hugging Face (`seed-uvit-whisper-small-wavenet`)
- `config`: caminho para a configuração do modelo se você treinou ou ajustou seu próprio modelo, deixe em branco para baixar automaticamente a configuração padrão do Hugging Face

Em seguida, abra o navegador e acesse `http://localhost:7860/` para usar a interface web.

## Interface Web de Conversão de Voz de Canto (V1‑f0)
```bash
python app_svc.py --checkpoint <caminho-para-checkpoint> --config <caminho-para-config> --fp16 True
```
- `checkpoint`: caminho para o checkpoint do modelo se você treinou ou ajustou seu próprio modelo, deixe em branco para baixar automaticamente o modelo padrão do Hugging Face (`seed-uvit-whisper-base`)
- `config`: caminho para a configuração do modelo se você treinou ou ajustou seu próprio modelo, deixe em branco para baixar automaticamente a configuração padrão do Hugging Face

## Interface Web do modelo V2
```bash
python app_vc_v2.py --cfm-checkpoint-path <caminho-para-checkpoint-cfm> --ar-checkpoint-path <caminho-para-checkpoint-ar>
```
- `cfm-checkpoint-path`: caminho para o checkpoint do modelo CFM, deixe em branco para baixar automaticamente o modelo padrão do Hugging Face
- `ar-checkpoint-path`: caminho para o checkpoint do modelo AR, deixe em branco para baixar automaticamente o modelo padrão do Hugging Face
- você pode considerar adicionar `--compile` para obter uma aceleração de ~6x na inferência do modelo AR

## Interface Web Integrada
```bash
python app.py --enable-v1 --enable-v2
```
Isso carregará apenas modelos pré-treinados para inferência zero-shot. Para usar checkpoints personalizados, execute `app_vc.py` ou `app_svc.py` como acima.
Se você tiver memória limitada, remova `--enable-v2` ou `--enable-v1` para carregar apenas um dos conjuntos de modelos.

GUI de conversão de voz em tempo real:
```bash
python real-time-gui.py --checkpoint-path <caminho-para-checkpoint> --config-path <caminho-para-config>
```
- `checkpoint`: caminho para o checkpoint do modelo se você treinou ou ajustou seu próprio modelo, deixe em branco para baixar automaticamente o modelo padrão do Hugging Face (`seed-uvit-tat-xlsr-tiny`)
- `config`: caminho para a configuração do modelo se você treinou ou ajustou seu próprio modelo, deixe em branco para baixar automaticamente a configuração padrão do Hugging Face

> [!IMPORTANT]
> É altamente recomendável usar uma GPU para conversão de voz em tempo real.
> Alguns testes de desempenho foram feitos em uma GPU NVIDIA RTX 3060 Laptop, os resultados e as configurações de parâmetros recomendadas estão listados abaixo:

| Configuração do Modelo          | Passos de Difusão | Taxa de CFG de Inferência | Comprimento Máx. do Prompt | Tempo de Bloco (s) | Duração do Crossfade (s) | Contexto Extra (esquerda) (s) | Contexto Extra (direita) (s) | Latência (ms) | Tempo de Inferência por Bloco (ms) |
|---------------------------------|-------------------|---------------------------|----------------------------|--------------------|--------------------------|-------------------------------|------------------------------|---------------|------------------------------------|
| seed-uvit-xlsr-tiny             | 10                | 0.7                       | 3.0                        | 0.18s              | 0.04s                    | 2.5s                          | 0.02s                        | 430ms         | 150ms                              |

Você pode ajustar os parâmetros na GUI de acordo com o desempenho do seu próprio dispositivo, o fluxo de conversão de voz deve funcionar bem desde que o Tempo de Inferência seja menor que o Tempo de Bloco.
Observe que a velocidade de inferência pode diminuir se você estiver executando outras tarefas intensivas de GPU (por exemplo, jogos, assistir a vídeos).

Explicações para os parâmetros da GUI de conversão de voz em tempo real:
- `Diffusion Steps`: número de passos de difusão a serem usados, no caso de tempo real, geralmente definido como 4~10 para a inferência mais rápida;
- `Inference CFG Rate`: tem uma diferença sutil na saída, o padrão é 0.7, definir como 0.0 ganha cerca de 1.5x de aceleração;
- `Max Prompt Length`: comprimento máximo do áudio de prompt, definir um valor baixo pode acelerar a inferência, mas pode reduzir a semelhança com a fala do prompt;
- `Block Time`: duração de cada bloco de áudio para inferência, quanto maior o valor, maior a latência, observe que este valor deve ser maior que o tempo de inferência por bloco, defina de acordo com a condição do seu hardware;
- `Crossfade Length`: duração do crossfade entre os blocos de áudio, normalmente não precisa ser alterado;
- `Extra context (left)`: duração do contexto histórico extra para inferência, quanto maior o valor, maior o tempo de inferência, mas pode aumentar a estabilidade;
- `Extra context (right)`: duração do contexto futuro extra para inferência, quanto maior o valor, maior o tempo de inferência e a latência, mas pode aumentar a estabilidade;

O atraso do algoritmo é calculado aproximadamente como `Tempo de Bloco * 2 + Contexto extra (direita)`, o atraso do lado do dispositivo é geralmente de ~100ms. O atraso geral é a soma dos dois.

Você pode usar o [VB-CABLE](https://vb-audio.com/Cable/) para rotear o áudio do fluxo de saída da GUI para um microfone virtual.

*(A GUI e a lógica de fragmentação de áudio foram modificadas do [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI), obrigado pela implementação brilhante!)*

## Treinamento (resumo)
O fine-tuning com dados personalizados permite que o modelo clone a voz de alguém com mais precisão. Isso melhorará muito a semelhança do locutor em locutores específicos, mas pode aumentar ligeiramente a WER (Taxa de Erro de Palavra).
Um Tutorial do Colab está aqui para você seguir: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1R1BJTqMsTXZzYAVx3j1BiemFXog9pbQG?usp=sharing)

1. Prepare seu próprio conjunto de dados. Ele deve satisfazer o seguinte:
    - A estrutura dos arquivos não importa
    - Cada arquivo de áudio deve ter entre 1 e 30 segundos, caso contrário, será ignorado
    - Todos os arquivos de áudio devem estar em um dos seguintes formatos: `.wav` `.flac` `.mp3` `.m4a` `.opus` `.ogg`
    - A etiqueta do locutor não é necessária, mas certifique-se de que cada locutor tenha pelo menos 1 amostra
    - Claro, quanto mais dados você tiver, melhor será o desempenho do modelo
    - Os dados de treinamento devem ser o mais limpos possível, música de fundo ou ruído não são desejados
2. Escolha um arquivo de configuração de modelo de `configs/presets/` para fine-tuning, ou crie o seu próprio para treinar do zero.
    - Para fine-tuning, deve ser um dos seguintes:
        - `./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml` para conversão de voz em tempo real
        - `./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml` para conversão de voz offline
        - `./configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml` para conversão de voz de canto
3. Execute o seguinte comando para iniciar o treinamento:
```bash
python train.py \
--config <caminho-para-config> \
--dataset-dir <caminho-para-dados> \
--run-name <nome-da-execucao> \
--batch-size 2 \
--max-steps 1000 \
--max-epochs 1000 \
--save-every 500 \
--num-workers 0
```
onde:
- `config`: caminho para a configuração do modelo, escolha um dos acima para fine-tuning ou crie o seu próprio para treinar do zero
- `dataset-dir`: caminho para o diretório do conjunto de dados, que deve ser uma pasta contendo todos os arquivos de áudio
- `run-name`: nome da execução, que será usado para salvar os checkpoints e logs do modelo
- `batch-size`: tamanho do lote para treinamento, escolha dependendo da memória da sua GPU.
- `max-steps`: número máximo de passos para treinar, escolha dependendo do tamanho do seu conjunto de dados e do tempo de treinamento
- `max-epochs`: número máximo de épocas para treinar, escolha dependendo do tamanho do seu conjunto de dados e do tempo de treinamento
- `save-every`: número de passos para salvar o checkpoint do modelo
- `num-workers`: número de workers para carregamento de dados, defina como 0 para Windows

Da mesma forma, para treinar o modelo V2, você pode executar: (observe que o script de treinamento V2 suporta treinamento multi-GPU)
```bash
accelerate launch train_v2.py \
--dataset-dir <caminho-para-dados> \
--run-name <nome-da-execucao> \
--batch-size 2 \
--max-steps 1000 \
--max-epochs 1000 \
--save-every 500 \
--num-workers 0 \
--train-cfm
```

4. Se o treinamento parar acidentalmente, você pode retomá-lo executando o mesmo comando novamente, o treinamento continuará do último checkpoint. (Certifique-se de que os argumentos `run-name` e `config` sejam os mesmos para que o último checkpoint possa ser encontrado)

5. Após o treinamento, você pode usar o modelo treinado para inferência, especificando o caminho para o checkpoint e o arquivo de configuração.
    - Eles devem estar em `./runs/<nome-da-execucao>/`, com o checkpoint nomeado `ft_model.pth` e o arquivo de configuração com o mesmo nome do arquivo de configuração de treinamento.
    - Você ainda precisa especificar um arquivo de áudio de referência do locutor que deseja usar durante a inferência, semelhante ao uso zero-shot.

## TODO📝
- [x] Lançar código
- [x] Lançar modelos pré-treinados: [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SeedVC-blue)](https://huggingface.co/Plachta/Seed-VC)
- [x] Demo no Hugging Face Space: [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)
- [x] Página de demonstração HTML: [Demo](https://plachtaa.github.io/seed-vc/)
- [x] Inferência em streaming
- [x] Reduzir a latência da inferência em streaming
- [x] Vídeo de demonstração para conversão de voz em tempo real
- [x] Conversão de voz de canto
- [x] Resiliência a ruído para áudio de origem
- [ ] Melhorias potenciais na arquitetura
    - [x] Conexões de salto no estilo U-ViT
    - [x] Mudança da entrada para OpenAI Whisper
    - [x] Tempo como Token
- [x] Código para treinamento com dados personalizados
- [x] Fine-tuning de locutor few-shot/one-shot
- [x] Mudança para BigVGAN da NVIDIA para decodificação de voz de canto
- [x] Modelo de versão Whisper para conversão de voz de canto
- [x] Avaliação objetiva e comparação com RVC/SoVITS para conversão de voz de canto
- [x] Melhorar a qualidade do áudio
- [ ] Vocoder NSF para melhor conversão de voz de canto
- [x] Corrigir artefato de conversão de voz em tempo real ao não falar (feito adicionando um modelo VAD)
- [x] Notebook Colab para exemplo de fine-tuning
- [x] Substituir o Whisper por um extrator de conteúdo linguístico mais avançado
- [ ] Mais a ser adicionado
- [x] Adicionar suporte para Apple Silicon
- [ ] Lançar artigo, avaliações e página de demonstração para o modelo V2

## Problemas Conhecidos
- No Mac - executar `real-time-gui.py` pode gerar um erro `ModuleNotFoundError: No module named '_tkinter'`, neste caso, uma nova versão do Python **com suporte a Tkinter** deve ser instalada. Consulte [Este Guia no Stack Overflow](https://stackoverflow.com/questions/76105218/why-does-tkinter-or-turtle-seem-to-be-missing-or-broken-shouldnt-it-be-part) para uma explicação do problema e uma correção detalhada.


## CHANGELOGS🗒️
- 2024-04-16
    - Lançado modelo V2 para conversão de voz e sotaque, com melhor anonimização do locutor de origem
- 2025-03-03:
    - Adicionado suporte para Mac Série M (Apple Silicon)
- 2024-11-26:
    - Atualizado modelo pré-treinado da versão v1.0 tiny, otimizado para conversão de voz em tempo real
    - Suporte para fine-tuning de um ou múltiplos locutores one-shot/few-shot
    - Suporte para uso de checkpoint personalizado para WebUI e GUI em tempo real
- 2024-11-19:
    - Artigo do arXiv lançado
- 2024-10-28:
    - Atualizado modelo de conversão de voz de canto de 44k com fine-tuning e melhor qualidade de áudio
- 2024-10-27:
    - Adicionada GUI de conversão de voz em tempo real
- 2024-10-25:
    - Adicionados resultados de avaliação exaustivos e comparações com RVCv2 para conversão de voz de canto
- 2024-10-24:
    - Atualizado modelo de conversão de voz de canto de 44kHz, com OpenAI Whisper como entrada de conteúdo de fala
- 2024-10-07:
    - Atualizado modelo pré-treinado v0.3, alterado o codificador de conteúdo de fala para OpenAI Whisper
    - Adicionados resultados de avaliação objetiva para o modelo pré-treinado v0.3
- 2024-09-22:
    - Atualizado modelo de conversão de voz de canto para usar BigVGAN da NVIDIA, proporcionando grande melhoria para vozes de canto agudas
    - Suporte para fragmentação e saída em streaming para arquivos de áudio longos na Web UI
- 2024-09-18:
    - Atualizado modelo condicionado a f0 para conversão de voz de canto
- 2024-09-14:
    - Atualizado modelo pré-treinado v0.2, com tamanho menor e menos passos de difusão para alcançar a mesma qualidade, e capacidade adicional de controlar a preservação da prosódia
    - Adicionado script de inferência por linha de comando
    - Adicionadas instruções de instalação e uso

## Agradecimentos🙏
- [Amphion](https://github.com/open-mmlab/Amphion) por fornecer recursos computacionais e inspiração!
- [Vevo](https://github.com/open-mmlab/Amphion/tree/main/models/vc/vevo) pela base teórica do modelo V2
- [MegaTTS3](https://github.com/bytedance/MegaTTS3) pela implementação da inferência CFG multi-condição no modelo V2
- [ASTRAL-quantiztion](https://github.com/Plachtaa/ASTRAL-quantization) pelo incrível tokenizador de fala com desentrelaçamento de locutor usado pelo modelo V2
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) pela base da conversão de voz em tempo real
- [SEED-TTS](https://arxiv.org/abs/2406.02430) pela ideia inicial
