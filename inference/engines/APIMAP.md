# Motori / inference/engines

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ENGINE](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

## Contratto dichiarato

- `open(request) -> InferenceSession [proposta]`

Dipendenze: nessuna.

## Contratti delegati

- [llama.cpp](llama_cpp/APIMAP.md)
- [vLLM](vllm/APIMAP.md)
- [ONNX Runtime](onnx_runtime/APIMAP.md)
- [PyTorch](pytorch/APIMAP.md)
