# NVIDIA: checkpoint NVIDIA-001

- GPU-SCOPE-001 approvato: solo NVIDIA tramite nvidia-smi
  (modello, VRAM totale e quantita), sola lettura.
- Implementazione candidata pronta, test sintetici e promozioni
  ancora pendenti; nessuna certificazione driver/CUDA/llama.cpp.
- Stato utility assente distinguibile da query vuota o fallimento,
  senza interpretare questi esiti come prova di GPU fisica assente.
- AMD/Intel separati e non implementati (DATA-001).
