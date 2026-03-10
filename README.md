# Performance Benchmark: No-GIL & Bloom Filter

Questo tool permette di confrontare le prestazioni di esecuzione in Python analizzando l'impatto del Global Interpreter Lock (GIL) e l'efficienza di interrogazione su strutture dati Bloom Filter.

## Modalità di Esecuzione

Il programma accetta tre comandi principali da interfaccia a riga di comando:

### 1. Multiprocessing
bash: ```python main.py process``` 

Esegue i task utilizzando processi separati. È la modalità standard per ottenere parallelismo in Python con GIL attivo, sfruttando la memoria separata per ogni processo.


### 2. Multithreading (No-GIL)
bash: ```python main.py threading```

Esegue i task utilizzando thread condivisi nello stesso spazio di memoria.
Nota: Per osservare un reale incremento di performance rispetto alla modalità standard, deve essere eseguito con un interprete Python 3.13+ compilato con supporto Free-threading (GIL disabilitato).

### 3. Bloom Filter Query

bash: ```python main.py query```

Esegue un test specifico sulle performance di lettura. Valuta la latenza e il throughput delle interrogazioni (query) all'interno del Bloom Filter per verificare l'efficienza della struttura dati.
Nota: Pure in questo caso sono stati usati  dei thread per il parallelismo, è necessario anche in questo caso compilare con un compilatore Python 3.13+ compilato con supporto Free-threading (GIL disabilitato).
Per completezza in questo caso ho aggiunto un controllo aggiuntivo. Se sei con il GIL attivo verrai avvertito e potrai decidere di non continuare l' esecuzione.
