# Pubblicazione di FusEUrope su GitHub (istruzioni per Mac)

**Stato del materiale:** codice e test sintetici pronti per il caricamento in un repository **privato**; NON ancora release scientifica definitivamente validata. Prima di passare a pubblico, leggere `docs/PUBLICATION_READINESS.md` e completare le decisioni su licenza e titolarità del codice.

## Cosa caricare

Usare **soltanto** la cartella `fuseurope-analysis` contenuta nel pacchetto GitHub fornito con questa guida. Non caricare i precedenti ZIP di lavoro, l'archivio `private_workspace`, le esportazioni WoS, il corpus di 45.453 record, gli UT dei cinque eventi esclusi, o i dati di audit privati.

**Non caricare il file ZIP come singolo file nel repository**: decomprimerlo e caricare la cartella come repository. Un archivio GitHub privato non è ancora pubblicazione scientifica.

## Prima di avviare GitHub Desktop

1. Estrarre lo ZIP nel Finder. Aprire `fuseurope-analysis`; la cartella deve contenere `README.md`, `src`, `scripts`, `tests`, `config` e `docs`.
2. Non aggiungere dati o output privati a quella cartella.
3. Nel Terminale, facoltativamente, se Python e le dipendenze sono già installati:

```bash
cd /percorso/alla/cartella/fuseurope-analysis
python scripts/audit_public_repository.py
python -m pytest -q
```

Il primo comando è un controllo automatico di sicurezza (non garantisce da solo la correttezza di licenze e contenuti); il secondo esegue i test sintetici.

## Metodo consigliato: GitHub Desktop, senza usare Git dal Terminale

**Compatibilità Mac:** la versione attuale di GitHub Desktop richiede **macOS 12 o successivo**. Su macOS 10.15/11, oppure se GitHub Desktop non si installa, utilizzare il metodo alternativo indicato dopo questi passaggi.

1. Installare [GitHub Desktop](https://desktop.github.com/) ed effettuare l'accesso al proprio account GitHub.
2. Aprire GitHub Desktop. Se è la prima volta, scegliere **File → Add Local Repository** e selezionare la cartella estratta `fuseurope-analysis`.
3. Se compare la notifica che non si tratta ancora di un repository, scegliere **Create a repository here**. Il nome può essere `fuseurope-analysis`. **Non scegliere una cartella contenente gli archivi privati.**
4. Nella scheda **Changes**, verificare l'elenco dei file. Non devono comparire le cartelle dei dati riservati, grandi CSV bibliografici, PDF di archivio, file `.graphml`, esportazioni WoS o dati originali.
5. Inserire il messaggio **Initial public-source candidate (synthetic tests verified)** e fare **Commit to main**.
6. Premere **Publish repository**. Usare il nome `fuseurope-analysis`; **lasciare attiva l'opzione `Keep this code private`** in questo primo passaggio.
7. Aprire il repository dalla voce **Repository → View on GitHub**. Copiare il suo indirizzo (esempio puramente illustrativo: `https://github.com/ACCOUNT/fuseurope-analysis`).
8. Dopo l'audit finale e l'approvazione della licenza, da GitHub: **Settings → General → Danger Zone → Change repository visibility → Make public**. Verificare nuovamente i file dalla pagina pubblica.

## Se il Mac non supporta GitHub Desktop

Il metodo alternativo più affidabile è Git tramite Terminale, dopo avere creato un **repository nuovo e vuoto, privato**, su [github.com/new](https://github.com/new). Non aggiungere README, `.gitignore` o LICENSE dalla pagina web: li gestiamo nella cartella locale.

Aprire il Terminale dalla cartella **estratta** `fuseurope-analysis` (in Finder: tasto destro sulla cartella, *Nuovo terminale nella cartella*, se disponibile) ed eseguire i comandi, sostituendo `ACCOUNT` con il proprio nome utente GitHub:

```bash
git init
git add .
git status
git commit -m "Initial public-source candidate (synthetic tests verified)"
git branch -M main
git remote add origin https://github.com/ACCOUNT/fuseurope-analysis.git
git push -u origin main
```

**Prima di `git commit`** controllare che l'elenco di `git status` non contenga dati riservati. GitHub può richiedere l'autenticazione tramite il browser, Git Credential Manager o un token: **non incollare password o token dentro file del repository**. Se `git` non è installato, macOS potrebbe proporre di installare gli strumenti della riga di comando.

Si può anche caricare dal sito web (*Add file → Upload files*), ma GitHub limita a **100 file per caricamento**. Il nostro pacchetto contiene 109 file: occorrerebbero almeno due caricamenti, mantenendo la struttura delle cartelle. Per evitare errori consiglio GitHub Desktop (se compatibile) o i comandi sopra.

## Creare la release citabile

1. Dopo la validazione pubblica/privata e la scelta definitiva della versione software, dal repository su GitHub aprire **Releases → Draft a new release**.
2. Creare un tag coerente con `pyproject.toml`: allo stato attuale contiene `0.19.0rc1`, quindi una possibile *pre-release* sarebbe `v0.19.0rc1`. **Non pubblicare una release stabile v1.0.0 senza aggiornare e testare il codice.**
3. Scrivere in inglese cosa contiene la release, quali dati NON sono inclusi e quali test sono stati completati. Se il lavoro resta un candidato, indicare **This is a pre-release**.
4. Fare **Publish release** soltanto dopo aver completato i controlli di `PUBLICATION_READINESS.md`.
5. Copiare l'URL effettivo della release, ad esempio `https://github.com/ACCOUNT/fuseurope-analysis/releases/tag/v0.19.0rc1` (esempio, **non link esistente**).
6. Per ottenere un DOI permanente, collegare il repository a [Zenodo](https://zenodo.org/) e archiviare la release attraverso l'integrazione GitHub–Zenodo, dopo aver verificato che l'archivio non contenga dati riservati e che la licenza sia corretta. Il DOI viene assegnato da Zenodo, non da GitHub.

## SI, articolo e DOI

Solo dopo che il repository sarà stato reso pubblico e la release archiviata, sostituire il segnaposto nella sezione S13 del joint Supplementary Information con **URL reale, tag e, se disponibile, DOI**. Evitare qualunque dichiarazione che faccia passare i test sintetici per una riproduzione completa delle serie WoS.

## Cosa verificare insieme prima di rendere pubblico

- Autore unico del software: Roberto Lalli (`CITATION.cff` gia presente). Verificare i diritti di ridistribuzione e scegliere la licenza; aggiungere il file `LICENSE` prima del rilascio pubblico.
- Il pacchetto non contiene dati WoS e output record-level protetti da licenza.
- La pipeline aggiornata ricostruisce correttamente il corpus e, con i dati concessi in licenza, le statistiche del SI e degli articoli, inclusi gli indicatori geografici C1/RP e i controlli finali separati.
- Il repository non contiene vecchi commit che abbiano mai incluso dati privati. Un file cancellato da un commit successivo non è necessariamente rimosso dalla cronologia Git.

**Documentazione GitHub:** [GitHub Desktop: creare e pubblicare un repository](https://docs.github.com/en/desktop/overview/creating-your-first-repository-using-github-desktop), [visibilità repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility), [creare una release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).
