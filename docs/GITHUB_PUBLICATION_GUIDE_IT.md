# Pubblicazione definitiva FusEurope — GitHub e Zenodo

## Situazione attuale

- Repository privato gia creato: https://github.com/Rlalli74/fuseurope-analysis
- Autore del software: Roberto Lalli.
- Licenza: MIT, copyright 2026 Roberto Lalli, aggiunta direttamente su GitHub come file `LICENSE`.
- Versione software preparata in `pyproject.toml` e `CITATION.cff`: **1.0.0**.
- Il progetto e stato validato privatamente contro **563/563** controlli finali e attraverso **69** test pubblici/sintetici. I dati WoS riservati non si pubblicano.

## 1. Aggiornare la cartella locale

In GitHub Desktop, aprire il repository locale `fuseurope-analysis` e fare **Fetch origin**; quando compare **Pull origin**, cliccarlo. Questo importa il file `LICENSE` che e stato aggiunto tramite il sito GitHub. Prima di apportare modifiche, verificare che `LICENSE` compaia nella cartella locale.

Applicare l'aggiornamento di versione fornito separatamente: cambia solo metadati e documentazione (non gli algoritmi). Controllare in GitHub Desktop che le modifiche riguardino `README.md`, `pyproject.toml`, `CITATION.cff`, file `docs/`, `CHANGELOG.md` e i manifest, oltre al `LICENSE` gia presente.

## 2. Controllare e inviare l'aggiornamento

Dalla cartella locale eseguire un controllo:

```bash
python3 scripts/audit_public_repository.py
```

Per i test sintetici, usare l'ambiente gia predisposto:

```bash
conda run --no-capture-output -n fuseurope-reproducibility python -m pytest -q
```

In GitHub Desktop: **Changes** > verificare i nomi dei file > **Commit to main** con messaggio `Prepare v1.0.0 release metadata and MIT licence documentation` > **Push origin**. Controllare nel browser che `CITATION.cff` e `pyproject.toml` riportino entrambi `1.0.0`.

## 3. Controllo prima dell'apertura al pubblico

Leggere `docs/PUBLICATION_READINESS.md`. Verificare manualmente contenuti e diritti di distribuzione dei dati di correzione; controllare tutti i commit GitHub per escludere che in passato siano stati inclusi dati WoS o altri file riservati. Il solo controllo automatico dei file correnti non basta.

Quando tutto e verificato: GitHub > repository > **Settings** > **General** > **Danger Zone** > **Change repository visibility** > **Make public**. Non rendere pubblico finche permangono dubbi sui diritti o sui dati.

## 4. Abilitare Zenodo PRIMA di pubblicare la release

Accedere a https://zenodo.org/ e collegare il proprio account GitHub. Dal profilo Zenodo scegliere **GitHub**, usare **Sync now** se necessario e attivare l'interruttore per `Rlalli74/fuseurope-analysis`. La documentazione ufficiale e https://help.zenodo.org/docs/github/enable-repository/.

**L'ordine e importante:** attivare la connessione Zenodo prima della prima release GitHub. Non creare la release mentre il repository e ancora privato o l'integrazione non e abilitata.

## 5. Creare e verificare la prima release

Nel repository pubblico GitHub aprire **Releases** > **Draft a new release**. In **Choose a tag** digitare `v1.0.0` e scegliere **Create new tag** a partire da `main`. Title: `FusEurope Analysis v1.0.0`. Nelle note descrivere il codice riproducibile, i controlli superati, e il fatto che il record-level WoS non e redistribuito. Non selezionare **pre-release** per la versione stabile.

Cliccare **Publish release** solo dopo aver verificato il contenuto e la connessione Zenodo. Controllare poi su Zenodo che la release sia stata acquisita, che il software risulti attribuito a Roberto Lalli e che sia stato creato il DOI della versione.

Il riferimento GitHub previsto e `https://github.com/Rlalli74/fuseurope-analysis/releases/tag/v1.0.0`, ma **non va considerato esistente fino a quando la release non viene pubblicata**.

## 6. Citare il codice nel SI e negli articoli

Solo dopo la pubblicazione della release usare l'URL GitHub reale e il DOI Zenodo **effettivamente assegnato**. Per una citazione riproducibile, preferire il DOI specifico della versione `v1.0.0` (non confonderlo con il DOI generale del progetto).

Testo temporaneo in inglese per il SI (da compilare con DOI reale):

> The analytical code is available as FusEurope Analysis, version 1.0.0 (Roberto Lalli), in the GitHub repository https://github.com/Rlalli74/fuseurope-analysis and in the Zenodo archived release [VERSION DOI]. Licensed Web of Science record-level data are not redistributed; full numerical reproduction requires lawful access to the documented source datasets and private input decisions.

Non aggiornare il SI con numeri DOI o date inventati.
