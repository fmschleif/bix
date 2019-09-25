# Anleitung
Sämtlicher Code der im Rahmen der BA entwickelt wurde, liegt im Ordner `bix/data/twitter` des bix-repos. Alle python-skripte die sich im 'fetch'-Ordner befinden wurden nicht im Rahmen meiner Bachelorarbeit entwickelt (sondern in meinem hiwi-job), jedoch waren diese essenziel um Daten von Twitter herunterzuladen.

## Requirements
Im Pfad `bix/data/twitter/requirements.txt` liegt eine requirements.txt mit allen für den code benötigten Python-Abhängigkeiten.

## Hinweise
- Alle Python-Skripte erwarten, dass sie im Ordner `bix/data/twitter/data` ausgeführt werden.
- Die meisten Skripte brauchen zimmlich viel RAM (oder VRAM) und laufen nicht auf üblichen Heimrechnern. Diese Skripte wurden auf dem Rechner (mit 2 Titan Grafikkarten) des BIX-Labors ausgeführt.

## Kurze Version (Testen der Ergebnisse)
Diese Anleitung beschriebt wo die erstellten Modelle und andere erstellte Artifakte heruntergeladen werden können und eine Sentimentanalyse auf aktuellen Twitterdaten durchgeführt werden kann.
 
- Herunterladen des data-Ordners (`https://drive.google.com/file/d/1xoN6OTQpxWM4qx8ZVexUnzcEqq6m2QuF/view?usp=sharing`) und verschieben des Ordners an die position `bix/data/twitter/data` im repo.
- Ausführen des Skripts `bix/data/twitter/analysis/analyse_fresh_twitter.py` (Dies wendet die Sentimentanalyse auf die aktuellen von Twitter heruntergeladenen Hashtags #sad und #love an)

## Lange Version (Generieren und lernen der Modelle. Dauert ca. 2 - 4h)
Diese Anleitung beschriebt den Kompletten Ablauf, wie die Daten heruntergeladen werden, das Preprocessing angewand wird, die Embeddings und die Sentimentanalysen trainiert werden und daraufhin auf Daten angewand werden.
- Herunterladen des Trainingsdatensatzes von `https://www.kaggle.com/kazanova/sentiment140` und kopieren des Datensatzes an die Position `bix/data/twitter/data/learn/training.1600000.processed.noemoticon.csv`
- Herunterladen des Pretrained GloVe Embeddnigs von `http://nlp.stanford.edu/data/glove.twitter.27B.zip` und kopieren des Datensatzes an die Position `bix/data/twitter/data/learn/glove.twitter.27B.100d.txt`
- Ausführen des Skriptes `bix/data/twitter/preprocessing/preprocess_learning_data.py` zum Anwenden des Preprocessings
- Ausführen des Skriptes `bix/data/twitter/learn/run_tokenizer.py` zur Tokenisation
- Ausführen des Skriptes `bix/data/twitter/learn/find_skip_grams.py` zum generieren der Skip-gramme
- Ausführen des Skriptes `bix/data/twitter/learn/learn_embeddings.py` mit dem Parameter `skip_gram` zum Erzeugen des Skip-gram Embeddings
- Alternativ zu den letzten beiden Schritten kann das Skip-gram Embedding auch mit dem Skript `bix/data/twitter/helper_scripts/gensim_test.py` erzeugt werden. Diese Implementierung ist besser Optimiert und deutlich schneller
- Ausführen des Skriptes `bix/data/twitter/learn/learn_embeddings.py` mit dem Parameter `glove` zum Erzeugen des GloVe Embeddings
- Ausführen des Skriptes `bix/data/twitter/learn/learn_embeddings.py` mit dem Parameter `word` zum Erzeugen des problemspezifischen Embeddings
- Ausführen des Skriptes `bix/data/twitter/analysis/analyse_sentiment_single.py` mit dem Parameter `skip_gram` zum lernen der Sentimentanalyse die nur das Skip-gram Embedding verwendet
- Ausführen des Skriptes `bix/data/twitter/analysis/analyse_sentiment_single.py` mit dem Parameter `glove` zum lernen der Sentimentanalyse die nur das GloVe Embedding verwendet
- Ausführen des Skriptes `bix/data/twitter/analysis/analyse_sentiment_single.py` mit dem Parameter `word` zum lernen der Sentimentanalyse die nur das problemspezifische Embedding verwendet
- Ausführen des Skriptes `bix/data/twitter/analysis/analyse_sentiment_conv.py` zum lernen der Sentimentanalyse die alle 3 Embeddings verwendet
- Ausführen des Skriptes `bix/data/twitter/fetch/fetch_sentiment_hashtags.py` welches Tweets zu den Hashtags #sad und #love herunterläd
- Ausführen des Skriptes `bix/data/twitter/preprocessing/preprocess.py` zum Anwenden des Preprocessings auf die heruntergeladenen Tweets
- Ausführen des Skripts `bix/data/twitter/analysis/analyse_fresh_twitter.py` (Dies wendet die Sentimentanalyse auf die aktuellen von Twitter heruntergeladenen Hashtags #sad und #love an)



