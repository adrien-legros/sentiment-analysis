import argparse
import kfp 

from kfp import dsl, Client
from kfp.dsl import Input, Output, Dataset, Model, Metrics, Artifact, ClassificationMetrics
from datetime import datetime

@dsl.component(
    base_image="quay.io/alegros/sentiment-runtime:latest",
    packages_to_install=[],
)
def load_datasets(dataset: Output[Dataset]):
    import os
    import pandas as pd
    
    dataset_url = "https://minio-s3-minio.apps.cluster-5nb6z.5nb6z.sandbox1242.opentlc.com/data/training.1600000.processed.noemoticon.csv"
    raw_dataset=pd.read_csv(dataset_url, encoding="latin_1",  names=['target', 'ids', 'date', 'flag', 'user', 'text'])
    with open(dataset.path, 'w') as f:
        raw_dataset.to_csv(f, index=False)
    
    dataset.metadata["version"] = "1.0"
    dataset.metadata["foo"] = "bar"

@dsl.component(
    base_image="quay.io/alegros/sentiment-runtime:latest",
    packages_to_install=[],
)
def pre_process(
        dataset: Input[Dataset],
        processed: Output[Dataset],
):
    import pandas as pd
    import numpy as np
    import re
    import nltk
    import re
    from nltk.stem import WordNetLemmatizer
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt


    df = pd.read_csv(dataset.path)
    
    df = df.drop("flag", axis = 1)
    df = df.drop("user", axis = 1)
    df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

    inconsistency = df.groupby(df['text'], as_index=False)['target'].mean()
    inconsistency_filter = inconsistency['target'].apply(lambda x: x not in [0,1])
    inconsistency_tweets_unique = inconsistency[inconsistency_filter]['text']
    inconsistency_tweets = df[df.apply(lambda x: x['text'] in inconsistency_tweets_unique.values, axis=1)]
    df = df.drop(inconsistency_tweets.index)
    df = df.drop_duplicates("text")
    nltk.download('stopwords')
    nltk.download('wordnet')

    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
              ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
              ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
              ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
              ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    lemmatizer = WordNetLemmatizer()

    def clean_tweet(tweet):
        tweet = tweet.lower()
        tweet = ' '.join([w for w in tweet.split(" ") if len(w) > 2])
        tweet = re.sub(urlPattern,'URL ',tweet)
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
        tweet = re.sub(userPattern, 'USER ', tweet)
        tweet = re.sub(alphaPattern, " ", tweet)
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        tweet = " ".join([lemmatizer.lemmatize(word) for word in tweet.split() if word not in stop_words])
        return tweet
    
    df['text'] = df["text"].apply(clean_tweet)
    empty_tweets = df[df['text'].apply(lambda x: len(x)==0)]
    df = df.drop(empty_tweets.index)
    df.to_csv(processed.path, index=False)

@dsl.component(
    base_image="quay.io/alegros/sentiment-runtime:latest",
    packages_to_install=[],
)
def logisticregression(
        processed: Input[Dataset],
        lr: Output[Model],
        tag: str
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    
    df = pd.read_csv(processed.path)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["target"],
                                                        test_size = 0.05, random_state = 0)
    pipe = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=500000)), ('model', LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1))])
    pipe.fit(X_train, y_train).score(X_test, y_test)
    from pickle import dump
    with open(lr.path, "wb") as f:
        dump(pipe, f, protocol=5)
    
    def model_Evaluate(model):
        
        # Predict values for Test dataset
        y_pred = model.predict(X_test)
    
        # Print the evaluation metrics for the dataset.
        print(classification_report(y_test, y_pred))
        
        # Compute and plot the Confusion matrix
        cf_matrix = confusion_matrix(y_test, y_pred)
    
        categories  = ['Negative','Positive']
        group_names = ['True Neg','False Pos', 'False Neg','True Pos']
        group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    
        labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
    
        sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                    xticklabels = categories, yticklabels = categories)
    
        plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
        plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
        plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    
    model_Evaluate(pipe)

@dsl.component(
    base_image="quay.io/alegros/sentiment-runtime:latest",
    packages_to_install=[],
)
def lstm(
        processed: Input[Dataset],
        lstm: Output[Model],
        tag: str
):
    import numpy as np
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from sklearn.model_selection import train_test_split
    from gensim.models import Word2Vec
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report    
    
    df = pd.read_csv(processed.path)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                        test_size = 0.05, random_state = 0)
    
    
    Embedding_dimensions = 100
    # Creating Word2Vec training dataset.
    Word2vec_train_data = list(map(lambda x: x.split(), X_train))
    
    # Defining the model and training it.
    word2vec_model = Word2Vec(Word2vec_train_data,
                     vector_size=Embedding_dimensions,
                     workers=8,
                     min_count=5)
    
    print("Vocabulary Length:", len(word2vec_model.wv.key_to_index))
    
    
    input_length = 60
    vocab_length = 60000
    
    tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
    tokenizer.fit_on_texts(X_data)
    tokenizer.num_words = vocab_length
    print("Tokenizer vocab length:", vocab_length)
    X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=input_length)
    X_test  = pad_sequences(tokenizer.texts_to_sequences(X_test) , maxlen=input_length)
    print("X_train.shape:", X_train.shape)
    print("X_test.shape :", X_test.shape)
    embedding_matrix = np.zeros((vocab_length, Embedding_dimensions))
    
    for word, token in tokenizer.word_index.items():
        if word2vec_model.wv.__contains__(word):
            embedding_matrix[token] = word2vec_model.wv.__getitem__(word)
    
    print("Embedding Matrix Shape:", embedding_matrix.shape)
    
    def getModel():
        embedding_layer = Embedding(input_dim = vocab_length,
                                    output_dim = Embedding_dimensions,
                                    weights=[embedding_matrix],
                                    input_length=input_length,
                                    trainable=False)
    
        model = Sequential([
            embedding_layer,
            Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
            Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
            Conv1D(100, 5, activation='relu'),
            GlobalMaxPool1D(),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid'),
        ],
        name="Sentiment_Model")
        return model
    training_model = getModel()
    training_model.summary()
    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
                 EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]
    training_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = training_model.fit(
        X_train, y_train,
        batch_size=4096,
        epochs=12,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )    
    acc,  val_acc  = history.history['accuracy'], history.history['val_accuracy']
    loss, val_loss = history.history['loss'], history.history['val_loss']
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()
    def ConfusionMatrix(y_pred, y_test):
        # Compute and plot the Confusion matrix
        cf_matrix = confusion_matrix(y_test, y_pred)
    
        categories  = ['Negative','Positive']
        group_names = ['True Neg','False Pos', 'False Neg','True Pos']
        group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    
        labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
    
        sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                    xticklabels = categories, yticklabels = categories)
    
        plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
        plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
        plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    print(classification_report(y_test, y_pred))
    training_model.save(lstr.path)


@dsl.pipeline(name="sentiment-analysis")
def sentiment_pipeline(tag: str = "latest"):
    # Pipeline steps
    load_datasets_task = load_datasets()
    pre_process_task = pre_process(dataset=load_datasets_task.outputs["dataset"])
    processed = pre_process_task.outputs["processed"]
    logisticregression_task = logisticregression(processed=processed, tag=tag)
    lstm_task = lstm(processed=processed, tag=tag)

if __name__ == '__main__':
    host = "https://ds-pipeline-dspa.edf:8888"
    parser = argparse.ArgumentParser(
                        prog='Model.py',
                        description='')
    parser.add_argument('-t', '--tag')
    args = parser.parse_args()
    tag = args.tag
    now = str(datetime.now())
    if not tag:
        tag = now
    kfp.compiler.Compiler().compile(
        pipeline_func=sentiment_pipeline,
        package_path='sentiment-pipeline.yaml',
        pipeline_parameters={'tag': tag},
    )
    client = Client(host=host, verify_ssl=False)
    pipeline_name = "Sentiment Pipeline"
    try:
        pipeline = client.upload_pipeline(pipeline_package_path='sentiment-pipeline.yaml', pipeline_name=pipeline_name, description="Sentiment Pipeline")
        print(pipeline)
        pipeline_id = pipeline.pipeline_id
        pipeline_versions = client.list_pipeline_versions(pipeline_id=pipeline_id)
        pipeline_version_id = pipeline_versions.pipeline_versions[0].pipeline_version_id
    except Exception as e:
        print(f"Exception raised: {e}")
        pipeline_id = client.get_pipeline_id(name=pipeline_name)
        pipeline_version = client.upload_pipeline_version(pipeline_id=pipeline_id, pipeline_version_name=now, pipeline_package_path='sentiment-pipeline.yaml')
        pipeline_version_id = pipeline_version.pipeline_version_id
    print(f"Pipeline Id: {pipeline_id}")
    print(f"Pipeline Version Id: {pipeline_version_id}")
    experiment = client.create_experiment(name="sentiment-pipeline", description="Sentiment Pipeline")
    experiment_id = experiment.experiment_id
    print(f"Experiment Id: {experiment_id}")
    pipeline_run = client.run_pipeline(job_name=tag, pipeline_id=pipeline_id, experiment_id=experiment_id, version_id=pipeline_version_id, enable_caching=True)
    print(pipeline_run)