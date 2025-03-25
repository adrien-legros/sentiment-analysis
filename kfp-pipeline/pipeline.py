import argparse
import kfp 

from kfp import dsl, Client
from kfp.dsl import Input, Output, Dataset, Model, Metrics, Artifact, ClassificationMetrics
from kfp.kubernetes import add_toleration
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
    from nltk.corpus import stopwords

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
    nltk.download('wordnet')
    nltk.download('stopwords')

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
    stop_words = set(stopwords.words('english'))
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
        classification_metrics: Output[ClassificationMetrics],
        metrics: Output[Metrics],
        tag: str
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
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
    y_pred = pipe.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories  = ['Negative','Positive']
    classification_metrics.log_confusion_matrix(categories, cf_matrix.tolist())
    f1 = f1_score(y_test, y_pred)
    metrics.log_metric("f1_score", f1)
    lr.metadata["model_name"] = "Logistic Regression"
    lr.metadata["model_format"] = "pipeline"
    lr.metadata["model_format_version"] = "1"
    print(classification_report(y_test, y_pred))

@dsl.component(
    base_image="quay.io/alegros/sentiment-runtime:latest",
    packages_to_install=[],
)
def lstm(
        processed: Input[Dataset],
        lstm: Output[Model],
        classification_metrics: Output[ClassificationMetrics],
        metrics: Output[Metrics],
        tag: str
):
    import numpy as np
    import pandas as pd
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from sklearn.model_selection import train_test_split
    from gensim.models import Word2Vec
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    
    df = pd.read_csv(processed.path)
    X_data, y_data = np.array(df['text']), np.array(df['target'])
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
    # Predicting on the Test dataset.
    y_pred = training_model.predict(X_test)
    # Converting prediction to reflect the sentiment predicted.
    y_pred = np.where(y_pred>=0.5, 1, 0)
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories  = ['Negative','Positive']
    classification_metrics.log_confusion_matrix(categories, cf_matrix.tolist())
    f1 = f1_score(y_test, y_pred)
    metrics.log_metric("f1_score", f1)
    # Printing out the Evaluation metrics.
    print(classification_report(y_test, y_pred))
    training_model.export(lstm.path)
    lstm.metadata["model_name"] = "LSTM"
    lstm.metadata["model_format"] = "keras"
    lstm.metadata["model_format_version"] = "2"


@dsl.component(
    base_image="quay.io/alegros/sentiment-runtime:latest",
    packages_to_install=["model-registry"],
)
def register_model(tag: str, model: Input[Model], metrics: Input[Metrics], classification_metrics: Input[ClassificationMetrics], model_regitry_endpoint: str, user_token: str):
    from model_registry import ModelRegistry
    from datetime import datetime
    # Register model
    model_path=model.path
    registered_model_name = model.metadata["model_name"]
    if not tag:
        tag=datetime.now().strftime('%y%m%d%H%M')
    model_name="sentiment-analysis"
    author_name="data-scientist@redhat.com"
    registry = ModelRegistry(server_address=model_regitry_endpoint, port=443, author=author_name, is_secure=False, user_token=user_token)
    metadata = {
        # "metrics": metrics,
        "license": "apache-2.0",
        "commit": tag,
        "classification_metrics": classification_metrics.path,
        "metrics": metrics.path
    }
    registry.register_model(
        registered_model_name,
        model_path,
        model_format_name=model.metadata["model_format"],
        model_format_version=model.metadata["model_format_version"],
        version=tag,
        description=f"Sentiment Analysis Model version {tag}",
        metadata=metadata
    )
    print("Model registered successfully")


@dsl.pipeline(name="sentiment-analysis")
def sentiment_pipeline(model_regitry_endpoint: str, user_token: str, tag: str = "latest"):
    # Pipeline steps
    load_datasets_task = load_datasets()
    pre_process_task = pre_process(dataset=load_datasets_task.outputs["dataset"])
    processed = pre_process_task.outputs["processed"]
    logisticregression_task = logisticregression(processed=processed, tag=tag)
    lstm_task = lstm(processed=processed, tag=tag)
    lstm_task.set_accelerator_limit("1").set_accelerator_type("nvidia.com/gpu")
    add_toleration(
        lstm_task,
        key="nvidia.com/gpu",
        operator="Exists",
        value=None,
        effect="NoSchedule",
    )
    register_lr_task = register_model(
        tag=tag, model=logisticregression_task.outputs["lr"], metrics=logisticregression_task.outputs["metrics"],
        classification_metrics=logisticregression_task.outputs["classification_metrics"],
        model_regitry_endpoint=model_regitry_endpoint, user_token=user_token)
    register_lstm_task = register_model(
        tag=tag, model=lstm_task.outputs["lstm"], metrics=lstm_task.outputs["metrics"],
        classification_metrics=lstm_task.outputs["classification_metrics"],
        model_regitry_endpoint=model_regitry_endpoint, user_token=user_token)


if __name__ == '__main__':
    host = "https://ds-pipeline-dspa.edf:8888"
    parser = argparse.ArgumentParser(
                        prog='Model.py',
                        description='')
    parser.add_argument('-t', '--tag')
    parser.add_argument('-r', '--model_regitry_endpoint')
    parser.add_argument('--user_token')
    args = parser.parse_args()
    tag = args.tag
    model_regitry_endpoint = args.model_regitry_endpoint
    user_token = args.user_token
    now = str(datetime.now())
    if not tag:
        tag = now
    kfp.compiler.Compiler().compile(
        pipeline_func=sentiment_pipeline,
        package_path='sentiment-pipeline.yaml',
        pipeline_parameters={'tag': tag, 'model_regitry_endpoint': model_regitry_endpoint, 'user_token': user_token},
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