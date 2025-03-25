Dataset:
````python
import requests
r = requests.get("https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip")
open(dataset_path_zip, 'wb').write(r.content)
```
S3 should contain:
```
/data/raw/dataset.zip
/models/sentiment-analysis/prod/lstm-cpu/
sentiment-analysis/prod/sklearn/lr-pipeline.pkl
```
Runtime image: quay.io/alegros/sentiment-runtime:latest