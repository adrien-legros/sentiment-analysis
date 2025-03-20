Dataset:
````python
import requests
r = requests.get("https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip")
open(dataset_path_zip, 'wb').write(r.content)
```