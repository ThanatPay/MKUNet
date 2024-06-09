when you training model in train.py, you can set "model_path" in line 32 -> "model/[name model].h5" to recieve best model weight
if want to test in test.py, you can set in line 32 -> model.load_weights("model/[name model].h5") to load [name model]
