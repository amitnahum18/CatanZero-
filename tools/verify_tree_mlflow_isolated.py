"""Run with python -I to check loading without repository imports."""
import json
from pathlib import Path
import sys
import mlflow
import numpy as np

root=Path(sys.argv[1]).resolve()
tracking=json.loads((root/'tracking.json').read_text())
registration=json.loads((root/'registration.json').read_text())
mlflow.set_tracking_uri(tracking['tracking_uri'])
model=mlflow.pyfunc.load_model(registration['model_uri'])
sample=dict(np.load(root/'isolated_predictions.npz'))
actual=model.predict(sample['raw'])
error=float(np.max(np.abs(actual-sample['expected'])))
assert error<1e-12
(root/'isolated_reload.json').write_text(json.dumps({'status':'passed','max_abs_error':error,'rows':len(actual),
     'mode':'python -I, no repo source imports required','model_uri':registration['model_uri']})+'\n',encoding='utf-8')
print(json.dumps({'status':'passed','max_abs_error':error,'rows':len(actual)}))
