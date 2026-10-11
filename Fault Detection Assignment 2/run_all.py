"""Run Tasks 1-5 end to end:  python run_all.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import eda, test_models, train_logistic, train_svm   # noqa: E402
from prepare import prepare                           # noqa: E402

print("== Task 1/2: data + EDA ==");  eda.make_plots(prepare()["df"])
print("== Task 3: logistic regression ==");  train_logistic.main()
print("== Task 4: SVMs ==");  train_svm.main()
print("== Task 5: test set ==");  test_models.main()
