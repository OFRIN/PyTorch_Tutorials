# 1.
import utility.test
print(utility.test.evaluate())
print(utility.test.debug())

from utility.test import evaluate
from utility.test import debug
print(evaluate())

# 2. 
from utility import test
print(test.evaluate())
print(test.debug())

from utility.test import evaluate as eval
print(eval())