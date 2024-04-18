# to run these, run 
# make tests

from guardrails import Guard
import pytest
from validator import DetectManyShotJailbreak


guard = Guard.use(
    DetectManyShotJailbreak,
    on="prompt",
    num_few_shot_examples=64
)


def test_pass():
    test_output = "pass"
    result = guard.parse(test_output)
  
    assert result.validation_passed is True
    assert result.validated_output == test_output


def test_fail():
    with pytest.raises(Exception) as exc_info:
        test_output = "fail"
        guard.parse(test_output)
  
    # Assert the exception has your error_message
    assert str(exc_info.value) == "Validation failed for field with errors: {A descriptive but concise error message about why validation failed}"
