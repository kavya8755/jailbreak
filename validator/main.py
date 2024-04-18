import asyncio
import os
from typing import Any, Callable, Dict, Optional

import nest_asyncio
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from huggingface_hub import AsyncInferenceClient

nest_asyncio.apply()


async def is_instruction(client: AsyncInferenceClient, inputs: list[str]) -> list[bool]:
    """Checks if the input is an instruction.

    Args:
        client (AsyncInferenceClient): 🤗 AsyncClient to use for text classification.
        inputs (list[str]): The inputs to check.

    Returns:
        list[bool]: A list of booleans indicating if each input is an instruction.
    """

    def convert_to_bool(result: dict) -> bool:
        return result["label"] == "LABEL_1"

    tasks = [client.text_classification(input) for input in inputs]
    results = await asyncio.gather(*tasks)
    results = [convert_to_bool(result) for result in results]
    return results


@register_validator(name="guardrails/detect_many_shot_jailbreak", data_type="string")
class DetectManyShotJailbreak(Validator):
    """Validates that {fill in how you validator interacts with the passed value}.

    **Key Properties**

    | Property                      | Description                               |
    | ----------------------------- | ----------------------------------------- |
    | Name for `format` attribute   | `guardrails/detect_many_shot_jailbreak`   |
    | Supported data types          | `string`                                  |
    | Programmatic fix              | None                                      |

    Args:
        num_few_shot_examples (int): The threshold for the number of few-shot examples
                to consider an input a many-shot jailbreak.
        on_fail (str): Action to perform if the input is detected as a many-shot jailbreak.

        api_endpoint (str): The API endpoint to use for the text classification model.
    """  # noqa

    def __init__(
        self,
        num_few_shot_examples: int = 64,
        on_fail: Optional[Callable] = None,
        api_endpoint: str = None,
    ):
        super().__init__(on_fail=on_fail, num_few_shot_examples=num_few_shot_examples)
        self.num_few_shot_examples = num_few_shot_examples
        if not api_endpoint:
            raise ValueError("API endpoint must be provided!")

        self.classification_client = AsyncInferenceClient(
            model=api_endpoint, token=os.environ["HF_API_KEY"], timeout=30
        )

        # Awaken client
        self.classification_client.text_classification(input)

    async def detect_many_shot_jailbreak(self, value: Any) -> bool:
        """Detects if the input contains a many-shot jailbreak.

        Args:
            value (Any): The input to check for a many-shot jailbreak.

        Returns:
            bool: True if the input contains a many-shot jailbreak, False otherwise.
        """

        # Split input, with the assumption that each line is a separate shot.
        lines = value.split("\n")

        # Batch check if each line is a question or an instruction
        results = await is_instruction(self.classification_client, lines)

        # Count number of instructions
        num_instructions = sum(int(result) for result in results)
        if num_instructions >= self.num_few_shot_examples:
            return True

        return False

    def validate(self, value: Any, metadata: Dict = {}) -> ValidationResult:
        """Validates that the input doesn't contain a many-shot jailbreak."""
        if len(value) >= 14500:
            return FailResult(
                error_message=f"Prompt length of {len(value)} is over a max supported length of 15,000 characters."
            )

        contains_many_shot_jailbreak = asyncio.run(
            self.detect_many_shot_jailbreak(value)
        )

        if contains_many_shot_jailbreak is not None and contains_many_shot_jailbreak:
            return FailResult(
                error_message="The input contains a many-shot jailbreak. "
                "Please ensure that the input is not a many-shot jailbreak.",
            )
        return PassResult()
