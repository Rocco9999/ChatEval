import os
from groq import Groq

client = Groq(
    # This is the default and can be omitted
    api_key="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",
)

models = client.models.list()

print(models)