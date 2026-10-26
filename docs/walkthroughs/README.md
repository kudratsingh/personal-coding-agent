# Component walkthroughs

Each core component ships with a reference test and a walkthrough in the same pull request. Pages are added when the code exists.

Use the [walkthrough template](TEMPLATE.md): start with a small worked example, show tensor shapes at each step, explain design choices and failure cases, then identify verification evidence.

Planned topics include tokenization, transformer layers, optimization, training and resume, sampling and key-value caching, isolated execution, and policy updates.

Implemented: [byte-level BPE](tokenizer.md).

Implemented: [token shards and integrity](shards.md).

Implemented: [decoder-only transformer](transformer.md).

Implemented: [loss and optimization](optimizer.md).

Implemented: [training and resume](training-loop.md).

Implemented: [one-pass loader and masked final update](one-pass-loader.md).

Implemented: [whole-document validation windows and batched validation](validation-windows.md).

Implemented: [cached generation](kv-cache.md).

Implemented: [evaluation metrics](metrics.md).

Implemented: [multiple-choice evaluation](multiple-choice.md).

Implemented: [rehearsal data binding](rehearsal-binding.md).

Implemented: [rehearsal run worker](rehearsal-run.md).

Implemented: [chat-v1 conversations, loss masks and code extraction](chat-format.md).

Implemented: [SFT batching, assistant-token normalization and dev loss](sft-batching.md).
