# Tutorial: Getting Started with rlm-secure

This tutorial walks you through installing rlm-secure, running your first query, and understanding what happens under the hood. By the end, you'll be able to use rlm-secure to answer questions about large text files.

## Prerequisites

- Python 3.11 or later
- An API key for an LLM provider (this tutorial uses OpenAI, but any [litellm-supported provider](https://docs.litellm.ai/docs/providers) works)

## Install rlm-secure

```bash
pip install -e .
```

Or using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Set your API key

```bash
export OPENAI_API_KEY=sk-...
```

For Anthropic, use `ANTHROPIC_API_KEY` instead. rlm-secure uses litellm, so any supported provider works.

## Create a sample data file

Create a file called `sample.log` with some data to query:

```bash
cat > sample.log << 'EOF'
2024-01-15 10:00:01 INFO  user=alice action=login ip=192.168.1.10
2024-01-15 10:00:15 INFO  user=bob action=login ip=192.168.1.11
2024-01-15 10:01:02 ERROR user=alice action=upload msg="file too large"
2024-01-15 10:01:45 INFO  user=charlie action=login ip=192.168.1.12
2024-01-15 10:02:30 INFO  user=alice action=download file=report.pdf
2024-01-15 10:03:00 ERROR user=bob action=upload msg="permission denied"
2024-01-15 10:03:15 INFO  user=alice action=logout
2024-01-15 10:04:00 INFO  user=dave action=login ip=192.168.1.13
2024-01-15 10:05:00 ERROR user=charlie action=delete msg="not found"
2024-01-15 10:06:00 INFO  user=bob action=logout
EOF
```

## Run your first query

```bash
rlm run -q "How many ERROR lines are in this log?" -c sample.log
```

You should see a short answer like:

```
There are 3 ERROR lines in the log.
```

Behind the scenes, rlm-secure didn't just pass the whole file to the LLM. It used the explore/commit protocol: the LLM first peeked at the data to understand its structure, then emitted a plan of operations (like `grep` and `count`) to answer the question.

## Turn on verbose mode

Run the same query with `-v` to see what rlm-secure is doing:

```bash
rlm run -q "How many ERROR lines are in this log?" -c sample.log -v
```

You'll see output like:

```
Model: gpt-4o-mini
Context: 629 chars

[explore 1] slice(0, 200) → "2024-01-15 10:00:01 INFO  user=alice…"
[explore 2] count(lines) → "10"
[commit] grep(ERROR) → count(lines)
Answer: There are 3 ERROR lines in the log.

Time: 4.2s
```

This shows the explore steps (the LLM peeking at the data), the commit plan (the operations it chose), and the final answer.

For a complete machine-readable record of every LLM message and operation, add `--trace`:

```bash
rlm run -q "How many ERROR lines are in this log?" -c sample.log --trace
```

This writes a JSON file to the `traces/` directory that you can inspect after the run. See the [How-to Guides](how-to-guides.md#how-to-trace-execution) for details.

## Try a more complex question

```bash
rlm run -q "Which users had errors, and what were the error messages?" -c sample.log -v
```

The LLM will use `grep` to filter ERROR lines, then synthesize the results into a readable answer.

## Check cache statistics

Every operation result is cached by its content hash. Run the same query twice and the second run will be faster:

```bash
rlm run -q "How many ERROR lines are in this log?" -c sample.log
rlm cache stats
```

You'll see something like:

```
Entries: 4
Size: 1.2 KB
Location: /home/you/.cache/rlm-secure
```

## Clear the cache

```bash
rlm cache clear
```

## Next steps

- Read the [How-to Guides](how-to-guides.md) for specific tasks like changing models, enabling Nix sandboxing, and processing large files
- Read the [Reference](reference.md) for a complete specification of all operations and configuration options
- Read the [Explanation](explanation.md) to understand the design decisions behind rlm-secure
