import bench_openai as bo


def test_parse_stream_completions_with_usage():
    ticks = iter([1.0, 1.5, 2.0])
    lines = ['data: {"choices":[{"text":"Hel"}]}', "", 'data: {"choices":[{"text":"lo"}]}',
             'data: {"choices":[{"text":"!"}]}',
             'data: {"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":3}}', "data: [DONE]"]
    t_first, t_end, n, usage = bo.parse_stream(lines, 0.0, clock=lambda: next(ticks))
    assert (t_first, t_end, n) == (1.0, 2.0, 3) and usage["prompt_tokens"] == 1000


def test_parse_stream_chat_deltas_without_usage():
    lines = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}', b'data: {"choices":[{"delta":{"content":"a"}}]}',
             b'data: {"choices":[{"delta":{"content":"b"}}]}', b"data: [DONE]"]
    t_first, _, n, usage = bo.parse_stream(lines, 0.0)
    assert n == 2 and usage is None and t_first is not None


def test_prompt_is_unique_per_call():
    assert bo.prompt_of(100) != bo.prompt_of(100)
